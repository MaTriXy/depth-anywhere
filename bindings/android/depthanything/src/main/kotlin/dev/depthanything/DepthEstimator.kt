// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yossi Elkrief
package dev.depthanything

import android.graphics.Bitmap
import java.io.Closeable

/**
 * Runs a Depth Anything model (a GGUF from depth-anywhere) on device.
 *
 *     DepthEstimator(modelFile.path).use { estimator ->
 *         estimator.processingSize = 252          // faster, for live video
 *         val depth = estimator.depth(bitmap)
 *         val map = depth.nearnessBitmap()        // brighter is nearer, ready for depth shaders
 *     }
 *
 * [threads] 0 (the default) uses the fastest CPU cluster's cores. The model must be a file on disk (copy it out of assets first). One estimator runs one image
 * at a time; calls from several threads are serialised. Call from a background thread.
 */
public class DepthEstimator(modelPath: String, threads: Int = 0) : Closeable {
    /** What the raw values mean. */
    public enum class OutputKind {
        /** Bigger is farther (DA3 relative models). */
        RelativeDepth,
        /** Bigger is nearer (DA2 relative models). */
        InverseDepth,
        /** Metres from the camera (metric models). */
        MetricDepth,
    }

    private var ctx: Long = nativeLoad(modelPath, threads)
    public val outputKind: OutputKind
    /** The model's own processing size (504 for DA3), restored by setting [processingSize] to null. */
    public val defaultProcessingSize: Int

    init {
        require(ctx != 0L) { "couldn't load the model at $modelPath" }
        outputKind = when (nativeOutputKind(ctx)) {
            1 -> OutputKind.InverseDepth
            2 -> OutputKind.MetricDepth
            else -> OutputKind.RelativeDepth
        }
        defaultProcessingSize = nativeSetResizeTarget(ctx, 0)
    }

    /**
     * The longest side images are resized to before inference (rounded to the model's patch
     * size). Lower is much faster; 252 suits live video. Null restores the default.
     */
    public var processingSize: Int? = null
        @Synchronized set(value) {
            check(ctx != 0L) { "closed" }
            val applied = nativeSetResizeTarget(ctx, value ?: 0)
            field = if (value == null) null else applied
        }

    /** Depth for an ARGB_8888 bitmap (others are converted). */
    @Synchronized
    public fun depth(bitmap: Bitmap): DepthMap {
        check(ctx != 0L) { "closed" }
        val rgba = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val size = IntArray(2)
        val values = nativeDepthBitmap(ctx, rgba, size) ?: error("depth failed: ${nativeLastError(ctx)}")
        if (rgba !== bitmap) rgba.recycle()
        return DepthMap(size[0], size[1], values, outputKind)
    }

    @Synchronized
    override fun close() {
        if (ctx != 0L) nativeFree(ctx)
        ctx = 0L
    }

    private companion object {
        init {
            System.loadLibrary("depthanything_jni")
        }

        @JvmStatic external fun nativeLoad(path: String, threads: Int): Long
        @JvmStatic external fun nativeFree(ctx: Long)
        @JvmStatic external fun nativeSetResizeTarget(ctx: Long, target: Int): Int
        @JvmStatic external fun nativeOutputKind(ctx: Long): Int
        @JvmStatic external fun nativeLastError(ctx: Long): String
        @JvmStatic external fun nativeDepthBitmap(ctx: Long, bitmap: Bitmap, outSize: IntArray): FloatArray?
    }
}

/** A depth map, row-major from the top-left, at the model's processing size. */
public class DepthMap(
    public val width: Int,
    public val height: Int,
    /** Raw values, in the units [kind] describes. */
    public val values: FloatArray,
    public val kind: DepthEstimator.OutputKind,
) {
    /** Nearness before stretching: bigger is nearer, in whatever units the model gives. */
    public fun rawNearness(): FloatArray =
        if (kind == DepthEstimator.OutputKind.InverseDepth) values.copyOf()
        else FloatArray(values.size) { 1f / maxOf(values[it], 1e-6f) }

    /**
     * The 2nd and 98th percentiles of [rawNearness]: the span [nearness] stretches to 0..1. A live
     * view can smooth this between frames so the contrast doesn't pump.
     */
    public fun nearnessRange(): ClosedFloatingPointRange<Float> = percentiles(rawNearness())

    /**
     * 0 (far) .. 1 (near), stretched between the 2nd and 98th percentiles, or across [range] (in
     * [rawNearness] units): the input depth shaders (Depth Parallax, Depth Fog) read.
     */
    public fun nearness(range: ClosedFloatingPointRange<Float>? = null): FloatArray {
        val near = rawNearness()
        if (near.isEmpty()) return near
        val span = range ?: percentiles(near)
        val lo = span.start
        val scale = 1f / maxOf(span.endInclusive - lo, 1e-9f)
        for (i in near.indices) near[i] = ((near[i] - lo) * scale).coerceIn(0f, 1f)
        return near
    }

    private fun percentiles(near: FloatArray): ClosedFloatingPointRange<Float> {
        if (near.isEmpty()) return 0f..1f
        val sorted = near.sortedArray()
        val lo = sorted[((sorted.size - 1) * 0.02f).toInt()]
        val hi = sorted[((sorted.size - 1) * 0.98f).toInt()]
        return lo..maxOf(lo, hi)
    }

    /** [nearness] as an ARGB_8888 grayscale bitmap, brighter nearer. */
    public fun nearnessBitmap(): Bitmap {
        val near = nearness()
        val pixels = IntArray(near.size) {
            val v = (near[it] * 255f + 0.5f).toInt()
            (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
