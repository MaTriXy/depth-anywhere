package dev.depthanything

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On a device. Push a model first:
 *   adb push models/depth-anything-small-f32.gguf /data/local/tmp/da-small.gguf
 */
@RunWith(AndroidJUnit4::class)
class DepthEstimatorTest {
    private val model = File("/data/local/tmp/da-small.gguf")

    @Test
    fun depthFromBitmapAndTiming() {
        assumeTrue("push a model to ${model.path}", model.exists())
        val width = 640
        val height = 480
        // A near-to-far gradient with a bright square, enough structure for a depth model
        val pixels = IntArray(width * height) { i ->
            val x = i % width
            val y = i / width
            val g = 40 + 180 * y / height
            val square = x in 260..380 && y in 300..420
            if (square) 0xFFE0C080.toInt() else (0xFF shl 24) or (g shl 16) or (g shl 8) or (255 - g)
        }
        val bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        DepthEstimator(model.path).use { estimator ->
            assertEquals(504, estimator.defaultProcessingSize)
            for (size in listOf(null, 336, 252)) {
                estimator.processingSize = size
                estimator.depth(bitmap) // warm up
                val t0 = SystemClock.elapsedRealtime()
                val runs = 3
                var depth: DepthMap? = null
                repeat(runs) { depth = estimator.depth(bitmap) }
                val ms = (SystemClock.elapsedRealtime() - t0) / runs
                val d = depth!!
                Log.i("DepthAnything", "size=${size ?: estimator.defaultProcessingSize} -> ${d.width}x${d.height}: $ms ms")
                assertEquals(d.width * d.height, d.values.size)
                assertTrue(maxOf(d.width, d.height) <= (size ?: 504))
                val near = d.nearness()
                assertTrue(near.max() == 1f && near.min() == 0f)
            }
        }
        InstrumentationRegistry.getInstrumentation()
    }
}
