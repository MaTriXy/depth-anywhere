// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yossi Elkrief
import CoreGraphics
import CoreVideo
import DepthAnythingC
import Foundation

/// Runs a Depth Anything model (a GGUF from depth-anywhere) on device.
///
///     let estimator = try DepthEstimator(modelURL: url)
///     estimator.processingSize = 252            // faster, for live video
///     let depth = try estimator.depth(from: photo)
///     let map = depth.nearnessImage()           // brighter is nearer, ready for depth shaders
///
/// One estimator runs one image at a time; calls from several threads are serialised.
public final class DepthEstimator: @unchecked Sendable {
    public enum EstimatorError: Error, CustomStringConvertible {
        case loadFailed(String)
        case inferenceFailed(String)
        case unsupportedPixelFormat(OSType)

        public var description: String {
            switch self {
            case let .loadFailed(m): "Couldn't load the model: \(m)"
            case let .inferenceFailed(m): "Depth failed: \(m)"
            case let .unsupportedPixelFormat(f): "Unsupported pixel format \(f); use 32BGRA or 32RGBA"
            }
        }
    }

    /// What the raw values mean.
    public enum OutputKind: Sendable {
        /// Bigger is farther (DA3 relative models).
        case relativeDepth
        /// Bigger is nearer (DA2 relative models).
        case inverseDepth
        /// Metres from the camera (metric models).
        case metricDepth
    }

    private let ctx: OpaquePointer
    private let lock = NSLock()
    public let outputKind: OutputKind
    /// The model's own processing size (504 for DA3), restored by setting `processingSize` to nil.
    public let defaultProcessingSize: Int

    /// `threads`: CPU threads for ops the GPU doesn't run; 0 picks the performance-core count.
    public init(modelURL: URL, threads: Int = 0) throws {
        #if targetEnvironment(simulator)
        // The simulator's Metal driver can't wrap host memory in GPU buffers, which ggml's Metal
        // backend relies on (it traps). Run on the CPU there; real devices use the GPU.
        setenv("DA_DEVICE", "cpu", 0)
        #endif
        guard let ctx = da_capi_load(modelURL.path, Int32(threads)) else {
            throw EstimatorError.loadFailed(modelURL.lastPathComponent)
        }
        self.ctx = ctx
        switch da_capi_output_kind(ctx) {
        case 1: outputKind = .inverseDepth
        case 2: outputKind = .metricDepth
        default: outputKind = .relativeDepth
        }
        defaultProcessingSize = Int(da_capi_set_resize_target(ctx, 0))
    }

    deinit { da_capi_free(ctx) }

    /// The longest side images are resized to before inference (rounded to the model's patch
    /// size). Lower is much faster; 252 suits live video on a phone. Nil restores the default.
    public var processingSize: Int? {
        get { lock.withLock { currentSize } }
        set {
            lock.withLock {
                // The engine rounds to its patch size; keep what it actually applied
                let applied = Int(da_capi_set_resize_target(ctx, Int32(newValue ?? 0)))
                currentSize = newValue == nil ? nil : applied
            }
        }
    }
    private var currentSize: Int?

    /// Depth for a photo.
    public func depth(from image: CGImage) throws -> DepthMap {
        let width = image.width, height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw EstimatorError.inferenceFailed("couldn't read the image") }
        return try pixels.withUnsafeBufferPointer { buffer in
            try run(buffer.baseAddress!, width: width, height: height, stride: width * 4, bgr: false)
        }
    }

    /// Depth for a camera frame (kCVPixelFormatType_32BGRA or 32RGBA), with no copy beyond the engine's own.
    public func depth(from pixelBuffer: CVPixelBuffer) throws -> DepthMap {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA else {
            throw EstimatorError.unsupportedPixelFormat(format)
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw EstimatorError.inferenceFailed("no pixel data")
        }
        return try run(base.assumingMemoryBound(to: UInt8.self),
                       width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer),
                       stride: CVPixelBufferGetBytesPerRow(pixelBuffer), bgr: format == kCVPixelFormatType_32BGRA)
    }

    private func run(_ pixels: UnsafePointer<UInt8>, width: Int, height: Int, stride: Int, bgr: Bool) throws -> DepthMap {
        try lock.withLock {
            var h: Int32 = 0, w: Int32 = 0
            guard let values = da_capi_depth_pixels(ctx, pixels, Int32(width), Int32(height), Int32(stride), 4,
                                                    bgr ? 1 : 0, &h, &w)
            else { throw EstimatorError.inferenceFailed(String(cString: da_capi_last_error(ctx))) }
            defer { da_capi_free_floats(values) }
            let count = Int(w) * Int(h)
            return DepthMap(width: Int(w), height: Int(h), values: Array(UnsafeBufferPointer(start: values, count: count)),
                            kind: outputKind)
        }
    }
}

/// A depth map, row-major from the top-left, at the model's processing size.
public struct DepthMap: Sendable {
    public let width: Int
    public let height: Int
    /// Raw values, in the units `kind` describes.
    public let values: [Float]
    public let kind: DepthEstimator.OutputKind

    /// Nearness before stretching: bigger is nearer, in whatever units the model gives.
    public func rawNearness() -> [Float] {
        kind == .inverseDepth ? values : values.map { 1 / max($0, 1e-6) }
    }

    /// The 2nd and 98th percentiles of `rawNearness()`: the span `nearness()` stretches to 0...1.
    /// A live view can smooth this between frames so the contrast doesn't pump.
    public func nearnessRange() -> ClosedRange<Float> {
        Self.percentiles(rawNearness())
    }

    /// 0 (far) ... 1 (near), stretched between the 2nd and 98th percentiles, or across `range`
    /// (in `rawNearness()` units): the input depth shaders (Depth Parallax, Depth Fog) read.
    public func nearness(range: ClosedRange<Float>? = nil) -> [Float] {
        let near = rawNearness()
        guard !near.isEmpty else { return [] }
        let span = range ?? Self.percentiles(near)
        let lo = span.lowerBound, scale = 1 / max(span.upperBound - span.lowerBound, 1e-9)
        return near.map { min(1, max(0, ($0 - lo) * scale)) }
    }

    private static func percentiles(_ near: [Float]) -> ClosedRange<Float> {
        guard !near.isEmpty else { return 0 ... 1 }
        let sorted = near.sorted()
        let lo = sorted[Int(Float(sorted.count - 1) * 0.02)], hi = sorted[Int(Float(sorted.count - 1) * 0.98)]
        return lo ... max(lo, hi)
    }

    /// `nearness()` as an 8-bit grayscale image, brighter nearer.
    public func nearnessImage() -> CGImage? {
        let bytes = nearness().map { UInt8(($0 * 255).rounded()) }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                       space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}
