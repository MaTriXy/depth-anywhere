// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yossi Elkrief
import CoreGraphics
@testable import DepthAnything
import Foundation
import ImageIO
import Testing

/// Needs a model in the repo's models/ folder (depth-anything-small-f32.gguf); skipped otherwise.
private let repo = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
private let model = repo.appending(path: "models/depth-anything-small-f32.gguf")
private let sample = repo.appending(path: "assets/samples/street.jpg")

@Suite(.enabled(if: FileManager.default.fileExists(atPath: model.path)))
struct DepthEstimatorTests {
    func photo() throws -> CGImage {
        let source = try #require(CGImageSourceCreateWithURL(sample as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    @Test func depthFromPhoto() throws {
        let estimator = try DepthEstimator(modelURL: model)
        #expect(estimator.outputKind == .relativeDepth)
        #expect(estimator.defaultProcessingSize == 504)
        let depth = try estimator.depth(from: photo())
        #expect(max(depth.width, depth.height) == 504)
        #expect(depth.values.count == depth.width * depth.height)
        let near = depth.nearness()
        #expect(near.min() == 0 && near.max() == 1)
        #expect(depth.nearnessImage() != nil)
        // An explicit range: the frame's own percentiles give the same map; a wider one lowers contrast
        let range = depth.nearnessRange()
        #expect(depth.nearness(range: range) == near)
        let wider = depth.nearness(range: range.lowerBound ... range.upperBound * 2)
        #expect(wider.max()! < 1)
    }

    @Test func smallerProcessingSize() throws {
        let estimator = try DepthEstimator(modelURL: model)
        estimator.processingSize = 250
        #expect(estimator.processingSize == 252)
        let depth = try estimator.depth(from: photo())
        #expect(max(depth.width, depth.height) == 252)
        estimator.processingSize = nil
        #expect(try max(estimator.depth(from: photo()).width, 1) > 252)
    }
}
