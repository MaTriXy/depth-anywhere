// swift-tools-version: 6.0
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yossi Elkrief
// DepthAnything for Apple platforms: depth maps from photos and camera frames, on device, with
// the Metal backend. Build the binary first: scripts/build-apple.sh (it copies
// DepthAnything.xcframework next to this file).
import PackageDescription

let package = Package(
    name: "DepthAnything",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "DepthAnything", targets: ["DepthAnything"]),
    ],
    targets: [
        .binaryTarget(name: "DepthAnythingC", path: "DepthAnything.xcframework"),
        .target(
            name: "DepthAnything",
            dependencies: ["DepthAnythingC"],
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
            ]
        ),
        .testTarget(name: "DepthAnythingTests", dependencies: ["DepthAnything"]),
    ]
)
