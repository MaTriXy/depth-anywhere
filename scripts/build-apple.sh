#!/usr/bin/env bash
# Builds DepthAnything.xcframework (iOS device, iOS simulator, macOS; arm64) with the Metal
# backend, for apps: bindings/apple wraps it as the DepthAnything Swift package.
#
#   scripts/build-apple.sh            -> build-apple/DepthAnything.xcframework
#
# Each slice is one static library (libdepthanything + ggml + backends merged with libtool),
# plus the C header and a module map so Swift can import it.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/build-apple"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
IOS_MIN="${IOS_MIN:-17.0}"
MACOS_MIN="${MACOS_MIN:-14.0}"
mkdir -p "$OUT"

build_slice() {   # name sysroot system-name min-flag-var min
    local name=$1 sysroot=$2 system=$3 minvar=$4 min=$5
    local dir="$OUT/$name"
    cmake -S "$ROOT" -B "$dir" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_SYSTEM_NAME="$system" -DCMAKE_OSX_SYSROOT="$sysroot" -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -D"$minvar"="$min" \
        -DDA_BUILD_CLI=OFF -DDA_BUILD_TESTS=OFF -DDA_SHARED=OFF -DDA_GGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BLAS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON >/dev/null
    cmake --build "$dir" -j "$JOBS" --target depthanything >/dev/null
    # One static library per slice: the engine plus every ggml archive it links
    local libs=()
    while IFS= read -r lib; do libs+=("$lib"); done < <(find "$dir" -name '*.a' -not -path '*/merged/*' | sort)
    mkdir -p "$dir/merged/include"
    libtool -static -o "$dir/merged/libDepthAnything.a" "${libs[@]}" 2>/dev/null
    cp "$ROOT/include/da_capi.h" "$dir/merged/include/"
    cat > "$dir/merged/include/module.modulemap" <<'MAP'
module DepthAnythingC {
    header "da_capi.h"
    link "c++"
    link framework "Metal"
    link framework "Foundation"
    link framework "Accelerate"
    export *
}
MAP
    echo "built $name: $(du -h "$dir/merged/libDepthAnything.a" | cut -f1)"
}

build_slice ios-arm64           iphoneos         iOS    CMAKE_OSX_DEPLOYMENT_TARGET "$IOS_MIN"
build_slice ios-arm64-simulator iphonesimulator  iOS    CMAKE_OSX_DEPLOYMENT_TARGET "$IOS_MIN"
build_slice macos-arm64         macosx           Darwin CMAKE_OSX_DEPLOYMENT_TARGET "$MACOS_MIN"

rm -rf "$OUT/DepthAnything.xcframework" "$ROOT/bindings/apple/DepthAnything.xcframework"
xcodebuild -create-xcframework \
    -library "$OUT/ios-arm64/merged/libDepthAnything.a"           -headers "$OUT/ios-arm64/merged/include" \
    -library "$OUT/ios-arm64-simulator/merged/libDepthAnything.a" -headers "$OUT/ios-arm64-simulator/merged/include" \
    -library "$OUT/macos-arm64/merged/libDepthAnything.a"         -headers "$OUT/macos-arm64/merged/include" \
    -output "$OUT/DepthAnything.xcframework" >/dev/null
# SwiftPM binary targets must live inside the package folder
cp -R "$OUT/DepthAnything.xcframework" "$ROOT/bindings/apple/"
echo "wrote $OUT/DepthAnything.xcframework (and a copy in bindings/apple)"
