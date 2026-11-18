# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yossi Elkrief
# Precompiled Metal kernels for Apple builds.
#
# With GGML_METAL_EMBED_LIBRARY ggml embeds its Metal *source* and compiles it when the app
# starts. macOS and iOS cache the result, but the first launch after an install or OS update
# pays the whole compile: 7-9 s on an iPhone before the first depth map. This compiles the
# kernels at build time for the target SDK and embeds the metallib next to the source; ggml
# loads it in milliseconds (patches/ggml/0001-*). Devices whose features need other macros
# (the Metal 4 tensor API on M5/A19, or no bfloat) still compile the embedded source.
#
# Included from the top-level CMakeLists after ggml, when DA_METAL_PRECOMPILED is ON.

if(NOT TARGET ggml-metal)
    return()
endif()
if(NOT GGML_METAL_EMBED_LIBRARY)
    message(STATUS "depth-anywhere: DA_METAL_PRECOMPILED needs GGML_METAL_EMBED_LIBRARY; skipped")
    return()
endif()

# Needs patches/ggml/0001 (applied by GgmlPatches.cmake), which loads the embedded metallib
if(NOT DA_GGML_PATCHES)
    message(STATUS "depth-anywhere: DA_METAL_PRECOMPILED needs DA_GGML_PATCHES; skipped")
    return()
endif()
set(_da_ggml "${DA_ROOT}/third_party/ggml")

# SDK and minimum OS for the metal compiler, from the CMake target
set(_da_sysroot "${CMAKE_OSX_SYSROOT}")
if(_da_sysroot MATCHES "iphonesimulator|iPhoneSimulator")
    set(_da_sdk iphonesimulator)
    set(_da_min "-mios-simulator-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
elseif(_da_sysroot MATCHES "iphoneos|iPhoneOS")
    set(_da_sdk iphoneos)
    set(_da_min "-mios-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
else()
    set(_da_sdk macosx)
    if(CMAKE_OSX_DEPLOYMENT_TARGET)
        set(_da_min "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    else()
        set(_da_min "")
    endif()
endif()

set(_da_src  "${_da_ggml}/src/ggml-metal/ggml-metal.metal")
set(_da_out  "${CMAKE_CURRENT_BINARY_DIR}/metal-precompiled")
set(_da_air  "${_da_out}/ggml-metal.air")
set(_da_lib  "${_da_out}/ggml-metal.metallib")
set(_da_asm  "${_da_out}/ggml-metal-bin.s")
file(MAKE_DIRECTORY "${_da_out}")

# Same macros ggml passes at runtime on a bfloat-capable, pre-tensor-API GPU; -O3 with the
# compiler's default fast math, as the runtime compile uses
add_custom_command(
    OUTPUT "${_da_asm}"
    COMMAND xcrun -sdk ${_da_sdk} metal -O3 ${_da_min} -DGGML_METAL_HAS_BF16=1 -w
            -I "${_da_ggml}/src" -I "${_da_ggml}/src/ggml-metal" -c "${_da_src}" -o "${_da_air}"
    COMMAND xcrun -sdk ${_da_sdk} metallib "${_da_air}" -o "${_da_lib}"
    COMMAND echo ".section __DATA,__ggml_metalbin" > "${_da_asm}"
    COMMAND echo ".globl _ggml_metallib_bin_start" >> "${_da_asm}"
    COMMAND echo ".p2align 4" >> "${_da_asm}"
    COMMAND echo "_ggml_metallib_bin_start:" >> "${_da_asm}"
    COMMAND echo ".incbin \"${_da_lib}\"" >> "${_da_asm}"
    COMMAND echo ".globl _ggml_metallib_bin_end" >> "${_da_asm}"
    COMMAND echo "_ggml_metallib_bin_end:" >> "${_da_asm}"
    DEPENDS "${_da_src}" "${_da_ggml}/src/ggml-common.h" "${_da_ggml}/src/ggml-metal/ggml-metal-impl.h"
    COMMENT "Precompiling Metal kernels (${_da_sdk})"
    VERBATIM)
# The rule lives in this directory, so build it from a target here and make ggml-metal wait
add_custom_target(da-metal-precompiled DEPENDS "${_da_asm}")
add_dependencies(ggml-metal da-metal-precompiled)
set_source_files_properties("${_da_asm}" TARGET_DIRECTORY ggml-metal PROPERTIES GENERATED TRUE)
target_sources(ggml-metal PRIVATE "${_da_asm}")
target_compile_definitions(ggml-metal PRIVATE GGML_METAL_EMBED_METALLIB=1)
message(STATUS "depth-anywhere: Metal kernels precompiled for ${_da_sdk} ${_da_min}")
