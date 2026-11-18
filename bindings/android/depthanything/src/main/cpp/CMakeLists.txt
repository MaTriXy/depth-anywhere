# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yossi Elkrief
cmake_minimum_required(VERSION 3.22)
project(depthanything_jni CXX)

# The engine, from the repository root, as a static library; ggml's CPU backend for arm64.
# Portable baseline (ARMv8.0) so it runs on every arm64 phone, including older Cortex-A73/A53.
set(DA_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../../..)
set(DA_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(DA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DA_SHARED OFF CACHE BOOL "" FORCE)
set(DA_GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${DA_ROOT} depth-anything)

add_library(depthanything_jni SHARED depth_anything_jni.cpp)
target_include_directories(depthanything_jni PRIVATE ${DA_ROOT}/include)
target_link_libraries(depthanything_jni PRIVATE depthanything jnigraphics log)
