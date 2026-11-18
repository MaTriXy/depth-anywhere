// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yossi Elkrief
// JNI glue for dev.depthanything.DepthEstimator: the flat C API, with Bitmaps read in place.
#include <android/bitmap.h>
#include <jni.h>
#include "da_capi.h"

extern "C" {

JNIEXPORT jlong JNICALL Java_dev_depthanything_DepthEstimator_nativeLoad(JNIEnv* env, jclass, jstring path, jint threads) {
    const char* p = env->GetStringUTFChars(path, nullptr);
    da_ctx* ctx = da_capi_load(p, threads);
    env->ReleaseStringUTFChars(path, p);
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL Java_dev_depthanything_DepthEstimator_nativeFree(JNIEnv*, jclass, jlong ctx) {
    da_capi_free(reinterpret_cast<da_ctx*>(ctx));
}

JNIEXPORT jint JNICALL Java_dev_depthanything_DepthEstimator_nativeSetResizeTarget(JNIEnv*, jclass, jlong ctx, jint target) {
    return da_capi_set_resize_target(reinterpret_cast<da_ctx*>(ctx), target);
}

JNIEXPORT jint JNICALL Java_dev_depthanything_DepthEstimator_nativeOutputKind(JNIEnv*, jclass, jlong ctx) {
    return da_capi_output_kind(reinterpret_cast<da_ctx*>(ctx));
}

JNIEXPORT jstring JNICALL Java_dev_depthanything_DepthEstimator_nativeLastError(JNIEnv* env, jclass, jlong ctx) {
    return env->NewStringUTF(da_capi_last_error(reinterpret_cast<da_ctx*>(ctx)));
}

// Returns the depth values and writes [width, height] into outSize; null on failure.
JNIEXPORT jfloatArray JNICALL Java_dev_depthanything_DepthEstimator_nativeDepthBitmap(
        JNIEnv* env, jclass, jlong ctx, jobject bitmap, jintArray outSize) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return nullptr;
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return nullptr;
    int h = 0, w = 0;
    // ARGB_8888 bitmaps are laid out R, G, B, A in memory
    float* depth = da_capi_depth_pixels(reinterpret_cast<da_ctx*>(ctx), static_cast<const unsigned char*>(pixels),
                                        (int)info.width, (int)info.height, (int)info.stride, 4, 0, &h, &w);
    AndroidBitmap_unlockPixels(env, bitmap);
    if (!depth) return nullptr;
    jfloatArray out = env->NewFloatArray(w * h);
    if (out) env->SetFloatArrayRegion(out, 0, w * h, depth);
    da_capi_free_floats(depth);
    jint size[2] = { w, h };
    env->SetIntArrayRegion(outSize, 0, 2, size);
    return out;
}

}  // extern "C"
