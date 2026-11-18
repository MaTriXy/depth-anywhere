// da_capi_depth_pixels / da_capi_set_resize_target: in-memory input for apps (camera frames,
// decoded photos). Checks that RGB, RGBA and BGRA buffers (with row padding) all give the same
// depth as the engine's native path on the same image, and that the processing resolution
// can be lowered and restored.
#include "da_capi.h"
#include "engine.hpp"
#include "image_io.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool exists(const char* p){ if (FILE* f = std::fopen(p, "rb")){ std::fclose(f); return true; } return false; }

int main(){
    // Any DA3 model works; the small one is the one apps ship
    const char* gguf = std::getenv("DA_TEST_GGUF");
    if (!gguf || !exists(gguf)) gguf = "models/depth-anything-small-f32.gguf";
    const char* image = std::getenv("DA_TEST_IMAGE");
    if (!image || !exists(image)) image = "assets/samples/street.jpg";
    if (!exists(gguf) || !exists(image)) return 77;

    da::Image img;
    if (!da::load_image_rgb(image, img)) return 1;
    auto eng = da::Engine::load(gguf, 4);
    if (!eng) return 1;
    std::vector<float> ref, conf; int RH = 0, RW = 0;
    if (!eng->depth_native_image(img, ref, conf, RH, RW)) return 1;

    da_ctx* c = da_capi_load(gguf, 4);
    if (!c) return 1;
    bool ok = true;
    auto check = [&](const char* label, const std::vector<unsigned char>& px, int stride, int ch, int bgr){
        int h = 0, w = 0;
        float* d = da_capi_depth_pixels(c, px.data(), img.w, img.h, stride, ch, bgr, &h, &w);
        if (!d){ std::fprintf(stderr, "%s: %s\n", label, da_capi_last_error(c)); ok = false; return; }
        double maxd = 0;
        if (h != RH || w != RW){ std::fprintf(stderr, "%s: %dx%d vs %dx%d\n", label, w, h, RW, RH); ok = false; }
        else for (size_t i = 0; i < ref.size(); ++i) maxd = std::fmax(maxd, std::fabs(d[i] - ref[i]));
        std::fprintf(stderr, "%s: %dx%d max|d|=%g\n", label, w, h, maxd);
        if (maxd > 1e-5) ok = false;
        da_capi_free_floats(d);
    };
    const int W = img.w, H = img.h, pad = 12;
    std::vector<unsigned char> rgb(img.rgb), rgba((size_t)(W * 4 + pad) * H), bgra(rgba.size());
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x){
        const unsigned char* s = &img.rgb[((size_t)y * W + x) * 3];
        unsigned char* a = &rgba[(size_t)y * (W * 4 + pad) + x * 4];
        unsigned char* b = &bgra[(size_t)y * (W * 4 + pad) + x * 4];
        a[0] = s[0]; a[1] = s[1]; a[2] = s[2]; a[3] = 255;
        b[0] = s[2]; b[1] = s[1]; b[2] = s[0]; b[3] = 255;
    }
    check("rgb", rgb, 0, 3, 0);
    check("rgba+padding", rgba, W * 4 + pad, 4, 0);
    check("bgra+padding", bgra, W * 4 + pad, 4, 1);

    // A lower processing resolution: smaller output, rounded to whole patches
    const int def = da_capi_set_resize_target(c, 0);
    const int low = da_capi_set_resize_target(c, 250);
    int h = 0, w = 0;
    float* d = da_capi_depth_pixels(c, rgb.data(), W, H, 0, 3, 0, &h, &w);
    std::fprintf(stderr, "resize target default=%d low=%d -> %dx%d\n", def, low, w, h);
    if (!d || low % 14 != 0 || low >= def || std::max(h, w) > low) ok = false;
    da_capi_free_floats(d);
    if (da_capi_set_resize_target(c, 0) != def) ok = false;
    if (da_capi_depth_pixels(c, rgb.data(), W, H, 1, 3, 0, &h, &w) != nullptr) ok = false;  // stride too small
    const int kind = da_capi_output_kind(c);
    std::fprintf(stderr, "output kind %d\n", kind);
    if (kind < 0 || kind > 2) ok = false;
    da_capi_free(c);
    return ok ? 0 : 1;
}
