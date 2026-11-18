#include "vit_block.hpp"
#include "ggml_extend.hpp"
#include "compute_mode.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define DA_GELU_NEON 1
#endif
namespace da {

#if defined(DA_GELU_NEON)
// ------------------------------------------------------------------------
// CPU exact-GELU for arm64. ggml's GELU_ERF calls scalar erff() per element
// (~4 ns each; 16M elements per DA3-Small forward @504). This evaluates
//   erf(z) = erf(r) + scale(r) * d * (1 - r*d + (2r^2 - 1)/3 * d^2),
//   r = round(128|z|)/128, d = |z| - r, scale(r) = 2/sqrt(pi) * exp(-r^2)
// from a 513-entry table built once with double-precision erf/exp. With
// |d| <= 1/256 the truncation error is < 2e-10, i.e. the result is erff()
// to within float rounding. |z| >= 4 saturates to +-1 (as erff does).
// DA_GELU=generic restores ggml_gelu_erf for A/B.
// ------------------------------------------------------------------------
struct ErfTable {
    float e[513], s[513];
    ErfTable() {
        for (int i = 0; i <= 512; ++i) {
            const double r = i / 128.0;
            e[i] = (float)std::erf(r);
            s[i] = (float)(1.1283791670955126 * std::exp(-r * r));
        }
    }
};
static const ErfTable& erf_table() { static const ErfTable t; return t; }

static void gelu_erf_neon(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void*) {
    const ErfTable& T = erf_table();
    const int64_t n = ggml_nelements(dst);
    const int64_t i0 = (n * ith / nth) & ~(int64_t)3;
    const int64_t i1 = (ith == nth - 1) ? n : ((n * (ith + 1) / nth) & ~(int64_t)3);
    const float* x = (const float*)a->data;
    float* y = (float*)dst->data;
    const float32x4_t inv_sqrt2 = vdupq_n_f32(0.70710678118654752440f);
    const float32x4_t c128 = vdupq_n_f32(128.0f), cinv128 = vdupq_n_f32(1.0f / 128.0f);
    const float32x4_t one = vdupq_n_f32(1.0f), half = vdupq_n_f32(0.5f);
    const float32x4_t third = vdupq_n_f32(1.0f / 3.0f), two = vdupq_n_f32(2.0f);
    const uint32x4_t imax = vdupq_n_u32(512);
    const uint32x4_t signmask = vdupq_n_u32(0x80000000u);
    int64_t i = i0;
    for (; i + 4 <= i1; i += 4) {
        const float32x4_t xv = vld1q_f32(x + i);
        const float32x4_t z  = vmulq_f32(xv, inv_sqrt2);
        const float32x4_t az = vabsq_f32(z);
        const uint32x4_t idx = vminq_u32(vcvtnq_u32_f32(vmulq_f32(az, c128)), imax);
        const float32x4_t r  = vmulq_f32(vcvtq_f32_u32(idx), cinv128);
        const float32x4_t d  = vsubq_f32(az, r);
        const uint32_t k0 = vgetq_lane_u32(idx, 0), k1 = vgetq_lane_u32(idx, 1),
                       k2 = vgetq_lane_u32(idx, 2), k3 = vgetq_lane_u32(idx, 3);
        const float ev[4] = { T.e[k0], T.e[k1], T.e[k2], T.e[k3] };
        const float sv[4] = { T.s[k0], T.s[k1], T.s[k2], T.s[k3] };
        const float32x4_t e = vld1q_f32(ev), sc = vld1q_f32(sv);
        // p = 1 - d*(r - d*(2r^2-1)/3)
        const float32x4_t q = vmulq_f32(vfmsq_f32(vnegq_f32(one), two, vmulq_f32(r, r)), third); // -(2r^2-1)/3
        const float32x4_t t = vfmaq_f32(r, d, q);                // r - d*(2r^2-1)/3
        const float32x4_t p = vfmsq_f32(one, d, t);              // 1 - d*t
        float32x4_t erfv = vfmaq_f32(e, sc, vmulq_f32(d, p));    // erf(|z|)
        erfv = vminq_f32(erfv, one);
        // restore sign of z (NaN in x propagates through d)
        erfv = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(erfv),
                                               vandq_u32(vreinterpretq_u32_f32(z), signmask)));
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(half, xv), vaddq_f32(one, erfv)));
    }
    for (; i < i1; ++i) y[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * 0.70710678118654752440f));
}
#endif

// Exact (erf) GELU with a CPU fast path on arm64; ggml's op everywhere else.
static ggml_tensor* gelu_erf_fast(ggml_context* ctx, ggml_tensor* x) {
#if defined(DA_GELU_NEON)
    static const bool generic = []{ const char* e = std::getenv("DA_GELU");
                                    return e && !std::strcmp(e, "generic"); }();
    if (!generic && !gpu_mode() && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x))
        return ggml_map_custom1(ctx, x, gelu_erf_neon, GGML_N_TASKS_MAX, nullptr);
#endif
    return gelu_erf(ctx, x);
}

// Reference parity note (DA3 / DINOv2):
//   Block.norm1/norm2 are nn.LayerNorm(dim, eps=ln_eps) with ln_eps=1e-6.
//   The Attention module's q_norm/k_norm are nn.LayerNorm(head_dim) constructed
//   WITHOUT an eps argument (Block does not forward norm_layer to attn_class),
//   so they use torch's DEFAULT nn.LayerNorm eps = 1e-5 -- NOT the block's 1e-6.
//   Therefore the qk_norm LayerNorm inside attention() must use 1e-5.
static constexpr float QK_NORM_EPS = 1e-5f;

BlockWeights load_block(const ModelLoader& ml, int i){
    auto t=[&](const char* s){ return ml.tensor("vit.blk."+std::to_string(i)+"."+s); };
    BlockWeights w;
    w.n1_w=t("norm1.weight"); w.n1_b=t("norm1.bias");
    w.n2_w=t("norm2.weight"); w.n2_b=t("norm2.bias");
    w.ls1=t("ls1"); w.ls2=t("ls2");
    w.swiglu = (ml.config().ffn_type == "swiglu");
    if (w.swiglu){
        w.w12_w=t("mlp_w12.weight"); w.w12_b=t("mlp_w12.bias");
        w.w3_w =t("mlp_w3.weight");  w.w3_b =t("mlp_w3.bias");
    } else {
        w.fc1_w=t("mlp_fc1.weight"); w.fc1_b=t("mlp_fc1.bias");
        w.fc2_w=t("mlp_fc2.weight"); w.fc2_b=t("mlp_fc2.bias");
    }
    w.attn=load_attn(ml, i);
    return w;
}
ggml_tensor* vit_block(ggml_context* ctx, ggml_tensor* x, const BlockWeights& w,
                       int H, int D, float eps, ggml_tensor* cosb, ggml_tensor* sinb){
    // qk_norm uses torch-default eps (1e-5), distinct from the block norm eps (1e-6).
    ggml_tensor* a = attention(ctx, layernorm(ctx, x, w.n1_w, w.n1_b, eps), w.attn, H, D, QK_NORM_EPS, cosb, sinb);
    if (w.ls1) a = layerscale(ctx, a, w.ls1);
    x = ggml_add(ctx, x, a);
    ggml_tensor* xn = layernorm(ctx, x, w.n2_w, w.n2_b, eps);
    ggml_tensor* m;
    if (w.swiglu){
        // SwiGLUFFNFused: x12 = w12(xn) [2*hidden, N]; (x1,x2)=chunk(2,-1);
        // hidden = silu(x1)*x2; m = w3(hidden). Chunk is along the channel dim (ne0):
        // torch x12.chunk(2,-1) -> x1 = first `hidden`, x2 = next `hidden`.
        ggml_tensor* x12 = linear(ctx, w.w12_w, xn, w.w12_b);   // [2*hidden, N]
        const int64_t two_hidden = x12->ne[0];
        const int64_t hidden = two_hidden / 2;
        const int64_t N = x12->ne[1];
        ggml_tensor* x1 = ggml_cont(ctx, ggml_view_2d(ctx, x12, hidden, N, x12->nb[1], 0));
        ggml_tensor* x2 = ggml_cont(ctx, ggml_view_2d(ctx, x12, hidden, N, x12->nb[1],
                                                      (size_t)hidden * x12->nb[0]));
        ggml_tensor* h = ggml_mul(ctx, silu(ctx, x1), x2);     // [hidden, N]
        m = linear(ctx, w.w3_w, h, w.w3_b);                    // [dim, N]
    } else {
        m = linear(ctx, w.fc1_w, xn, w.fc1_b);
        m = gelu_erf_fast(ctx, m);
        m = linear(ctx, w.fc2_w, m, w.fc2_b);
    }
    if (w.ls2) m = layerscale(ctx, m, w.ls2);
    return ggml_add(ctx, x, m);
}
}
