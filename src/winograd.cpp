#include "winograd.hpp"
#include <algorithm>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif
// ARM NEON path (Apple silicon, Android arm64): hand-written winograd-domain GEMM
// plus input/output transforms vectorised 8-wide with GCC/Clang vector extensions
// (lowered to pairs of NEON registers).
#if !defined(__AVX512F__) && defined(__ARM_NEON) && defined(__aarch64__) && \
    (defined(__clang__) || defined(__GNUC__))
#include <arm_neon.h>
#define DA_WINO_NEON 1
#endif

namespace da {
namespace {

// ========================================================================
// Mode selection (DA_WINO env): which Winograd algorithm + inner kernel.
//   "f2"  : F(2x2,3x3), per-tile AVX-512 GEMV (the original CPU-opt #4 path)
//   "f2b" : F(2x2,3x3), blocked GEMM over a block of tiles (parity-identical,
//           pure-speed: reuses each U row across TB tiles)  <-- auto default
//   "f4"  : F(4x4,3x3), blocked GEMM (4x fewer mults vs direct; less accurate)
// Parity: f2/f2b are exact (halves+ints, max|d|~1e-5). f4 uses 1/6,1/24
// fractions so it is less accurate (max|d|~1e-4..1e-3) — gated by the suite.
// ========================================================================
enum class Mode { F2, F2B, F4 };

static Mode parse_mode() {
    const char* m = std::getenv("DA_WINO");
    if (m) {
        if (!std::strcmp(m, "f2"))  return Mode::F2;
        if (!std::strcmp(m, "f2b")) return Mode::F2B;
        if (!std::strcmp(m, "f4"))  return Mode::F4;
    }
    return Mode::F2B;   // auto default (fastest parity-exact variant)
}

// DA_WINO_SCALAR=1 forces the portable scalar kernels (A/B against the NEON path).
static bool scalar_forced() {
    static const bool f = std::getenv("DA_WINO_SCALAR") != nullptr;
    return f;
}

// Tile-block width for the blocked GEMM microkernel (number of tiles batched
// per winograd-domain multiply). 8 keeps all accumulators in zmm registers
// while amortizing each U-row load across 8 tiles.
constexpr int TB = 8;

// ------------------------------------------------------------------------
// F(2x2,3x3) transforms (exact: halves + integers).
//   B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
//   G   = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
//   A^T = [[1,1,1,0],[0,1,-1,-1]]
// ------------------------------------------------------------------------
struct F2Policy {
    static constexpr int IT = 4, OT = 2, P = 16;   // input tile / output tile / positions

    // U = G g G^T, 3x3 -> 4x4 (u[16]).
    static void filt(const float g[9], float u[16]) {
        float Gg[4][3];
        for (int j = 0; j < 3; ++j) {
            float c0 = g[0*3 + j], c1 = g[1*3 + j], c2 = g[2*3 + j];
            Gg[0][j] = c0;
            Gg[1][j] = 0.5f * (c0 + c1 + c2);
            Gg[2][j] = 0.5f * (c0 - c1 + c2);
            Gg[3][j] = c2;
        }
        for (int i = 0; i < 4; ++i) {
            float c0 = Gg[i][0], c1 = Gg[i][1], c2 = Gg[i][2];
            u[i*4 + 0] = c0;
            u[i*4 + 1] = 0.5f * (c0 + c1 + c2);
            u[i*4 + 2] = 0.5f * (c0 - c1 + c2);
            u[i*4 + 3] = c2;
        }
    }
    // V = B^T d B, 4x4 -> 4x4 (v[16]).
    template<class T> static inline void inp(const T d[16], T v[16]) {
        T m[16];
        for (int j = 0; j < 4; ++j) {
            T r0 = d[0*4 + j], r1 = d[1*4 + j], r2 = d[2*4 + j], r3 = d[3*4 + j];
            m[0*4 + j] = r0 - r2;
            m[1*4 + j] = r1 + r2;
            m[2*4 + j] = r2 - r1;
            m[3*4 + j] = r1 - r3;
        }
        for (int i = 0; i < 4; ++i) {
            T c0 = m[i*4 + 0], c1 = m[i*4 + 1], c2 = m[i*4 + 2], c3 = m[i*4 + 3];
            v[i*4 + 0] = c0 - c2;
            v[i*4 + 1] = c1 + c2;
            v[i*4 + 2] = c2 - c1;
            v[i*4 + 3] = c1 - c3;
        }
    }
    // Y = A^T m A, 4x4 -> 2x2 (y[4]).
    template<class T> static inline void outp(const T m[16], T y[4]) {
        T p[8];
        for (int j = 0; j < 4; ++j) {
            T r0 = m[0*4 + j], r1 = m[1*4 + j], r2 = m[2*4 + j], r3 = m[3*4 + j];
            p[0*4 + j] = r0 + r1 + r2;
            p[1*4 + j] = r1 - r2 - r3;
        }
        for (int i = 0; i < 2; ++i) {
            T c0 = p[i*4 + 0], c1 = p[i*4 + 1], c2 = p[i*4 + 2], c3 = p[i*4 + 3];
            y[i*2 + 0] = c0 + c1 + c2;
            y[i*2 + 1] = c1 - c2 - c3;
        }
    }
};

// ------------------------------------------------------------------------
// F(4x4,3x3) transforms (Lavin & Gray). Verified numerically against a direct
// 3x3 conv (float64 max|d|~7e-14). Uses 1/6 and 1/24, so float32 accuracy is
// lower than F(2x2). 6x6 input tile -> 4x4 output, 36 winograd positions.
//   B^T row ops on a length-6 vector x:
//     r0=4x0-5x2+x4; r1=-4x1-4x2+x3+x4; r2=4x1-4x2-x3+x4;
//     r3=-2x1-x2+2x3+x4; r4=2x1-x2-2x3+x4; r5=4x1-5x3+x5
//   G row ops on a length-3 vector y:
//     u0=y0/4; u1=-(y0+y1+y2)/6; u2=(-y0+y1-y2)/6;
//     u3=y0/24+y1/12+y2/6; u4=y0/24-y1/12+y2/6; u5=y2
//   A^T row ops on a length-6 vector m:
//     o0=m0+m1+m2+m3+m4; o1=m1-m2+2m3-2m4; o2=m1+m2+4m3+4m4; o3=m1-m2+8m3-8m4+m5
// ------------------------------------------------------------------------
struct F4Policy {
    static constexpr int IT = 6, OT = 4, P = 36;

    template<class T> static inline void Brow(const T x[6], T r[6]) {
        r[0] = 4.0f*x[0] - 5.0f*x[2] + x[4];
        r[1] = -4.0f*x[1] - 4.0f*x[2] + x[3] + x[4];
        r[2] = 4.0f*x[1] - 4.0f*x[2] - x[3] + x[4];
        r[3] = -2.0f*x[1] - x[2] + 2.0f*x[3] + x[4];
        r[4] = 2.0f*x[1] - x[2] - 2.0f*x[3] + x[4];
        r[5] = 4.0f*x[1] - 5.0f*x[3] + x[5];
    }
    static inline void Grow(const float y[3], float u[6]) {
        const float a = y[0], b = y[1], c = y[2];
        u[0] = 0.25f * a;
        u[1] = -(a + b + c) * (1.0f/6.0f);
        u[2] = (-a + b - c) * (1.0f/6.0f);
        u[3] = a*(1.0f/24.0f) + b*(1.0f/12.0f) + c*(1.0f/6.0f);
        u[4] = a*(1.0f/24.0f) - b*(1.0f/12.0f) + c*(1.0f/6.0f);
        u[5] = c;
    }
    template<class T> static inline void Arow(const T m[6], T o[4]) {
        o[0] = m[0] + m[1] + m[2] + m[3] + m[4];
        o[1] = m[1] - m[2] + 2.0f*m[3] - 2.0f*m[4];
        o[2] = m[1] + m[2] + 4.0f*m[3] + 4.0f*m[4];
        o[3] = m[1] - m[2] + 8.0f*m[3] - 8.0f*m[4] + m[5];
    }

    // U = G g G^T, 3x3 -> 6x6 (u[36]).
    static void filt(const float g[9], float u[36]) {
        float Gg[6][3];   // apply G on the 3 columns of g
        for (int j = 0; j < 3; ++j) {
            float col[3] = { g[0*3 + j], g[1*3 + j], g[2*3 + j] };
            float out[6]; Grow(col, out);
            for (int i = 0; i < 6; ++i) Gg[i][j] = out[i];
        }
        for (int i = 0; i < 6; ++i) {       // apply G on the 3 columns of each Gg row
            float out[6]; Grow(Gg[i], out);
            for (int k = 0; k < 6; ++k) u[i*6 + k] = out[k];
        }
    }
    // V = B^T d B, 6x6 -> 6x6 (v[36]).
    template<class T> static inline void inp(const T d[36], T v[36]) {
        T m[36];
        for (int j = 0; j < 6; ++j) {        // B^T on columns
            T col[6] = { d[0*6+j], d[1*6+j], d[2*6+j], d[3*6+j], d[4*6+j], d[5*6+j] };
            T out[6]; Brow(col, out);
            for (int i = 0; i < 6; ++i) m[i*6 + j] = out[i];
        }
        for (int i = 0; i < 6; ++i) {        // B^T on rows
            T out[6]; Brow(m + i*6, out);
            for (int k = 0; k < 6; ++k) v[i*6 + k] = out[k];
        }
    }
    // Y = A^T m A, 6x6 -> 4x4 (y[16]).
    template<class T> static inline void outp(const T m[36], T y[16]) {
        T p[24];   // 4x6 (A^T on columns)
        for (int j = 0; j < 6; ++j) {
            T col[6] = { m[0*6+j], m[1*6+j], m[2*6+j], m[3*6+j], m[4*6+j], m[5*6+j] };
            T out[4]; Arow(col, out);
            for (int i = 0; i < 4; ++i) p[i*6 + j] = out[i];
        }
        for (int i = 0; i < 4; ++i) {        // A^T on rows
            T out[4]; Arow(p + i*6, out);
            for (int k = 0; k < 4; ++k) y[i*4 + k] = out[k];
        }
    }
};

// ------------------------------------------------------------------------
// Persistent per-op state: caches the filter transform U (computed once from
// w->data; reused across forwards). Scratch (V,M) is per-thread on the stack
// / in small per-thread vectors.
// ------------------------------------------------------------------------
struct WinogradState {
    Mode mode = Mode::F2B;
    int W = 0, H = 0, IC = 0, OC = 0, N = 0, pad = 0;
    int Wout = 0, Hout = 0, tilesX = 0, tilesY = 0;
    const void* wdata = nullptr;
    // U layout: U[pos*IC*OC + ic*OC + oc], pos in 0..P-1. OC innermost so the
    // winograd-domain multiply vectorizes over OC.
    std::vector<float> U;
    std::once_flag once;
};

template<class Pol>
static void build_U(WinogradState* st, const float* w) {
    const int IC = st->IC, OC = st->OC;
    st->U.assign((size_t)Pol::P * IC * OC, 0.0f);
    float u[Pol::P];
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            const float* g = w + ((size_t)oc * IC + ic) * 9;
            Pol::filt(g, u);
            for (int pos = 0; pos < Pol::P; ++pos)
                st->U[(size_t)pos * IC * OC + (size_t)ic * OC + oc] = u[pos];
        }
    }
}

// ------------------------------------------------------------------------
// Per-tile GEMV (original F2 path): M[oc] = sum_ic U[ic,oc]*V[ic].
// ------------------------------------------------------------------------
static inline void wino_gemv(const float* Upos, const float* Vpos, float* out, int IC, int OC) {
#if defined(__AVX512F__)
    int oc = 0;
    for (; oc + 16 <= OC; oc += 16) {
        __m512 acc = _mm512_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(up + (size_t)ic * OC),
                                  _mm512_set1_ps(Vpos[ic]), acc);
        _mm512_storeu_ps(out + oc, acc);
    }
    if (oc < OC) {
        const int rem = OC - oc;
        const __mmask16 mask = (__mmask16)((1u << rem) - 1u);
        __m512 acc = _mm512_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic)
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, up + (size_t)ic * OC),
                                  _mm512_set1_ps(Vpos[ic]), acc);
        _mm512_mask_storeu_ps(out + oc, mask, acc);
    }
#else
    for (int oc = 0; oc < OC; ++oc) out[oc] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const float vv = Vpos[ic];
        const float* up = Upos + (size_t)ic * OC;
        for (int oc = 0; oc < OC; ++oc) out[oc] += up[oc] * vv;
    }
#endif
}

// ------------------------------------------------------------------------
// Blocked GEMM microkernel for one winograd position:
//   M[t][oc] = sum_ic U[ic][oc] * V[ic][t],  t in [0,TBcur), oc in [0,OC).
// U: [IC][OC] row-major (OC innermost). V: [IC][TB] row-major. M: [TB][OC].
// Each loaded U-row (16 OC lanes) is reused across all TBcur tiles -> far
// better arithmetic intensity than the per-tile GEMV.
// ------------------------------------------------------------------------
static inline void wino_gemm_block(const float* U, const float* V, float* M,
                                   int IC, int OC, int TBcur) {
#if defined(__AVX512F__)
    int oc = 0;
    for (; oc + 16 <= OC; oc += 16) {
        __m512 acc[TB];
        for (int t = 0; t < TB; ++t) acc[t] = _mm512_setzero_ps();
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const __m512 u = _mm512_loadu_ps(up + (size_t)ic * OC);
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t)
                acc[t] = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[t]), acc[t]);
        }
        for (int t = 0; t < TBcur; ++t)
            _mm512_storeu_ps(M + (size_t)t * OC + oc, acc[t]);
    }
    if (oc < OC) {
        const int rem = OC - oc;
        const __mmask16 mask = (__mmask16)((1u << rem) - 1u);
        __m512 acc[TB];
        for (int t = 0; t < TB; ++t) acc[t] = _mm512_setzero_ps();
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const __m512 u = _mm512_maskz_loadu_ps(mask, up + (size_t)ic * OC);
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t)
                acc[t] = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[t]), acc[t]);
        }
        for (int t = 0; t < TBcur; ++t)
            _mm512_mask_storeu_ps(M + (size_t)t * OC + oc, mask, acc[t]);
    }
#else
    for (int t = 0; t < TBcur; ++t)
        for (int oc = 0; oc < OC; ++oc) M[(size_t)t * OC + oc] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const float* up = U + (size_t)ic * OC;
        const float* vp = V + (size_t)ic * TB;
        for (int t = 0; t < TBcur; ++t) {
            const float vv = vp[t];
            float* mt = M + (size_t)t * OC;
            for (int oc = 0; oc < OC; ++oc) mt[oc] += up[oc] * vv;
        }
    }
#endif
}

// ------------------------------------------------------------------------
// Original F(2x2) per-tile path (mode "f2").
// ------------------------------------------------------------------------
static void compute_f2_gemv(WinogradState* st, ggml_tensor* dst, int ith, int nth) {
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;
    const float* U = st->U.data();
    const float* bias = dst->src[2] ? (const float*)dst->src[2]->data : nullptr;

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)st->N * ntiles;
    const int64_t beg = total * ith / nth;
    const int64_t end = total * (ith + 1) / nth;

    std::vector<float> Vbuf((size_t)16 * IC);
    std::vector<float> Mbuf((size_t)16 * OC);
    float dpatch[16], vpatch[16], mpatch[16], ypatch[4];

    for (int64_t idx = beg; idx < end; ++idx) {
        const int n  = (int)(idx / ntiles);
        const int t  = (int)(idx % ntiles);
        const int ty = t / tilesX, tx = t % tilesX;
        const int oy0 = ty * 2, ox0 = tx * 2;
        const int iy0 = oy0 - pad, ix0 = ox0 - pad;
        const float* xn = x + (size_t)n * IC * H * W;

        for (int ic = 0; ic < IC; ++ic) {
            const float* xc = xn + (size_t)ic * H * W;
            for (int i = 0; i < 4; ++i) {
                const int yy = iy0 + i;
                const bool yok = (yy >= 0 && yy < H);
                const float* row = yok ? (xc + (size_t)yy * W) : nullptr;
                for (int j = 0; j < 4; ++j) {
                    const int xx = ix0 + j;
                    dpatch[i*4 + j] = (yok && xx >= 0 && xx < W) ? row[xx] : 0.0f;
                }
            }
            F2Policy::inp(dpatch, vpatch);
            for (int pos = 0; pos < 16; ++pos) Vbuf[(size_t)pos * IC + ic] = vpatch[pos];
        }
        for (int pos = 0; pos < 16; ++pos)
            wino_gemv(U + (size_t)pos * IC * OC, Vbuf.data() + (size_t)pos * IC,
                      Mbuf.data() + (size_t)pos * OC, IC, OC);

        float* yn = y + (size_t)n * OC * Hout * Wout;
        for (int oc = 0; oc < OC; ++oc) {
            for (int pos = 0; pos < 16; ++pos) mpatch[pos] = Mbuf[(size_t)pos * OC + oc];
            F2Policy::outp(mpatch, ypatch);
            if (bias) for (int k = 0; k < 4; ++k) ypatch[k] += bias[oc];
            float* yc = yn + (size_t)oc * Hout * Wout;
            for (int i = 0; i < 2; ++i) {
                const int oy = oy0 + i; if (oy >= Hout) continue;
                for (int j = 0; j < 2; ++j) {
                    const int ox = ox0 + j; if (ox >= Wout) continue;
                    yc[(size_t)oy * Wout + ox] = ypatch[i*2 + j];
                }
            }
        }
    }
}

// ------------------------------------------------------------------------
// Blocked path (modes "f2b" / "f4"): batch TB tiles per winograd-domain GEMM.
// ------------------------------------------------------------------------
template<class Pol>
static void compute_blocked(WinogradState* st, ggml_tensor* dst, int ith, int nth) {
    constexpr int IT = Pol::IT, OT = Pol::OT, P = Pol::P;
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;
    const float* U = st->U.data();
    const float* bias = dst->src[2] ? (const float*)dst->src[2]->data : nullptr;

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)st->N * ntiles;
    // Split the work by tile-BLOCKS so each thread's GEMMs stay full-width.
    const int64_t nblocks = (total + TB - 1) / TB;
    const int64_t bbeg = nblocks * ith / nth;
    const int64_t bend = nblocks * (ith + 1) / nth;

    // Per-thread scratch.
    std::vector<float> Vblk((size_t)P * IC * TB);   // V[pos][ic][t]
    std::vector<float> Mblk((size_t)P * TB * OC);   // M[pos][t][oc]
    float dpatch[IT*IT], vpatch[P], mpatch[P], ypatch[OT*OT];

    for (int64_t b = bbeg; b < bend; ++b) {
        const int64_t t0 = b * TB;
        const int TBcur = (int)std::min<int64_t>(TB, total - t0);

        // 1. Input transform for each tile in the block -> Vblk[pos][ic][t].
        for (int tb = 0; tb < TBcur; ++tb) {
            const int64_t idx = t0 + tb;
            const int n  = (int)(idx / ntiles);
            const int t  = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            const int iy0 = ty * OT - pad, ix0 = tx * OT - pad;
            const float* xn = x + (size_t)n * IC * H * W;
            for (int ic = 0; ic < IC; ++ic) {
                const float* xc = xn + (size_t)ic * H * W;
                for (int i = 0; i < IT; ++i) {
                    const int yy = iy0 + i;
                    const bool yok = (yy >= 0 && yy < H);
                    const float* row = yok ? (xc + (size_t)yy * W) : nullptr;
                    for (int j = 0; j < IT; ++j) {
                        const int xx = ix0 + j;
                        dpatch[i*IT + j] = (yok && xx >= 0 && xx < W) ? row[xx] : 0.0f;
                    }
                }
                Pol::inp(dpatch, vpatch);
                float* vbase = Vblk.data() + (size_t)ic * TB + tb;
                for (int pos = 0; pos < P; ++pos)
                    vbase[(size_t)pos * IC * TB] = vpatch[pos];
            }
        }

        // 2. Winograd-domain blocked GEMM per position.
        for (int pos = 0; pos < P; ++pos)
            wino_gemm_block(U + (size_t)pos * IC * OC,
                            Vblk.data() + (size_t)pos * IC * TB,
                            Mblk.data() + (size_t)pos * TB * OC, IC, OC, TBcur);

        // 3. Output transform per tile per oc -> scatter OTxOT into dst.
        for (int tb = 0; tb < TBcur; ++tb) {
            const int64_t idx = t0 + tb;
            const int n  = (int)(idx / ntiles);
            const int t  = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            const int oy0 = ty * OT, ox0 = tx * OT;
            float* yn = y + (size_t)n * OC * Hout * Wout;
            for (int oc = 0; oc < OC; ++oc) {
                const float* mbase = Mblk.data() + (size_t)tb * OC + oc;
                for (int pos = 0; pos < P; ++pos)
                    mpatch[pos] = mbase[(size_t)pos * TB * OC];
                Pol::outp(mpatch, ypatch);
                if (bias) for (int k = 0; k < OT*OT; ++k) ypatch[k] += bias[oc];
                float* yc = yn + (size_t)oc * Hout * Wout;
                for (int i = 0; i < OT; ++i) {
                    const int oy = oy0 + i; if (oy >= Hout) continue;
                    for (int j = 0; j < OT; ++j) {
                        const int ox = ox0 + j; if (ox >= Wout) continue;
                        yc[(size_t)oy * Wout + ox] = ypatch[i*OT + j];
                    }
                }
            }
        }
    }
}

#if defined(DA_WINO_NEON)
// ------------------------------------------------------------------------
// ARM NEON blocked path (modes "f2b" / "f4").
//
// Same algorithm, tile blocking and summation order as compute_blocked (so the
// F(2x2) result is bit-identical to the scalar fallback), but:
//   * the winograd-domain GEMM keeps an 8-tile x 8-OC accumulator block in 16
//     NEON registers (4 loads + 16 FMAs per input channel);
//   * the input transform runs across the 8 tiles of a block at once and stores
//     V[pos][ic][0..7] contiguously;
//   * the output transform runs across 8 output channels at once, and the conv
//     bias is added here instead of in a separate full-resolution ADD node.
// ------------------------------------------------------------------------
typedef float vf8 __attribute__((vector_size(32)));
static_assert(TB == 8, "NEON winograd path assumes 8-tile blocks");
static inline vf8  ld8(const float* p)        { vf8 v; std::memcpy(&v, p, sizeof(v)); return v; }
static inline void st8(float* p, const vf8& v){ std::memcpy(p, &v, sizeof(v)); }

// M[t][oc] = sum_ic U[ic][oc] * V[ic][t] for t in [0,8), oc in [0,OC).
// Row strides (in floats): us for U rows, vs for V rows, ms for M rows. Callers
// zero the unused V lanes of a short block.
static inline void gemm8_neon(const float* U, size_t us, const float* V, size_t vs,
                              float* M, size_t ms, int IC, int OC) {
    int oc = 0;
    for (; oc + 8 <= OC; oc += 8) {
        float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a10 = a00, a11 = a00,
                    a20 = a00, a21 = a00, a30 = a00, a31 = a00,
                    a40 = a00, a41 = a00, a50 = a00, a51 = a00,
                    a60 = a00, a61 = a00, a70 = a00, a71 = a00;
        const float* up = U + oc;
        const float* vp = V;
        for (int ic = 0; ic < IC; ++ic, up += us, vp += vs) {
            const float32x4_t u0 = vld1q_f32(up), u1 = vld1q_f32(up + 4);
            const float32x4_t v0 = vld1q_f32(vp), v1 = vld1q_f32(vp + 4);
            a00 = vfmaq_laneq_f32(a00, u0, v0, 0); a01 = vfmaq_laneq_f32(a01, u1, v0, 0);
            a10 = vfmaq_laneq_f32(a10, u0, v0, 1); a11 = vfmaq_laneq_f32(a11, u1, v0, 1);
            a20 = vfmaq_laneq_f32(a20, u0, v0, 2); a21 = vfmaq_laneq_f32(a21, u1, v0, 2);
            a30 = vfmaq_laneq_f32(a30, u0, v0, 3); a31 = vfmaq_laneq_f32(a31, u1, v0, 3);
            a40 = vfmaq_laneq_f32(a40, u0, v1, 0); a41 = vfmaq_laneq_f32(a41, u1, v1, 0);
            a50 = vfmaq_laneq_f32(a50, u0, v1, 1); a51 = vfmaq_laneq_f32(a51, u1, v1, 1);
            a60 = vfmaq_laneq_f32(a60, u0, v1, 2); a61 = vfmaq_laneq_f32(a61, u1, v1, 2);
            a70 = vfmaq_laneq_f32(a70, u0, v1, 3); a71 = vfmaq_laneq_f32(a71, u1, v1, 3);
        }
        float* m = M + oc;
        vst1q_f32(m + 0*ms, a00); vst1q_f32(m + 0*ms + 4, a01);
        vst1q_f32(m + 1*ms, a10); vst1q_f32(m + 1*ms + 4, a11);
        vst1q_f32(m + 2*ms, a20); vst1q_f32(m + 2*ms + 4, a21);
        vst1q_f32(m + 3*ms, a30); vst1q_f32(m + 3*ms + 4, a31);
        vst1q_f32(m + 4*ms, a40); vst1q_f32(m + 4*ms + 4, a41);
        vst1q_f32(m + 5*ms, a50); vst1q_f32(m + 5*ms + 4, a51);
        vst1q_f32(m + 6*ms, a60); vst1q_f32(m + 6*ms + 4, a61);
        vst1q_f32(m + 7*ms, a70); vst1q_f32(m + 7*ms + 4, a71);
    }
    for (; oc + 4 <= OC; oc += 4) {
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0,
                    a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        const float* up = U + oc;
        const float* vp = V;
        for (int ic = 0; ic < IC; ++ic, up += us, vp += vs) {
            const float32x4_t u0 = vld1q_f32(up);
            const float32x4_t v0 = vld1q_f32(vp), v1 = vld1q_f32(vp + 4);
            a0 = vfmaq_laneq_f32(a0, u0, v0, 0); a1 = vfmaq_laneq_f32(a1, u0, v0, 1);
            a2 = vfmaq_laneq_f32(a2, u0, v0, 2); a3 = vfmaq_laneq_f32(a3, u0, v0, 3);
            a4 = vfmaq_laneq_f32(a4, u0, v1, 0); a5 = vfmaq_laneq_f32(a5, u0, v1, 1);
            a6 = vfmaq_laneq_f32(a6, u0, v1, 2); a7 = vfmaq_laneq_f32(a7, u0, v1, 3);
        }
        float* m = M + oc;
        vst1q_f32(m + 0*ms, a0); vst1q_f32(m + 1*ms, a1);
        vst1q_f32(m + 2*ms, a2); vst1q_f32(m + 3*ms, a3);
        vst1q_f32(m + 4*ms, a4); vst1q_f32(m + 5*ms, a5);
        vst1q_f32(m + 6*ms, a6); vst1q_f32(m + 7*ms, a7);
    }
    for (; oc < OC; ++oc) {
        float acc[TB] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int ic = 0; ic < IC; ++ic) {
            const float u = U[(size_t)ic * us + oc];
            const float* vp = V + (size_t)ic * vs;
            for (int t = 0; t < TB; ++t) acc[t] += u * vp[t];
        }
        for (int t = 0; t < TB; ++t) M[(size_t)t * ms + oc] = acc[t];
    }
}
static inline void wino_gemm_block_neon(const float* U, const float* V, float* M,
                                        int IC, int OC) {
    gemm8_neon(U, (size_t)OC, V, (size_t)TB, M, (size_t)OC, IC, OC);
}

template<class Pol>
static void compute_blocked_neon(WinogradState* st, ggml_tensor* dst, int ith, int nth) {
    constexpr int IT = Pol::IT, OT = Pol::OT, P = Pol::P;
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;
    const float* bias = dst->src[2] ? (const float*)dst->src[2]->data : nullptr;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;
    const float* U = st->U.data();
    const size_t HW = (size_t)H * W, HWout = (size_t)Hout * Wout;

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)st->N * ntiles;
    const int64_t nblocks = (total + TB - 1) / TB;
    const int64_t bbeg = nblocks * ith / nth;
    const int64_t bend = nblocks * (ith + 1) / nth;

    std::vector<float> Vblk((size_t)P * IC * TB);   // V[pos][ic][t]
    std::vector<float> Mblk((size_t)P * TB * OC);   // M[pos][t][oc]
    float dT[P][TB];                                // one channel's tiles, [pos][t]
    float yT[OT*OT][8];                             // one tile's outputs, [k][oc lane]

    for (int64_t b = bbeg; b < bend; ++b) {
        const int64_t t0 = b * TB;
        const int TBcur = (int)std::min<int64_t>(TB, total - t0);

        // Per-tile geometry for this block.
        int iy0s[TB], ix0s[TB], oy0s[TB], ox0s[TB], ns[TB];
        bool inner[TB];
        for (int tb = 0; tb < TBcur; ++tb) {
            const int64_t idx = t0 + tb;
            const int t  = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            ns[tb] = (int)(idx / ntiles);
            oy0s[tb] = ty * OT;       ox0s[tb] = tx * OT;
            iy0s[tb] = ty * OT - pad; ix0s[tb] = tx * OT - pad;
            inner[tb] = iy0s[tb] >= 0 && ix0s[tb] >= 0 &&
                        iy0s[tb] + IT <= H && ix0s[tb] + IT <= W;
        }
        if (TBcur < TB) std::memset(dT, 0, sizeof(dT));   // unused lanes -> V = 0

        // 1. Input transform, vectorised across the block's tiles.
        for (int ic = 0; ic < IC; ++ic) {
            for (int tb = 0; tb < TBcur; ++tb) {
                const float* xc = x + (size_t)ns[tb] * IC * HW + (size_t)ic * HW;
                const int iy0 = iy0s[tb], ix0 = ix0s[tb];
                if (inner[tb]) {
                    const float* row = xc + (size_t)iy0 * W + ix0;
                    for (int i = 0; i < IT; ++i, row += W)
                        for (int j = 0; j < IT; ++j) dT[i*IT + j][tb] = row[j];
                } else {
                    for (int i = 0; i < IT; ++i) {
                        const int yy = iy0 + i;
                        const bool yok = (yy >= 0 && yy < H);
                        for (int j = 0; j < IT; ++j) {
                            const int xx = ix0 + j;
                            dT[i*IT + j][tb] = (yok && xx >= 0 && xx < W)
                                             ? xc[(size_t)yy * W + xx] : 0.0f;
                        }
                    }
                }
            }
            vf8 d[P], v[P];
            for (int pos = 0; pos < P; ++pos) d[pos] = ld8(dT[pos]);
            Pol::inp(d, v);
            float* vb = Vblk.data() + (size_t)ic * TB;
            for (int pos = 0; pos < P; ++pos) st8(vb + (size_t)pos * IC * TB, v[pos]);
        }

        // 2. Winograd-domain blocked GEMM per position.
        for (int pos = 0; pos < P; ++pos)
            wino_gemm_block_neon(U + (size_t)pos * IC * OC,
                                 Vblk.data() + (size_t)pos * IC * TB,
                                 Mblk.data() + (size_t)pos * TB * OC, IC, OC);

        // 3. Output transform (+bias), vectorised across output channels.
        for (int tb = 0; tb < TBcur; ++tb) {
            const int oy0 = oy0s[tb], ox0 = ox0s[tb];
            const bool full = (oy0 + OT <= Hout && ox0 + OT <= Wout);
            float* yn = y + (size_t)ns[tb] * OC * HWout;
            const float* mb = Mblk.data() + (size_t)tb * OC;
            int oc = 0;
            for (; oc + 8 <= OC; oc += 8) {
                vf8 m[P], yv[OT*OT];
                for (int pos = 0; pos < P; ++pos) m[pos] = ld8(mb + (size_t)pos * TB * OC + oc);
                Pol::outp(m, yv);
                if (bias) {
                    const vf8 bv = ld8(bias + oc);
                    for (int k = 0; k < OT*OT; ++k) yv[k] += bv;
                }
                for (int k = 0; k < OT*OT; ++k) st8(yT[k], yv[k]);
                for (int l = 0; l < 8; ++l) {
                    float* yc = yn + (size_t)(oc + l) * HWout;
                    if (full) {
                        float* yr = yc + (size_t)oy0 * Wout + ox0;
                        for (int i = 0; i < OT; ++i, yr += Wout)
                            for (int j = 0; j < OT; ++j) yr[j] = yT[i*OT + j][l];
                    } else {
                        for (int i = 0; i < OT; ++i) {
                            const int oy = oy0 + i; if (oy >= Hout) continue;
                            for (int j = 0; j < OT; ++j) {
                                const int ox = ox0 + j; if (ox >= Wout) continue;
                                yc[(size_t)oy * Wout + ox] = yT[i*OT + j][l];
                            }
                        }
                    }
                }
            }
            for (; oc < OC; ++oc) {
                float mpatch[P], ypatch[OT*OT];
                for (int pos = 0; pos < P; ++pos) mpatch[pos] = mb[(size_t)pos * TB * OC + oc];
                Pol::outp(mpatch, ypatch);
                if (bias) for (int k = 0; k < OT*OT; ++k) ypatch[k] += bias[oc];
                float* yc = yn + (size_t)oc * HWout;
                for (int i = 0; i < OT; ++i) {
                    const int oy = oy0 + i; if (oy >= Hout) continue;
                    for (int j = 0; j < OT; ++j) {
                        const int ox = ox0 + j; if (ox >= Wout) continue;
                        yc[(size_t)oy * Wout + ox] = ypatch[i*OT + j];
                    }
                }
            }
        }
    }
}
#endif // DA_WINO_NEON

static void winograd_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    WinogradState* st = (WinogradState*)userdata;
    const ggml_tensor* wt = dst->src[1];
    const float* w = (const float*)wt->data;

    switch (st->mode) {
        case Mode::F4:
            std::call_once(st->once, [&]{ build_U<F4Policy>(st, w); });
#if defined(DA_WINO_NEON)
            if (!scalar_forced()) { compute_blocked_neon<F4Policy>(st, dst, ith, nth); break; }
#endif
            compute_blocked<F4Policy>(st, dst, ith, nth);
            break;
        case Mode::F2B:
            std::call_once(st->once, [&]{ build_U<F2Policy>(st, w); });
#if defined(DA_WINO_NEON)
            if (!scalar_forced()) { compute_blocked_neon<F2Policy>(st, dst, ith, nth); break; }
#endif
            compute_blocked<F2Policy>(st, dst, ith, nth);
            break;
        case Mode::F2:
        default:
            std::call_once(st->once, [&]{ build_U<F2Policy>(st, w); });
            compute_f2_gemv(st, dst, ith, nth);
            break;
    }
}

// ------------------------------------------------------------------------
// Keyed cache of op states (U transformed once per (filter,shape,mode)).
// ------------------------------------------------------------------------
struct StateKey {
    const void* wdata; int W, H, IC, OC, N, pad; int mode;
    bool operator==(const StateKey& o) const {
        return wdata == o.wdata && W == o.W && H == o.H && IC == o.IC &&
               OC == o.OC && N == o.N && pad == o.pad && mode == o.mode;
    }
};
struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        size_t h = (size_t)k.wdata;
        auto mix = [&h](int v) { h = h * 1000003u + (size_t)(uint32_t)v; };
        mix(k.W); mix(k.H); mix(k.IC); mix(k.OC); mix(k.N); mix(k.pad); mix(k.mode);
        return h;
    }
};

static std::mutex g_states_mtx;
static std::unordered_map<StateKey, WinogradState*, StateKeyHash> g_states;

// F(4x4) outputs in blocks of 4; the output region must cover Wout/Hout. tiles
// step by OT (=2 for F2, =4 for F4).
static int out_tile(Mode m) { return m == Mode::F4 ? 4 : 2; }

static WinogradState* get_state(ggml_tensor* w, ggml_tensor* x, int pad, Mode mode) {
    const int W = (int)x->ne[0], H = (int)x->ne[1], IC = (int)x->ne[2], N = (int)x->ne[3];
    const int OC = (int)w->ne[3];
    StateKey key{ w->data, W, H, IC, OC, N, pad, (int)mode };
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    WinogradState* st = new WinogradState();
    st->mode = mode;
    st->W = W; st->H = H; st->IC = IC; st->OC = OC; st->N = N; st->pad = pad;
    st->Wout = W + 2 * pad - 2;
    st->Hout = H + 2 * pad - 2;
    const int OT = out_tile(mode);
    st->tilesX = (st->Wout + OT - 1) / OT;
    st->tilesY = (st->Hout + OT - 1) / OT;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

#if defined(DA_WINO_NEON)
// ------------------------------------------------------------------------
// 1x1 stride-1 conv straight on the planar [W,H,C] layout (NEON).
//   y[oc][p] = sum_ic w[oc][ic] * x[ic][p] (+ bias[oc])
// The generic path is im2col (a [W,H,IC]->[IC,W,H] permute copy) + sgemm + a
// permute back + a broadcast bias ADD: four full-resolution passes. Here the
// same 8x8 register-blocked kernel runs with the input planes as "U" rows and
// 8 output planes as accumulator rows, so nothing is copied or permuted.
// ------------------------------------------------------------------------
static void conv1x1_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* wt = dst->src[1];                   // [OC, IC]: Wt[ic*OC + oc]
    const float* Wt = (const float*)wt->data;
    const float* bias = dst->src[2] ? (const float*)dst->src[2]->data : nullptr;
    const int OC = (int)wt->ne[0], IC = (int)wt->ne[1];
    const size_t HW = (size_t)xt->ne[0] * xt->ne[1];
    const int N = (int)xt->ne[3];
    // Pixel chunk: IC * CH floats of input stay L1-resident across the oc blocks.
    const size_t CH = (size_t)std::max(64, std::min(1024, (16384 / std::max(IC, 1)) & ~7));
    const size_t nchunks = (HW + CH - 1) / CH;
    const size_t cbeg = nchunks * (size_t)ith / (size_t)nth;
    const size_t cend = nchunks * (size_t)(ith + 1) / (size_t)nth;
    // A short last oc block (OC % 8 != 0) runs from a zero-padded copy of its
    // weights into scratch rows, so the kernel never reads or writes out of range.
    const int ntail = OC % 8, obtail = OC - ntail;
    std::vector<float> wtail, scratch;
    if (ntail) {
        wtail.assign((size_t)IC * 8, 0.0f);
        for (int ic = 0; ic < IC; ++ic)
            for (int t = 0; t < ntail; ++t)
                wtail[(size_t)ic * 8 + t] = Wt[(size_t)ic * OC + obtail + t];
        scratch.resize(8 * CH);
    }

    for (int n = 0; n < N; ++n) {
        const float* xn = (const float*)xt->data + (size_t)n * IC * HW;
        float* yn = (float*)dst->data + (size_t)n * OC * HW;
        for (size_t c = cbeg; c < cend; ++c) {
            const size_t p0 = c * CH;
            const int len = (int)std::min(CH, HW - p0);
            for (int ob = 0; ob < OC; ob += 8) {
                const bool tail = (ob == obtail && ntail);
                const int nb = tail ? ntail : 8;
                float* M = tail ? scratch.data() : yn + (size_t)ob * HW + p0;
                const size_t ms = tail ? CH : HW;
                if (tail) gemm8_neon(xn + p0, HW, wtail.data(), 8, M, ms, IC, len);
                else      gemm8_neon(xn + p0, HW, Wt + ob, (size_t)OC, M, ms, IC, len);
                for (int t = 0; t < nb; ++t) {
                    float* yr = yn + (size_t)(ob + t) * HW + p0;
                    const float* mr = M + (size_t)t * ms;
                    if (bias) { const float bv = bias[ob + t]; for (int i = 0; i < len; ++i) yr[i] = mr[i] + bv; }
                    else if (tail) std::memcpy(yr, mr, (size_t)len * sizeof(float));
                }
            }
        }
    }
}
#endif // DA_WINO_NEON

} // namespace

ggml_tensor* conv1x1_cpu(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* bias) {
#if defined(DA_WINO_NEON)
    static const bool off = std::getenv("DA_CONV1X1") != nullptr &&
                            !std::strcmp(std::getenv("DA_CONV1X1"), "generic");
    const int IC = (int)w->ne[2], OC = (int)w->ne[3];
    if (off || w->ne[0] != 1 || w->ne[1] != 1 || x->ne[2] != IC ||
        w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 ||
        (bias && (bias->type != GGML_TYPE_F32 || ggml_nelements(bias) != OC)))
        return nullptr;
    // Weights [1,1,IC,OC] -> [OC,IC] (ic-major rows of OC) so 8 output channels at a
    // fixed ic are adjacent. A tiny in-graph transpose per forward (<= IC*OC floats)
    // instead of a pointer-keyed cache that could go stale across model reloads.
    ggml_tensor* wt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, w, IC, OC)));
    // The op indexes x as dense planes; make sure it is.
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    ggml_tensor* args[3] = { x, wt, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], OC, x->ne[3],
                          args, bias ? 3 : 2, conv1x1_compute, GGML_N_TASKS_MAX, nullptr);
#else
    (void)ctx; (void)w; (void)x; (void)bias;
    return nullptr;
#endif
}

ggml_tensor* winograd_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad,
                              ggml_tensor* bias) {
    const int OC = (int)w->ne[3];
    const int N  = (int)x->ne[3];
    const int Wout = (int)x->ne[0] + 2 * pad - 2;
    const int Hout = (int)x->ne[1] + 2 * pad - 2;
    const Mode mode = parse_mode();
    WinogradState* st = get_state(w, x, pad, mode);
    // Optional [OC] F32 bias, added inside the op's output transform (saves a
    // separate full-resolution broadcast ADD node per conv).
    ggml_tensor* args[3] = { x, w, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, OC, N,
                          args, bias ? 3 : 2, winograd_compute, GGML_N_TASKS_MAX, st);
}

} // namespace da
