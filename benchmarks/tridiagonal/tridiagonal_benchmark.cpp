// Batched tridiagonal solver: vecrec-vectorized division-free Thomas vs scalar.
//
// Solves S independent tridiagonal systems of size R each, one system at a
// time. Coefficients are staged through per-system scratch arrays (length R,
// L1-resident) computed right before the kernel loop that consumes them, so
// nothing is swept at batch granularity and no chunking is needed. The kernel
// loops themselves contain nothing but run() calls: interleaving coefficient
// blocks between run() calls was measured 1.5-2x slower, because the kernels'
// live vector state then spills around the interleaved code every iteration.
// Reports the wall-clock time and throughput for solving the whole batch
// (N = S*R elements, 2^20 by default).
//
// Build:
//   uv run python benchmarks/tridiagonal/tridiagonal_gen.py      # writes tridiagonal_kernels.h
//   clang++ -std=c++20 -O3 -march=native benchmarks/tridiagonal/tridiagonal_benchmark.cpp -o benchmarks/tridiagonal/tridiagonal_benchmark
// Run:
//   benchmarks/tridiagonal/tridiagonal_benchmark [S] [R] [iters]
//
// Methods compared (all single-threaded, float32). Speedups are relative to
// divfree_scalar, the algorithm-parity baseline:
//   divfree_scalar  scalar version of the same two-pass formulation the vecrec
//                   pipelines use. "Division-free" refers to the recurrence
//                   chains: every division (1/b, 1/D) depends only on inputs or
//                   already-computed arrays, never on the previous iteration,
//                   so the loop-carried chains are pure FMA (vs. classic Thomas,
//                   whose pivot division sits on the chain and costs ~3x more)
//   vecrec split    forward pass as two kernels, vr_tv2 (D) then vr_tv1 (Nd),
//                   with km and u = d'*D[i-1] staged through L1 scratch
//   vecrec fused    forward pass as ONE tuple-output recurrence-vector kernel
//                   vr_p1 that emits D and Nd together; km = -a'*c' is
//                   computed inside the kernel and never touches memory
//   vecrec fused-b  like fused, but normalization is fused too: the kernel
//                   reads the raw subdiagonal/diagonal/rhs and computes
//                   rb = 1/b, an, dn as in-kernel expressions; only cn is
//                   materialized (the back-substitution needs it)
//   vecrec sp1      forward pass as ONE tuple-output kernel vr_sp1 holding the
//                   two recurrences CHAINED: Nd's Recurse consumes D's Recurse
//                   directly as a register stream (sequential triangular
//                   structure inside one kernel; no km/u materialization)
//   vecrec sp1b     sp1 with normalization fused too: the kernel reads the
//                   raw system (al/bd/cu/rhs) and computes rb = 1/b, an, dn
//                   in-kernel (never materialized); cn — needed later by the
//                   back-substitution — is emitted as a THIRD tuple output,
//                   so no coefficient loop remains before the forward pass
//
// Math (per system, after normalizing each row by its diagonal so b == 1;
// a'[i] is the subdiagonal coefficient of row i, row-aligned, a'[0] = 0;
// c'[i] the superdiagonal coefficient of row i, c'[R-1] = 0):
//
//   pass 1a:  D[i]  = D[i-1] + km[i]*D[i-2] + kg[i]      (vr_tv2)
//             km[i] = -a'[i]*c'[i-1], kg = {1, km[1], 0, 0, ...}
//             (kg is the impulse that plants the true boundary D[-1] = 1
//              into vecrec's zero-initialized recurrence)
//   pass 1b:  Nd[i] = (-a'[i])*Nd[i-1] + d'[i]*D[i-1]    (vr_tv1)
//   pass 2:   back-substitution x[i] = q[i] - w[i]*x[i+1] run as a forward
//             recurrence on per-system-reversed arrays:
//             y[j] = kcr[j]*y[j-1] + qr[j]               (vr_tv1)
//             qr[j] = Nd[i]/D[i], kcr[j] = -c'[i]*D[i-1]/D[i], i = R-1-j
//
// All divisions live in elementwise coefficient blocks (or vectorized
// in-kernel pointwise expressions); the loop-carried recurrences are
// division-free.

#include "tridiagonal_kernels.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Buf {
    float *p;
    explicit Buf(size_t n) : p(new (std::align_val_t(64)) float[n]) { std::memset(p, 0, n * sizeof(float)); }
    ~Buf() { ::operator delete[](p, std::align_val_t(64)); }
    Buf(const Buf &) = delete;
    Buf &operator=(const Buf &) = delete;
    operator float *() const { return p; }
};

double now_us() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------
// Baseline: scalar division-free two-pass (same math as the vecrec
// pipelines; divisions off the recurrence chains, chains are pure FMA).
// ---------------------------------------------------------------------------
void solve_divfree_scalar(const float *al, const float *bd, const float *cu, const float *rhs,
                          float *x, int S, int R, float *D, float *Nd, float *cn,
                          float *q, float *w) {
    for (int s = 0; s < S; ++s) {
        const size_t base = (size_t)s * R;
        const float *a = al + base, *b = bd + base, *c = cu + base, *d = rhs + base;
        float *xs = x + base;

        // forward: D and Nd recurrences (divisions feed the chain but are not on it)
        float Dp = 1.0f, Dpp = 0.0f, Ndp = 0.0f, cn_prev = 0.0f;
        for (int i = 0; i < R; ++i) {
            float r = 1.0f / b[i];
            float an = -a[i] * r;  // -a'
            float cni = c[i] * r;
            float dn = d[i] * r;
            float Di = Dp + (an * cn_prev) * Dpp;
            float Ndi = an * Ndp + dn * Dp;
            D[i] = Di;
            Nd[i] = Ndi;
            cn[i] = cni;
            Dpp = Dp;
            Dp = Di;
            Ndp = Ndi;
            cn_prev = cni;
        }
        // elementwise: q = Nd/D, w[i] = c'[i]*D[i-1]/D[i] (clang auto-vectorizes)
        for (int i = 0; i < R; ++i) {
            float rD = 1.0f / D[i];
            q[i] = Nd[i] * rD;
            w[i] = cn[i] * (i > 0 ? D[i - 1] : 1.0f) * rD;
        }
        // backward: pure FMA chain
        xs[R - 1] = q[R - 1];
        for (int i = R - 2; i >= 0; --i) {
            xs[i] = q[i] - w[i] * xs[i + 1];
        }
    }
}

// ---------------------------------------------------------------------------
// vecrec pipelines. Everything is processed one system at a time: the
// per-system scratch arrays (length R) stay L1-resident between the
// coefficient loops and the kernel loops that consume them, so there is no
// chunk-level sweeping. Kernel loops contain nothing but run() calls — mixing
// other work into them makes the kernels' live vector state spill every
// iteration (measured 1.5-2x slower). Kernels are constructed fresh per
// system (zero recurrence state), so reset_and_next_row is never needed.
// ---------------------------------------------------------------------------

constexpr int EPI = VR_TV1_ELE_PER_ITER;
static_assert(EPI == VR_TV2_ELE_PER_ITER && EPI == VR_P1_ELE_PER_ITER &&
                  EPI == VR_P1B_ELE_PER_ITER && EPI == VR_SP1_ELE_PER_ITER &&
                  EPI == VR_SP1B_ELE_PER_ITER,
              "the block loops assume one common ele_per_iter");

struct VecrecScratch {
    // per-system buffers; kg/hg are zero except boundary slots
    Buf an, cn, dn, km, kg, hg, u, D, Nd, kcr, qr, y;
    explicit VecrecScratch(size_t r)
        : an(r), cn(r), dn(r), km(r), kg(r), hg(r), u(r), D(r), Nd(r), kcr(r), qr(r), y(r) {}
};

// Shared tail of the vecrec pipelines: back-substitution for one system.
// qr[j] = Nd[i]/D[i], kcr[j] = -c'[i]*D[i-1]/D[i] with i = R-1-j; kcr[0]
// lands on 0 automatically because the row-aligned superdiagonal has
// cn[R-1] = 0. y[j] = kcr[j]*y[j-1] + qr[j], un-reversed into x.
void back_substitute_system(const float *cn, const float *D, const float *Nd,
                            VecrecScratch &sc, float *xs, int R) {
    float *kcr = sc.kcr, *qr = sc.qr, *y = sc.y;
    for (int j = 0; j < R - 1; ++j) {
        const int i = R - 1 - j;
        const float rD = 1.0f / D[i];
        qr[j] = Nd[i] * rD;
        kcr[j] = -cn[i] * D[i - 1] * rD;
    }
    // j = R-1 (i = 0): D[-1] = 1
    qr[R - 1] = Nd[0] / D[0];
    kcr[R - 1] = -cn[0] / D[0];

    vr_tv1 k(kcr, qr, R);
    for (int col = 0; col < R; col += EPI) {
        auto *op = reinterpret_cast<vr_tv1_vector_type *>(y + col);
        k.run(&op);
    }
    for (int j = 0; j < R; ++j) {
        xs[R - 1 - j] = y[j];
    }
}

// Split pipeline: D and Nd each computed by their own kernel, with km and
// u = d'*D[i-1] staged through L1-resident per-system arrays.
void solve_vecrec_split(const float *al, const float *bd, const float *cu, const float *rhs,
                        float *x, int S, int R, VecrecScratch &sc) {
    float *an = sc.an, *cn = sc.cn, *dn = sc.dn, *km = sc.km, *kg = sc.kg, *u = sc.u;
    float *D = sc.D, *Nd = sc.Nd;
    for (int s = 0; s < S; ++s) {
        const size_t base = (size_t)s * R;
        const float *a = al + base, *b = bd + base, *c = cu + base, *d = rhs + base;

        for (int t = 0; t < R; ++t) {
            float r = 1.0f / b[t];
            an[t] = -a[t] * r;  // an = -a' (the sign every consumer wants)
            cn[t] = c[t] * r;
            dn[t] = d[t] * r;
        }
        // km[t] = an[t]*cn[t-1]; km[0] = 0 (an[0] = 0 anyway)
        km[0] = 0.0f;
        for (int t = 1; t < R; ++t) {
            km[t] = an[t] * cn[t - 1];
        }
        kg[0] = 1.0f;  // kg zero elsewhere (buffer zero-initialized once)
        kg[1] = km[1];

        vr_tv2 k1(kg, km, R);
        for (int col = 0; col < R; col += EPI) {
            auto *op = reinterpret_cast<vr_tv2_vector_type *>(D + col);
            k1.run(&op);
        }

        u[0] = dn[0];
        for (int t = 1; t < R; ++t) {
            u[t] = dn[t] * D[t - 1];
        }
        vr_tv1 k2(an, u, R);
        for (int col = 0; col < R; col += EPI) {
            auto *op = reinterpret_cast<vr_tv1_vector_type *>(Nd + col);
            k2.run(&op);
        }

        back_substitute_system(cn, D, Nd, sc, x + base, R);
    }
}

// Fused pipelines: the whole forward pass is ONE tuple-output kernel that
// emits D and Nd together, with km computed in-kernel. Two kernels share this
// driver shape (identical constructor signatures):
//   vr_p1   [D, Nd] as a 2x2 RecurseV (recurrence-vector dilation)
//   vr_sp1  [D, Nd] as a tuple of chained scalar recurrences: Nd's Recurse
//           consumes D's Recurse directly as a register stream (the
//           sequential triangular structure, inside one kernel)
template <typename Kernel, typename VecT>
void solve_fused_impl(const float *al, const float *bd, const float *cu, const float *rhs,
                      float *x, int S, int R, VecrecScratch &sc) {
    float *an = sc.an, *cn = sc.cn, *dn = sc.dn, *kg = sc.kg, *hg = sc.hg;
    float *D = sc.D, *Nd = sc.Nd;
    for (int s = 0; s < S; ++s) {
        const size_t base = (size_t)s * R;
        const float *a = al + base, *b = bd + base, *c = cu + base, *d = rhs + base;

        for (int t = 0; t < R; ++t) {
            float r = 1.0f / b[t];
            an[t] = -a[t] * r;
            cn[t] = c[t] * r;
            dn[t] = d[t] * r;
        }
        kg[0] = 1.0f;
        kg[1] = an[1] * cn[0];  // km[1]
        hg[0] = dn[0];

        Kernel k(an, cn, dn, hg, kg, R);
        for (int col = 0; col < R; col += EPI) {
            VecT *out[2] = {
                reinterpret_cast<VecT *>(D + col),
                reinterpret_cast<VecT *>(Nd + col),
            };
            k.run(out);
        }

        back_substitute_system(cn, D, Nd, sc, x + base, R);
    }
}

void solve_vecrec_fused(const float *al, const float *bd, const float *cu, const float *rhs,
                        float *x, int S, int R, VecrecScratch &sc) {
    solve_fused_impl<vr_p1, vr_p1_vector_type>(al, bd, cu, rhs, x, S, R, sc);
}

void solve_vecrec_sp1(const float *al, const float *bd, const float *cu, const float *rhs,
                      float *x, int S, int R, VecrecScratch &sc) {
    solve_fused_impl<vr_sp1, vr_sp1_vector_type>(al, bd, cu, rhs, x, S, R, sc);
}

// sp1b: sp1 with normalization fused in-kernel and cn emitted as a third
// tuple output. NO coefficient loop remains before the forward pass — the
// per-system prologue is just the three boundary impulse scalars.
void solve_vecrec_sp1b(const float *al, const float *bd, const float *cu, const float *rhs,
                       float *x, int S, int R, VecrecScratch &sc) {
    float *cn = sc.cn, *kg = sc.kg, *hg = sc.hg;
    float *D = sc.D, *Nd = sc.Nd;
    for (int s = 0; s < S; ++s) {
        const size_t base = (size_t)s * R;
        const float *a = al + base, *b = bd + base, *c = cu + base, *d = rhs + base;

        kg[0] = 1.0f;
        kg[1] = (-a[1] / b[1]) * (c[0] / b[0]);  // km[1] = an[1]*cn[0]
        hg[0] = d[0] / b[0];                     // dn[0]

        vr_sp1b k(a, b, c, hg, kg, d, R);
        for (int col = 0; col < R; col += EPI) {
            vr_sp1b_vector_type *out[VR_SP1B_N_OUTPUTS] = {
                reinterpret_cast<vr_sp1b_vector_type *>(D + col),
                reinterpret_cast<vr_sp1b_vector_type *>(Nd + col),
                reinterpret_cast<vr_sp1b_vector_type *>(cn + col),
            };
            k.run(out);
        }

        back_substitute_system(cn, D, Nd, sc, x + base, R);
    }
}

// Fused pipeline, variant B: normalization also fused into the kernel. The
// only materialized coefficient array is cn (the back-substitution needs it);
// an/dn/rb never touch memory.
void solve_vecrec_fused_b(const float *al, const float *bd, const float *cu, const float *rhs,
                          float *x, int S, int R, VecrecScratch &sc) {
    float *cn = sc.cn, *kg = sc.kg, *hg = sc.hg;
    float *D = sc.D, *Nd = sc.Nd;
    for (int s = 0; s < S; ++s) {
        const size_t base = (size_t)s * R;
        const float *a = al + base, *b = bd + base, *c = cu + base, *d = rhs + base;

        for (int t = 0; t < R; ++t) {
            cn[t] = c[t] / b[t];
        }
        kg[0] = 1.0f;
        kg[1] = (-a[1] / b[1]) * cn[0];  // km[1]
        hg[0] = d[0] / b[0];

        vr_p1b k(a, b, cn, hg, kg, d, R);
        for (int col = 0; col < R; col += EPI) {
            vr_p1b_vector_type *out[VR_P1B_N_OUTPUTS] = {
                reinterpret_cast<vr_p1b_vector_type *>(D + col),
                reinterpret_cast<vr_p1b_vector_type *>(Nd + col),
            };
            k.run(out);
        }

        back_substitute_system(cn, D, Nd, sc, x + base, R);
    }
}

// ---------------------------------------------------------------------------
// float64 reference + error measurement
// ---------------------------------------------------------------------------
void solve_thomas_f64(const float *al, const float *bd, const float *cu, const float *rhs,
                      double *x, int S, int R, double *cp, double *dp) {
    for (int s = 0; s < S; ++s) {
        const size_t base = (size_t)s * R;
        const float *a = al + base, *b = bd + base, *c = cu + base, *d = rhs + base;
        double *xs = x + base;
        cp[0] = (double)c[0] / b[0];
        dp[0] = (double)d[0] / b[0];
        for (int i = 1; i < R; ++i) {
            double m = 1.0 / (b[i] - a[i] * cp[i - 1]);
            cp[i] = c[i] * m;
            dp[i] = (d[i] - a[i] * dp[i - 1]) * m;
        }
        xs[R - 1] = dp[R - 1];
        for (int i = R - 2; i >= 0; --i) {
            xs[i] = dp[i] - cp[i] * xs[i + 1];
        }
    }
}

double max_rel_err(const float *x, const double *ref, size_t n) {
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
        double denom = std::abs(ref[i]) > 1e-3 ? std::abs(ref[i]) : 1e-3;
        double e = std::abs((double)x[i] - ref[i]) / denom;
        if (e > worst) worst = e;
    }
    return worst;
}

template <typename F>
std::pair<double, double> time_method(F &&f, int iters) {  // (min_us, avg_us)
    double best = 1e300, sum = 0;
    f();
    f();  // warmup
    for (int it = 0; it < iters; ++it) {
        double t0 = now_us();
        f();
        double dt = now_us() - t0;
        best = std::min(best, dt);
        sum += dt;
    }
    return {best, sum / iters};
}

}  // namespace

int main(int argc, char **argv) {
    int S = argc > 1 ? std::atoi(argv[1]) : 1024;
    int R = argc > 2 ? std::atoi(argv[2]) : 1024;
    int iters = argc > 3 ? std::atoi(argv[3]) : 20;

    if (R % EPI != 0 || R < 2 * EPI) {
        std::fprintf(stderr, "R must be a multiple of %d (and at least %d)\n", EPI, 2 * EPI);
        return 1;
    }
    const size_t N = (size_t)S * R;

    std::printf("batched tridiagonal solve: S=%d systems, R=%d unknowns each (N=%zu), %d iters\n\n",
                S, R, N, iters);

    // ---- inputs: diagonally dominant systems, row-aligned layout ----
    // al[i] multiplies x[i-1] (al[0]=0), cu[i] multiplies x[i+1] (cu[R-1]=0)
    Buf al(N), bd(N), cu(N), rhs(N);
    {
        std::mt19937 gen(12345);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f), s05(-0.5f, 0.5f), s1(-1.0f, 1.0f);
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < R; ++i) {
                size_t t = (size_t)s * R + i;
                al[t] = (i == 0) ? 0.0f : s05(gen);
                cu[t] = (i == R - 1) ? 0.0f : s05(gen);
                bd[t] = 1.5f + u01(gen);
                rhs[t] = s1(gen);
            }
        }
    }

    // ---- float64 reference ----
    std::vector<double> x_ref(N), cp64(R), dp64(R);
    solve_thomas_f64(al, bd, cu, rhs, x_ref.data(), S, R, cp64.data(), dp64.data());

    Buf x_divfree(N), x_split(N), x_fused(N), x_fusedb(N), x_sp1(N), x_sp1b(N);
    Buf sD(R), sNd(R), scn(R), sq(R), sw(R);
    VecrecScratch scratch(R);

    // correctness first
    solve_divfree_scalar(al, bd, cu, rhs, x_divfree, S, R, sD, sNd, scn, sq, sw);
    solve_vecrec_split(al, bd, cu, rhs, x_split, S, R, scratch);
    solve_vecrec_fused(al, bd, cu, rhs, x_fused, S, R, scratch);
    solve_vecrec_fused_b(al, bd, cu, rhs, x_fusedb, S, R, scratch);
    solve_vecrec_sp1(al, bd, cu, rhs, x_sp1, S, R, scratch);
    solve_vecrec_sp1b(al, bd, cu, rhs, x_sp1b, S, R, scratch);
    std::printf("max rel error vs float64 Thomas:\n");
    std::printf("  divfree scalar (f32):  %.3e\n", max_rel_err(x_divfree, x_ref.data(), N));
    std::printf("  vecrec split (f32):    %.3e\n", max_rel_err(x_split, x_ref.data(), N));
    std::printf("  vecrec fused (f32):    %.3e\n", max_rel_err(x_fused, x_ref.data(), N));
    std::printf("  vecrec fused-b (f32):  %.3e\n", max_rel_err(x_fusedb, x_ref.data(), N));
    std::printf("  vecrec sp1 (f32):      %.3e\n", max_rel_err(x_sp1, x_ref.data(), N));
    std::printf("  vecrec sp1b (f32):     %.3e\n\n", max_rel_err(x_sp1b, x_ref.data(), N));

    // ---- timing ----
    // Spin for a moment so frequency-scaling governors ramp the core up
    // before we start measuring (matters under the powersave governor).
    {
        volatile double spin = 1.0;
        double t0 = now_us();
        while (now_us() - t0 < 8e5) spin = spin * 1.0000001 + 1e-9;
    }
    volatile float sink = 0;
    auto consume = [&](float *x) { sink += x[N / 2]; };

    auto [divfree_min, divfree_avg] = time_method(
        [&] { solve_divfree_scalar(al, bd, cu, rhs, x_divfree, S, R, sD, sNd, scn, sq, sw); consume(x_divfree); },
        iters);
    auto [split_min, split_avg] = time_method(
        [&] { solve_vecrec_split(al, bd, cu, rhs, x_split, S, R, scratch); consume(x_split); }, iters);
    auto [fused_min, fused_avg] = time_method(
        [&] { solve_vecrec_fused(al, bd, cu, rhs, x_fused, S, R, scratch); consume(x_fused); }, iters);
    auto [fusedb_min, fusedb_avg] = time_method(
        [&] { solve_vecrec_fused_b(al, bd, cu, rhs, x_fusedb, S, R, scratch); consume(x_fusedb); }, iters);
    auto [sp1_min, sp1_avg] = time_method(
        [&] { solve_vecrec_sp1(al, bd, cu, rhs, x_sp1, S, R, scratch); consume(x_sp1); }, iters);
    auto [sp1b_min, sp1b_avg] = time_method(
        [&] { solve_vecrec_sp1b(al, bd, cu, rhs, x_sp1b, S, R, scratch); consume(x_sp1b); }, iters);

    auto report = [&](const char *name, double min_us, double avg_us) {
        std::printf("  %-16s %9.3f ms  (avg %9.3f)  %8.1f Melem/s  speedup vs divfree: %.2fx\n",
                    name, min_us / 1e3, avg_us / 1e3, (double)N / min_us, divfree_min / min_us);
    };
    std::printf("time to solve the whole batch (N=%zu; min over %d iters):\n", N, iters);
    report("divfree scalar", divfree_min, divfree_avg);
    report("vecrec split", split_min, split_avg);
    report("vecrec fused", fused_min, fused_avg);
    report("vecrec fused-b", fusedb_min, fusedb_avg);
    report("vecrec sp1", sp1_min, sp1_avg);
    report("vecrec sp1b", sp1b_min, sp1b_avg);

    (void)sink;
    return 0;
}
