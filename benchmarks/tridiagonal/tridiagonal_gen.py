#!/usr/bin/env python
"""Generate vectorized kernels for a batched tridiagonal solver.

This maps the division-free Thomas algorithm (see tridiagonal.py) onto two
vecrec passes. Each system is first normalized by its diagonal (b == 1), so
per system of size n (0-indexed, zero state before index 0):

  Pass 1a (2nd-order TV, determinant minors):
      D[i]  = D[i-1] + km[i] * D[i-2] + kg[i]
      km[i] = -a'[i] * c'[i-1]          (a' = a/b row-aligned, c' = c/b)
      kg    = impulse fixing the D[-1] = 1 boundary: kg[0] = 1, kg[1] = km[1]

  Pass 1b (1st-order TV, rhs numerators):
      Nd[i] = kc[i] * Nd[i-1] + u[i],   kc[i] = -a'[i],  u[i] = d'[i] * D[i-1]

  Pass 2 (1st-order TV, back-substitution on reversed arrays):
      y[j]  = kcr[j] * y[j-1] + qr[j]
      qr[j] = Nd[i]/D[i], kcr[j] = -c'[i]*D[i-1]/D[i]  with i = n-1-j
      x[i]  = y[n-1-i]

Passes 1b and 2 share one generated kernel (vr_tv1); pass 1a uses vr_tv2.

Alternatively, the whole forward pass is one 2-component recurrence vector
(tuple output): [D, Nd] = RecurseV over a lower-triangular 2x2 TV kernel
matrix, with km computed *inside* the kernel (km never touches memory):

      [D ]      [ z^-1 + km*z^-2      0        ] [D ]     [kg]
      [Nd]  =   [ dn*z^-1             an*z^-1  ] [Nd]  +  [hg]

This is emitted as vr_p1 (n_outputs = 2), with variant vr_p1b that fuses the
normalization (rb = 1/b, an, dn) into the kernel too.

A third fused form exploits the triangular structure directly: [D, Nd] as a
tuple of CHAINED scalar recurrences (SignalVector of two Recurse nodes, where
Nd's input expression references D's Recurse — codegen CSE turns that into a
register stream between the two recurrences, inside one kernel). Emitted as
vr_sp1 (n_outputs = 2). Because D's Recurse appears twice in that tuple (as
component 0 and inside Nd's input), the stock candidate enumeration would take
a cartesian product across the two occurrences; vr_sp1 is instead scheduled by
a deterministic mini-pipeline that keeps a single dilation per recurrence.

The C++ driver benchmarks all designs. The divisions all live in elementwise
bookkeeping (vectorizable by the C++ compiler) or vectorized in-kernel
pointwise expressions; the loop-carried recurrences are division-free.

Usage:
  uv run python benchmarks/tridiagonal/tridiagonal_gen.py           # write tridiagonal_kernels.h
  uv run python benchmarks/tridiagonal/tridiagonal_gen.py --sweep   # re-validate/time all candidate
                                                                   # schedules on realistic inputs
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import numpy as np

from vecrec.codegen import CodeGen, generate_kernel_executable, instantiate_kernels
from vecrec.expr import (
    Convolve,
    KernelMatrix,
    Num,
    PointwiseDiv,
    PointwiseMul,
    Recurse,
    RecurseV,
    SAdd,
    SignalExpr,
    SignalVector,
    SNeg,
    TIKernel,
    TVKernel,
    Type,
    Var,
    with_types,
)
from vecrec.optimize import get_candidate_schedules
from vecrec.transform import (
    AnnotateLanes,
    ConstantFold,
    InferNRows,
    Preorder,
    PushDownConvertLanes,
    SaturatePreorder,
    Transform,
    Try,
)
from vecrec.transform.core import DilateLTVArith
from vecrec.util import ElementType

MAX_BITS = 512
HEADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tridiagonal_kernels.h")

# Best schedules found by `--sweep` on an AVX-512 machine (see the sweep table):
# all are delay-8 dilations, ~2-3.7x faster than the undilated reference.
TV2_SCHEDULE_INDEX = 20
TV1_SCHEDULE_INDEX = 12
P1_SCHEDULE_INDEX = 64
P1B_SCHEDULE_INDEX = 72
# vr_sp1's recurrences are dilated to K = SP1_MAX_BITS/32 = 4; deeper dilation
# measured slower for the chained form (the coefficient FIR work grows faster
# than the shrinking chain pays back).
SP1_MAX_BITS = 128


def build_exprs() -> tuple[SignalExpr, SignalExpr, RecurseV, RecurseV]:
    with with_types(Type.Arith, ElementType.Float):
        # D[i] = 1*D[i-1] + km[i]*D[i-2] + kg[i]
        tv2 = Recurse(TVKernel([Num(0.0), Num(1.0), Var("km")]), Var("kg"))
        # f[i] = kc[i]*f[i-1] + kg[i]; the trailing Num(0) makes the kernel
        # 2nd-order in shape so the LTV dilation transforms match it.
        tv1 = Recurse(TVKernel([Num(0.0), Var("kc"), Num(0.0)]), Var("kg"))
        # Fused forward pass: [D, Nd] as a tuple-output recurrence vector.
        # km[i] = an[i]*cn[i-1] is computed in-kernel (an = -a' carries the sign).
        zero = Num(0.0)
        km_expr = PointwiseMul(Var("an"), Convolve(TIKernel.z(n=-1), Var("cn")))
        p1 = RecurseV(
            KernelMatrix(
                [
                    [TVKernel([zero, Num(1.0), km_expr]), TVKernel([zero, Num(0.0), Num(0.0)])],
                    [TVKernel([zero, Var("dn"), Num(0.0)]), TVKernel([zero, Var("an"), Num(0.0)])],
                ]
            ),
            SignalVector([Var("kg"), Var("hg")]),
        )
        # Variant B: normalization fused in-kernel too. The kernel reads the raw
        # subdiagonal/diagonal/rhs and computes rb = 1/bd, an, dn itself; only cn
        # stays materialized (the back-substitution bookkeeping needs it anyway).
        rb_expr = PointwiseDiv(Num(1.0), Var("bd"))
        an_expr = SNeg(PointwiseMul(Var("al"), rb_expr))
        dn_expr = PointwiseMul(Var("rhs"), rb_expr)
        km_expr_b = PointwiseMul(an_expr, Convolve(TIKernel.z(n=-1), Var("cn")))
        p1b = RecurseV(
            KernelMatrix(
                [
                    [TVKernel([zero, Num(1.0), km_expr_b]), TVKernel([zero, Num(0.0), Num(0.0)])],
                    [TVKernel([zero, dn_expr, Num(0.0)]), TVKernel([zero, an_expr, Num(0.0)])],
                ]
            ),
            SignalVector([Var("kg"), Var("hg")]),
        )
    return tv2, tv1, p1, p1b


class _PickLast(Transform):
    """Keep only the deepest dilation variant a transform offers.

    Used to schedule the sp1 tuple deterministically: D's Recurse appears twice
    in that expression (component 0 and inside Nd's input), so enumerating all
    dilation variants per occurrence would explode combinatorially AND produce
    mostly non-matching pairs that codegen CSE cannot share. One variant per
    node keeps Preorder linear and the two occurrences identical.
    """

    def __init__(self, t: Transform) -> None:
        self.t = t

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        r = self.t.apply_signal(expr)
        return [r[-1]] if len(r) > 1 else []


def build_sp1_expr() -> SignalVector:
    """[D, Nd] as a tuple of chained scalar recurrences.

    Nd's input references D's Recurse directly; codegen CSE materializes D
    once, in registers, and feeds both the tuple output and Nd's recurrence.
    """
    with with_types(Type.Arith, ElementType.Float):
        zero = Num(0.0)
        km_expr = PointwiseMul(Var("an"), Convolve(TIKernel.z(n=-1), Var("cn")))
        f_d = Recurse(TVKernel([zero, Num(1.0), km_expr]), Var("kg"))
        nd_input = SAdd(PointwiseMul(Var("dn"), Convolve(TIKernel.z(n=-1), f_d)), Var("hg"))
        f_nd = Recurse(TVKernel([zero, Var("an"), zero]), nd_input)
        return SignalVector([f_d, f_nd])


def build_sp1b_expr() -> SignalVector:
    """[D, Nd, cn] with normalization fused: the kernel reads the raw
    subdiagonal/diagonal/superdiagonal/rhs. rb = 1/bd, an, dn are in-kernel
    expressions with a single consumer each and never touch memory; cn is
    also computed in-kernel but the back-substitution bookkeeping needs it,
    so it is emitted as a third tuple output instead of a separate pass.
    """
    with with_types(Type.Arith, ElementType.Float):
        zero = Num(0.0)
        rb_expr = PointwiseDiv(Num(1.0), Var("bd"))
        an_expr = SNeg(PointwiseMul(Var("al"), rb_expr))
        cn_expr = PointwiseMul(Var("cu"), rb_expr)
        dn_expr = PointwiseMul(Var("rhs"), rb_expr)
        km_expr = PointwiseMul(an_expr, Convolve(TIKernel.z(n=-1), cn_expr))
        f_d = Recurse(TVKernel([zero, Num(1.0), km_expr]), Var("kg"))
        nd_input = SAdd(PointwiseMul(dn_expr, Convolve(TIKernel.z(n=-1), f_d)), Var("hg"))
        f_nd = Recurse(TVKernel([zero, an_expr, zero]), nd_input)
        return SignalVector([f_d, f_nd, cn_expr])


def staged_sp_schedule(expr: SignalVector) -> SignalVector:
    """Deterministic staged schedule for the chained-recurrence tuples: keep
    the first result after every stage (letting the stages cross-multiply
    would enumerate thousands of equivalent fold orders of the duplicated
    subtrees, and non-matching pairs would break codegen CSE sharing)."""
    xs = Preorder(Try(_PickLast(DilateLTVArith(SP1_MAX_BITS)))).apply_generic(expr)
    xs = SaturatePreorder(ConstantFold).apply_generic(xs[0])
    xs = InferNRows().apply_generic(xs[0])
    xs = AnnotateLanes(MAX_BITS).apply_generic(xs[0])
    xs = PushDownConvertLanes().apply_generic(xs[0])
    sched = xs[0]
    assert isinstance(sched, SignalVector)
    return sched


def sp1_schedule() -> SignalVector:
    return staged_sp_schedule(build_sp1_expr())


def sp1b_schedule() -> SignalVector:
    return staged_sp_schedule(build_sp1b_expr())


def generate_header(path: str) -> None:
    tv2, tv1, p1, p1b = build_exprs()
    tv2_scheds = get_candidate_schedules(tv2, MAX_BITS)
    tv1_scheds = get_candidate_schedules(tv1, MAX_BITS)
    p1_scheds = get_candidate_schedules(p1, MAX_BITS)
    p1b_scheds = get_candidate_schedules(p1b, MAX_BITS)
    codes = [
        CodeGen().generate(tv2_scheds[TV2_SCHEDULE_INDEX], "vr_tv2"),
        CodeGen().generate(tv1_scheds[TV1_SCHEDULE_INDEX], "vr_tv1"),
        CodeGen().generate(p1_scheds[P1_SCHEDULE_INDEX], "vr_p1"),
        CodeGen().generate(p1b_scheds[P1B_SCHEDULE_INDEX], "vr_p1b"),
        CodeGen().generate(sp1_schedule(), "vr_sp1"),
        CodeGen().generate(sp1b_schedule(), "vr_sp1b"),
    ]
    instantiate_kernels(path, codes)
    with open(path, "a") as f:
        f.write(f"\nconstexpr int VR_TV2_ELE_PER_ITER = {codes[0].element_per_iter};\n")
        f.write(f"constexpr int VR_TV1_ELE_PER_ITER = {codes[1].element_per_iter};\n")
        f.write(f"constexpr int VR_P1_ELE_PER_ITER = {codes[2].element_per_iter};\n")
        f.write(f"constexpr int VR_P1_N_OUTPUTS = {codes[2].n_outputs};\n")
        f.write(f"constexpr int VR_P1B_ELE_PER_ITER = {codes[3].element_per_iter};\n")
        f.write(f"constexpr int VR_P1B_N_OUTPUTS = {codes[3].n_outputs};\n")
        f.write(f"constexpr int VR_SP1_ELE_PER_ITER = {codes[4].element_per_iter};\n")
        f.write(f"constexpr int VR_SP1_N_OUTPUTS = {codes[4].n_outputs};\n")
        f.write(f"constexpr int VR_SP1B_ELE_PER_ITER = {codes[5].element_per_iter};\n")
        f.write(f"constexpr int VR_SP1B_N_OUTPUTS = {codes[5].n_outputs};\n")
    print(f"wrote {path}")
    print(f"  vr_tv2: schedule k{TV2_SCHEDULE_INDEX}, ele_per_iter={codes[0].element_per_iter}")
    print(f"  vr_tv1: schedule k{TV1_SCHEDULE_INDEX}, ele_per_iter={codes[1].element_per_iter}")
    print(
        f"  vr_p1:  schedule k{P1_SCHEDULE_INDEX}, ele_per_iter={codes[2].element_per_iter}, "
        f"n_outputs={codes[2].n_outputs}"
    )
    print(
        f"  vr_p1b: schedule k{P1B_SCHEDULE_INDEX}, ele_per_iter={codes[3].element_per_iter}, "
        f"n_outputs={codes[3].n_outputs}"
    )
    print(
        f"  vr_sp1: PickLast(DilateLTVArith({SP1_MAX_BITS})), "
        f"ele_per_iter={codes[4].element_per_iter}, n_outputs={codes[4].n_outputs}"
    )
    print(
        f"  vr_sp1b: PickLast(DilateLTVArith({SP1_MAX_BITS})), "
        f"ele_per_iter={codes[5].element_per_iter}, n_outputs={codes[5].n_outputs}"
    )


def _realistic_inputs(
    rng: np.random.Generator, n_systems: int, size: int
) -> dict[str, tuple[dict[str, np.ndarray], np.ndarray]]:
    """Coefficient arrays as produced by normalized diagonally dominant systems."""
    f32 = np.float32
    a = (rng.random((n_systems, size - 1)) - 0.5).astype(f32)
    c = (rng.random((n_systems, size - 1)) - 0.5).astype(f32)
    b = (rng.random((n_systems, size)) + 1.5).astype(f32)
    d = (rng.random((n_systems, size)) * 2 - 1).astype(f32)

    rb = (1.0 / b).astype(f32)
    an = (a * rb[:, 1:]).astype(f32)
    cn = (c * rb[:, :-1]).astype(f32)
    dn = (d * rb).astype(f32)

    km = np.zeros((n_systems, size), dtype=f32)
    km[:, 1:] = -an * cn
    kg1 = np.zeros((n_systems, size), dtype=f32)
    kg1[:, 0] = 1.0
    kg1[:, 1] = km[:, 1]

    # float64 references
    D = np.zeros((n_systems, size), dtype=np.float64)
    km64 = km.astype(np.float64)
    D[:, 0] = 1.0
    D[:, 1] = D[:, 0] + km64[:, 1]
    for i in range(2, size):
        D[:, i] = D[:, i - 1] + km64[:, i] * D[:, i - 2]

    kc = np.zeros((n_systems, size), dtype=f32)
    kc[:, 1:] = -an
    u = np.empty((n_systems, size), dtype=f32)
    u[:, 0] = dn[:, 0]
    u[:, 1:] = dn[:, 1:] * D[:, :-1].astype(f32)

    Nd = np.zeros((n_systems, size), dtype=np.float64)
    kc64, u64 = kc.astype(np.float64), u.astype(np.float64)
    Nd[:, 0] = u64[:, 0]
    for i in range(1, size):
        Nd[:, i] = kc64[:, i] * Nd[:, i - 1] + u64[:, i]

    # Row-aligned inputs of the fused pass (an/cn padded to full length).
    an_full = np.zeros((n_systems, size), dtype=f32)
    an_full[:, 1:] = kc[:, 1:]  # an = -a', an[0] = 0
    cn_full = np.zeros((n_systems, size), dtype=f32)
    cn_full[:, :-1] = cn
    hg = np.zeros((n_systems, size), dtype=f32)
    hg[:, 0] = dn[:, 0]
    # Exact Nd reference for the fused pass (u above rounds D to f32 first).
    Nd_p1 = np.zeros((n_systems, size), dtype=np.float64)
    dn64, an64 = dn.astype(np.float64), an_full.astype(np.float64)
    Nd_p1[:, 0] = dn64[:, 0]
    for i in range(1, size):
        Nd_p1[:, i] = dn64[:, i] * D[:, i - 1] + an64[:, i] * Nd_p1[:, i - 1]

    al_full = np.zeros((n_systems, size), dtype=f32)
    al_full[:, 1:] = a
    cu_full = np.zeros((n_systems, size), dtype=f32)
    cu_full[:, :-1] = c

    p1_inputs = {
        "an": an_full.ravel(),
        "cn": cn_full.ravel(),
        "dn": dn.ravel(),
        "kg": kg1.ravel(),
        "hg": hg.ravel(),
    }
    return {
        "tv2": ({"km": km.ravel(), "kg": kg1.ravel()}, D.ravel()),
        "tv1": ({"kc": kc.ravel(), "kg": u.ravel()}, Nd.ravel()),
        "p1": (p1_inputs, np.stack([D.ravel(), Nd_p1.ravel()])),
        # sp1 takes the same inputs and computes the same pair.
        "sp1": (p1_inputs, np.stack([D.ravel(), Nd_p1.ravel()])),
        "p1b": (
            {
                "al": al_full.ravel(),
                "bd": b.ravel(),
                "rhs": d.ravel(),
                "cn": cn_full.ravel(),
                "kg": kg1.ravel(),
                "hg": hg.ravel(),
            },
            np.stack([D.ravel(), Nd_p1.ravel()]),
        ),
        # sp1b reads the raw system and additionally emits cn.
        "sp1b": (
            {
                "al": al_full.ravel(),
                "bd": b.ravel(),
                "cu": cu_full.ravel(),
                "rhs": d.ravel(),
                "kg": kg1.ravel(),
                "hg": hg.ravel(),
            },
            np.stack([D.ravel(), Nd_p1.ravel(), cn_full.astype(np.float64).ravel()]),
        ),
    }


def sweep() -> None:
    """Validate and time every candidate schedule on realistic inputs.

    The stock benchmark harness draws coefficients from U[-1,1], which makes
    the kb=1 determinant recurrence blow up over a 1024-long system and fail
    validation spuriously; here the coefficients come from actual normalized
    diagonally dominant systems.
    """
    size, n_systems = 1024, 256
    n = size * n_systems
    data = _realistic_inputs(np.random.default_rng(42), n_systems, size)

    tv2, tv1, p1, p1b = build_exprs()
    targets: list[tuple[str, list]] = [
        ("tv2", get_candidate_schedules(tv2, MAX_BITS)),
        ("tv1", get_candidate_schedules(tv1, MAX_BITS)),
        ("p1", get_candidate_schedules(p1, MAX_BITS)),
        ("p1b", get_candidate_schedules(p1b, MAX_BITS)),
        # sp1/sp1b have one deterministic schedule each (see staged_sp_schedule).
        ("sp1", [sp1_schedule()]),
        ("sp1b", [sp1b_schedule()]),
    ]
    for tag, scheds in targets:
        inputs, want = data[tag]
        print(f"\n=== {tag}: {len(scheds)} candidate schedules ===")
        best: tuple[float, int] | None = None
        for i, s in enumerate(scheds):
            ke = generate_kernel_executable(CodeGen(), s, row_size=size)
            best_us = np.inf
            out = None
            for _ in range(3):
                o, us, _ = ke.run(inputs, n=n)
                if us < best_us:
                    best_us, out = us, o
            assert out is not None
            # `want` is 1-D for single-output kernels, (n_outputs, n) for tuples.
            err = float(np.max(np.abs(out.astype(np.float64) - want) / np.maximum(np.abs(want), 1e-3)))
            ns = best_us * 1000.0 / n
            ok = err < 1e-3
            print(f"  k{i:3d}: {ns:7.3f} ns/elem  relerr={err:.2e}  {'OK' if ok else 'BAD'}")
            if ok and (best is None or ns < best[0]):
                best = (ns, i)
        assert best is not None, f"no valid schedule for {tag}"
        print(f"  best: k{best[1]} at {best[0]:.3f} ns/elem")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="time all candidate schedules")
    parser.add_argument("--output", default=HEADER_PATH, help="header output path")
    args = parser.parse_args()
    if args.sweep:
        sweep()
    else:
        generate_header(args.output)


if __name__ == "__main__":
    main()
