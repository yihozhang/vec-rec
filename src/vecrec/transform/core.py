from __future__ import annotations
from functools import reduce
from typing import Generator, List, Optional, Sequence, Tuple
import typing
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr, Type
from vecrec.expr.kernel import TIKernel2D
from vecrec.expr.signal_ops import WindowMap, WindowMapVec, WMVar
from vecrec.expr.signal_vec import RecurseV, SignalVector, VectorAt
from vecrec.expr.kernel_matrix import KernelMatrix
from vecrec.factorize import factorize_polynomial, factorize_kernel2d
from vecrec.util import ElementType
from . import Transform, ArithTransform, TropTransform

__all__ = [
    "FuseRecurse",
    "Dilate",
    "DilateTrop",
    "Delay",
    "ComposeRecurse",
    "Factorize",
    "Factorize2D",
    "DilateTVWithSingleOddOrder",
    "DilateLTVArith",
    "DilateLTVTrop",
    "DilateLTVHalfPoints",
    "DilateLTVArithVec",
    "DilateLTVTropVec",
    "DilateLTVHalfPointsVec",
]


@ArithTransform
class FuseRecurse(Transform):
    """Fuse nested IIRs"""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, Recurse(b, g)):
                return [Recurse(KSub(KAdd(a, b), KConvolve(a, b)), g)]
            case _:
                return []


@ArithTransform
class Dilate(Transform):
    """Dilate an IIR"""

    max_bits: int

    def __init__(self, max_bits: int = 2**63):
        self.max_bits = max_bits

    def max_lanes(self, element_type: ElementType) -> int:
        """Get the maximal number of lanes possible given the type"""
        return self.max_bits // element_type.bit_width()

    def op(self, k: TIKernel) -> typing.Optional[Tuple[TIKernel, TIKernel]]:
        assert k.ty == Type.Arith
        max_stride = self.max_lanes(k.element_type)

        stride = 2
        while stride <= max_stride:
            even = [0.0] * len(k)
            even[0::stride] = k.data[0::stride]
            odd = [a - b for a, b in zip(k.data, even)]

            even_k = TIKernel(even, Type.Arith, k.element_type)
            odd_k = TIKernel(odd, Type.Arith, k.element_type)

            if len(odd_k) == 0:
                stride *= 2
                continue

            f = TIKernel.i(Type.Arith, k.element_type) - even_k + odd_k
            i = -even_k * even_k + 2 * even_k + odd_k * odd_k
            return f, i
        return None

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                result = self.op(a)
                if not result:
                    return []
                f, i = result
                return [Recurse(i, Convolve(f, g))]
            case _:
                return []
@TropTransform
class DilateTrop(Transform):
    """Dilate a tropical IIR"""

    max_bits: int

    def __init__(self, max_bits: int = 2**63):
        self.max_bits = max_bits

    def max_lanes(self, element_type: ElementType) -> int:
        """Get the maximal number of lanes possible given the type"""
        return self.max_bits // element_type.bit_width()

    def op(self, k: TIKernel) -> typing.Optional[Tuple[TIKernel, TIKernel]]:
        assert k.ty == Type.TropMax or k.ty == Type.TropMin
        max_stride = self.max_lanes(k.element_type)

        stride = 2
        while stride <= max_stride:
            even = [k.ty.zero()] * len(k)
            even[0::stride] = k.data[0::stride]
            odd = [k.ty.zero() if a == b else a for a, b in zip(k.data, even)]

            even_k = TIKernel(even, k.ty, k.element_type)
            odd_k = TIKernel(odd, k.ty, k.element_type)

            if len(odd_k) == 0:
                stride *= 2
                continue

            f = TIKernel.i(k.ty, k.element_type) + odd_k
            i = even_k + odd_k * odd_k
            return f, i
        return None

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                result = self.op(a)
                if not result:
                    return []
                f, i = result
                return [Recurse(i, Convolve(f, g))]
            case _:
                return []


class Delay(Transform):
    """Delay an IIR by identifying leading coefficient v at feedback p and negating by v*z^-p"""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(a, g) if isinstance(a, TIKernel):
                pos, val = next(
                    ((i, v) for i, v in enumerate(a.data) if not a.ty.is_zero(v)),
                    (len(a), 0),
                )
                if pos == len(a):
                    return []
                new_a = a.copy()
                new_a.data[pos] = expr.ty.zero()
                coeff = TIKernel.z(expr.ty, expr.element_type, -pos) * val
                return [
                    Recurse(
                        new_a + coeff * a,
                        Convolve(TIKernel.i(expr.ty, expr.element_type) + coeff, g),
                    )
                ]
            case _:
                return []


@ArithTransform
class ComposeRecurse(Transform):
    """Compose two IIRs R(a, g) + R(b, h)"""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            # Requires commutativity
            # G/(I-A) + H/(I-B) = (I-B)G/(I-B)(I-A) + (I-A)H/(I-A)(I-B) = ((I-B)G + (I-A)H) / (I - A)(I - B)
            case SAdd(Recurse(TIKernel(_) as a, g), Recurse(TIKernel(_) as b, h)):
                c = KSub(KAdd(a, b), KConvolve(a, b))
                w = SAdd(
                    Convolve(KSub(TIKernel.i(Type.Arith, a.element_type), b), g),
                    Convolve(KSub(TIKernel.i(Type.Arith, b.element_type), a), h),
                )
                return [Recurse(c, w)]
            case _:
                return []


@ArithTransform
class Factorize(Transform):
    """Factorize a TIKernel into products of first-order and second-order factors."""

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Convolve(TIKernel(a) as kernel, g):
                if len(a) < 2:
                    return []  # Constant kernel, nothing to factorize
                factors: Sequence[KernelExpr] = factorize_polynomial(a, kernel.element_type)
                if len(factors) == 0:
                    return []
                # e = reduce(lambda acc, factor: KConvolve(factor, acc), factors)
                e = reduce(lambda acc, factor: Convolve(factor, acc), factors, g)
                return [e]
            case _:
                return []


@ArithTransform
class Factorize2D(Transform):
    """Factorize a separable 2D TIKernel into row and column 1D kernels.

    If the coefficient matrix of TIKernel2D has rank 1, it can be written as
    an outer product: A[r][c] = col_vec[r] * row_vec[c].  In that case:
        Convolve2D(A, S) → Convolve(row_kernel, Convolve2D(col_kernel_2d, S))
    where col_kernel_2d is a single-column 2D kernel.  This splits a 2D
    convolution into two independent 1D passes, reducing complexity from
    O(rows * cols) to O(rows + cols) per output sample.
    """

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Convolve2D(TIKernel2D(_) as kernel, s):
                # Skip single-column kernels: already in factored form; factorizing
                # them again would create an infinite preorder-traversal loop.
                n_cols = max((len(row) for row in kernel.data), default=0)
                if n_cols <= 1:
                    return []
                result = factorize_kernel2d(kernel)
                if result is None:
                    return []
                row_kernel, col_kernel_2d = result
                # Convolve2D goes inside.
                return [Convolve(row_kernel, Convolve2D(col_kernel_2d, s))]
            case _:
                return []


@ArithTransform
class DilateTVWithSingleOddOrder(Transform):
    def all_nonzeros(self, a: Sequence[SignalExpr]) -> List[int]:
        inzs = []
        for i, v in enumerate(a):
            if not (isinstance(v, Num) and v.ty.is_zero(v.value)):
                inzs.append(i)
        return inzs

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(TVKernel(a) as kernel, g):
                # This transformation is at least correct for second order LTV IIR.
                # However, because of the division by zero error in the initial condition. I can't really get it to work
                with with_types(Type.Arith, kernel.element_type):
                    even: List[SignalExpr] = [Num(0)] * len(a)
                    odd: List[SignalExpr] = [Num(0)] * len(a)
                    even[0::2] = a[0::2]
                    odd[1::2] = a[1::2]
                    inzs = self.all_nonzeros(odd)
                    if len(inzs) == 1:
                        inz = inzs[0]
                        A = TVKernel(even)
                        B = TVKernel(odd)
                        C = TVKernel(
                            [
                                (
                                    PointwiseDiv(
                                        PointwiseMul(odd[inz], Convolve(TIKernel.z(n=-1), even[2])),
                                        Convolve(TIKernel.z(n=-i), odd[inz]),
                                    )
                                    if a != Num(0)
                                    else Num(0)
                                )
                                for i, a in enumerate(even)
                            ]
                        )
                        new_numer = [TIKernel.i(), B, KNeg(C)]
                        new_denom = [
                            KConvolve(B, B),
                            A,
                            KConvolve(
                                C,
                                (TIKernel.i() - A),
                            ),
                        ]
                        expr = Recurse(KAdd.of(new_denom), Convolve(KAdd.of(new_numer), g))
                        return [expr]
                    elif len(inzs) == 0:
                        # TODO: dilate with stride greater than two
                        raise NotImplementedError()
                    else:
                        return []

            case _:
                return []


# ---------------------------------------------------------------------------
# LTV (linear time-varying) tower decomposition: dk/fk algorithm
# ---------------------------------------------------------------------------


def _ltv_delay(sig: SignalExpr, n: int, ty: Type, et: ElementType) -> SignalExpr:
    """Return sig delayed by n positions.  Constants are invariant to delay."""
    if n == 0:
        return sig
    if isinstance(sig, Num):
        return sig
    return Convolve(TIKernel.z(ty, et, -n), sig)


def _ltv_mul(a: SignalExpr, b: SignalExpr, ty: Type, et: ElementType) -> SignalExpr:
    """Pointwise semiring-multiply with zero/one simplifications."""
    zero = Num(ty.zero(), ty, et)
    if isinstance(a, Num) and ty.is_zero(a.value):
        return zero
    if isinstance(b, Num) and ty.is_zero(b.value):
        return zero
    if isinstance(a, Num) and ty.is_one(a.value):
        return b
    if isinstance(b, Num) and ty.is_one(b.value):
        return a
    return PointwiseMul(a, b)


def _ltv_add(a: SignalExpr, b: SignalExpr, ty: Type, et: ElementType) -> SignalExpr:
    """Semiring add (+ for arith, min/max for tropical) with zero simplification."""
    if isinstance(a, Num) and ty.is_zero(a.value):
        return b
    if isinstance(b, Num) and ty.is_zero(b.value):
        return a
    return SAdd(a, b)


def _ltv_sub(a: SignalExpr, b: SignalExpr, ty: Type, et: ElementType) -> SignalExpr:
    """Arithmetic subtraction with zero simplification."""
    assert ty == Type.Arith
    if isinstance(b, Num) and ty.is_zero(b.value):
        return a
    return SSub(a, b)


def _build_ltv_tower(
    a_sig: SignalExpr,
    b_sig: SignalExpr,
    g: SignalExpr,
    ty: Type,
    et: ElementType,
    K: int,
    tropical: bool,
) -> Tuple[SignalExpr, SignalExpr, SignalExpr]:
    """Build the dk/fk logarithmic tower for a 2nd-order LTV recurrence.

    d_k(x) is the propagation weight (total contribution of f(x-k) to f(x)
    summed over all paths in the arithmetic case, or minimum-weight path in
    the tropical case).  f_k(x) is the windowed partial sum: the value f(x)
    would take if all positions strictly before x-k+1 had g=zero.

    Base cases: d[0]=1, d[1]=a, f[0]=0, f[1]=g.
    Even doubling formula (same for arith and tropical, different ⊗/⊕):
        d[2k] = d[k] ⊗ d[k](x-k)  ⊕  d[k-1] ⊗ b(x-k+1) ⊗ d[k-1](x-k-1)
        f[2k] = f[k]  ⊕  d[k] ⊗ f[k](x-k)  ⊕  d[k-1] ⊗ b(x-k+1) ⊗ f_bridge(x-k-1)
            where f_bridge = f[k-1] (arithmetic) or f[k] (tropical, wider is fine)
    Odd window formula:
        arithmetic: 3-term inclusion-exclusion for d[2k-1] and f[2k-1]
        tropical:   2-term (idempotency eliminates subtraction) for d[2k-1];
                    no f[2k-1] needed at all

    Returns (d[K], d[K-1], f[K]).
    """

    def delay(sig: SignalExpr, n: int) -> SignalExpr:
        return _ltv_delay(sig, n, ty, et)

    def mul(a: SignalExpr, b: SignalExpr) -> SignalExpr:
        return _ltv_mul(a, b, ty, et)

    def add(a: SignalExpr, b: SignalExpr) -> SignalExpr:
        return _ltv_add(a, b, ty, et)

    def sub(a: SignalExpr, b: SignalExpr) -> SignalExpr:
        return _ltv_sub(a, b, ty, et)

    zero = Num(ty.zero(), ty, et)
    one = Num(ty.one(), ty, et)

    d: dict[int, SignalExpr] = {0: one, 1: a_sig}
    f: dict[int, SignalExpr] = {0: zero, 1: g}

    # Base case: k=1 -> 2k=2
    # d[2] = a(x)*a(x-1) + b(x)   [one*b*one = b simplifies the third term]
    d[2] = add(mul(d[1], delay(d[1], 1)), mul(d[0], mul(b_sig, delay(d[0], 2))))
    # f[2] = g(x) + a(x)*g(x-1)   [f[0]=zero so third term vanishes in both algebras]
    f[2] = add(add(f[1], mul(d[1], delay(f[1], 1))), mul(d[0], mul(b_sig, delay(f[0], 2))))

    # Tower iterations: k = 2, 4, 8, ..., K/2
    k = 2
    while k < K:

        # d[2k]: even window, split at midpoint x-k
        d[2 * k] = add(
            mul(d[k], delay(d[k], k)),
            mul(d[k - 1], mul(delay(b_sig, k - 1), delay(d[k - 1], k + 1))),
        )

        # d[2k-1]: odd window
        if tropical:
            # Two-term: idempotency of ⊕ makes subtraction unnecessary
            d[2 * k - 1] = add(
                mul(d[k], delay(d[k - 1], k)),
                mul(d[k - 1], delay(d[k], k - 1)),
            )
        else:
            # Three-term inclusion-exclusion
            a_del_km1 = delay(a_sig, k - 1)
            d[2 * k - 1] = sub(
                add(mul(d[k], delay(d[k - 1], k)), mul(d[k - 1], delay(d[k], k - 1))),
                mul(d[k - 1], mul(a_del_km1, delay(d[k - 1], k))),
            )

        # f[2k]: even window
        # Tropical uses f[k] as the bridge (wider window is ok by idempotency);
        # arithmetic needs f[k-1] for exact counting.
        f_bridge = f[k] if tropical else f[k - 1]
        f[2 * k] = add(
            add(f[k], mul(d[k], delay(f[k], k))),
            mul(d[k - 1], mul(delay(b_sig, k - 1), delay(f_bridge, k + 1))),
        )

        # f[2k-1]: only needed for arithmetic (used as bridge in next iteration).
        # f[2k](x) covers sources [x-2k+1, x]; f[2k-1](x) covers [x-2k+2, x].
        # The difference is the leftmost source at y = x-2k+1:
        #   f[2k-1](x) = f[2k](x) - g(x-(2k-1)) * d[2k-1](x)
        if not tropical:
            f[2 * k - 1] = sub(
                f[2 * k],
                mul(delay(g, 2 * k - 1), d[2 * k - 1]),
            )

        k *= 2

    return d[K], d[K - 1], f[K]


def _ltv_target_K(max_bits: int, et: ElementType) -> Generator[int, None, None]:
    """Return the largest power of 2 lanes that fit in max_bits for element type et."""
    max_lanes = max_bits // et.bit_width()
    K = 1
    while K * 2 <= max_lanes:
        K *= 2
        yield K


@ArithTransform
class DilateLTVArith(Transform):
    """Vectorize a 2nd-order arithmetic LTV recurrence via the dk/fk tower.

    Transforms::

        Recurse(TVKernel([0, a(x), b(x)]), g)

    into a stride-K recurrence::

        Recurse(TVKernel([0]*K + [d_K(x), d_{K-1}(x)*b(x-K+1)]), f_K(x))

    where K = max_bits // element_type.bit_width() (rounded to power of 2).
    The resulting kernel has K leading zeros so AnnotateLanes assigns K lanes.
    """

    max_bits: int

    def __init__(self, max_bits: int = 2**63) -> None:
        self.max_bits = max_bits

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(TVKernel(data) as kernel, g):
                if len(data) != 3:
                    return []
                if not (isinstance(data[0], Num) and kernel.ty.is_zero(data[0].value)):
                    return []

                ty = kernel.ty
                et = kernel.element_type
                results: List[SignalExpr] = [expr]
                for K in _ltv_target_K(self.max_bits, et):

                    a_sig: SignalExpr = data[1]
                    b_sig: SignalExpr = data[2]

                    d_K, d_Km1, f_K = _build_ltv_tower(a_sig, b_sig, g, ty, et, K, tropical=False)

                    zero = Num(ty.zero(), ty, et)
                    coeff_Kp1 = _ltv_mul(d_Km1, _ltv_delay(b_sig, K - 1, ty, et), ty, et)
                    stitching_data: List[SignalExpr] = [zero] * K + [d_K, coeff_Kp1]
                    results.append(Recurse(TVKernel(stitching_data, ty, et), f_K))
                return results

            case _:
                return []


class DilateLTVHalfPoints(Transform):
    """Vectorize a 2nd-order arithmetic LTV recurrence via the half-points parallel scan.

    Transforms::

        Recurse(TVKernel([0, a(x), b(x)]), g)

    into a WindowMap that processes 2K outputs per K-lane SIMD iteration using the
    half-points matrix prefix scan described in ltv-recurrences-half-points.md.
    """

    max_bits: int

    def __init__(self, max_bits: int = 2**63) -> None:
        self.max_bits = max_bits

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(TVKernel(data) as kernel, g):
                if len(data) != 3:
                    return []
                if not (isinstance(data[0], Num) and kernel.ty.is_zero(data[0].value)):
                    return []

                ty = kernel.ty
                et = kernel.element_type
                a_sig: SignalExpr = data[1]
                b_sig: SignalExpr = data[2]
                results: List[SignalExpr] = [expr]

                for K in [1] + list(_ltv_target_K(self.max_bits, et)):

                    def make_body(wm_inputs: List[List[WMVar]]) -> SignalVectorExpr:
                        a_even = wm_inputs[0][0]
                        a_odd = wm_inputs[0][1]
                        b_even = wm_inputs[1][0]
                        b_odd = wm_inputs[1][1]
                        g_even = wm_inputs[2][0]
                        g_odd = wm_inputs[2][1]

                        def mul(a: SignalExpr, b: SignalExpr) -> SignalExpr:
                            return _ltv_mul(a, b, ty, et)

                        def add(a: SignalExpr, b: SignalExpr) -> SignalExpr:
                            return _ltv_add(a, b, ty, et)

                        def delay(s: SignalExpr, d: int) -> SignalExpr:
                            return _ltv_delay(s, d, ty, et)

                        # Phase 1: M2' (first two rows of the 3x3 affine matrix product)
                        # M2'(x) = M(2x+1) * M(2x)
                        M: List[List[SignalExpr]] = [
                            [
                                add(mul(a_odd, a_even), b_odd),
                                mul(a_odd, b_even),
                                add(mul(a_odd, g_even), g_odd),
                            ],
                            [a_even, b_even, g_even],
                        ]

                        # Phase 2: log2(K) prefix scan layers
                        # Layer ell uses delay d = 2^ell in the half-rate domain
                        for ell in range(K.bit_length() - 1):
                            d = 1 << ell
                            M_new: List[List[SignalExpr]] = [[Num(ty.zero(), ty, et)] * 3 for _ in range(2)]
                            for row in range(2):
                                for col in range(2):
                                    terms = [mul(M[row][m], delay(M[m][col], d)) for m in range(2)]
                                    M_new[row][col] = add(terms[0], terms[1])
                                # Affine column 2: sum + M[row][2]
                                affine = [mul(M[row][m], delay(M[m][2], d)) for m in range(2)]
                                M_new[row][2] = add(add(affine[0], affine[1]), M[row][2])
                            M = M_new

                        # Phase 3: RecurseV — F[0]=f_odd, F[1]=f_even
                        # Kernel has K leading zeros so time_delay(K)=K → lanes=K
                        zero = Num(ty.zero(), ty, et)

                        def make_kernel(m_ij: SignalExpr) -> TVKernel:
                            return TVKernel([zero] * K + [m_ij], ty, et)

                        km = KernelMatrix(
                            [
                                [make_kernel(M[0][0]), make_kernel(M[0][1])],
                                [make_kernel(M[1][0]), make_kernel(M[1][1])],
                            ]
                        )
                        gvec = SignalVector([M[0][2], M[1][2]])
                        return RecurseV(km, gvec)

                    results.append(WindowMap(2, [a_sig, b_sig, g], make_body))
                return results

            case _:
                return []


@TropTransform
class DilateLTVTrop(Transform):
    """Vectorize a 2nd-order tropical LTV recurrence via the dk/fk tower.

    Transforms::

        Recurse(TVKernel([zero, a(x), b(x)]), g)   (TropMin or TropMax type)

    into a stride-K recurrence::

        Recurse(TVKernel([zero]*K + [d_K(x), d_{K-1}(x)⊗b(x-K+1)]), f_K(x))

    The tropical version is simpler: odd-window f arrays are not needed at any
    layer (idempotency of ⊕ absorbs double-counting), and d_{2k-1} uses only
    two terms instead of three.
    """

    max_bits: int

    def __init__(self, max_bits: int = 2**63) -> None:
        self.max_bits = max_bits

    def apply_signal(self, expr: SignalExpr) -> Sequence[SignalExpr]:
        match expr:
            case Recurse(TVKernel(data) as kernel, g):
                if len(data) != 3:
                    return []
                if not (isinstance(data[0], Num) and kernel.ty.is_zero(data[0].value)):
                    return []

                ty = kernel.ty
                et = kernel.element_type
                results: List[SignalExpr] = [expr]
                for K in _ltv_target_K(self.max_bits, et):
                    a_sig: SignalExpr = data[1]
                    b_sig: SignalExpr = data[2]

                    d_K, d_Km1, f_K = _build_ltv_tower(a_sig, b_sig, g, ty, et, K, tropical=True)

                    zero = Num(ty.zero(), ty, et)
                    coeff_Kp1 = _ltv_mul(d_Km1, _ltv_delay(b_sig, K - 1, ty, et), ty, et)
                    stitching_data: List[SignalExpr] = [zero] * K + [d_K, coeff_Kp1]
                    results.append(Recurse(TVKernel(stitching_data, ty, et), f_K))

                return results

            case _:
                return []


# ---------------------------------------------------------------------------
# LTV vector (mutual recurrence) helpers
# ---------------------------------------------------------------------------

MatExpr = List[List[SignalExpr]]
VecExpr = List[SignalExpr]


def _ltvv_zero_mat(n: int, ty: Type, et: ElementType) -> MatExpr:
    return [[Num(ty.zero(), ty, et) for _ in range(n)] for _ in range(n)]


def _ltvv_identity_mat(n: int, ty: Type, et: ElementType) -> MatExpr:
    return [[Num(ty.one(), ty, et) if i == j else Num(ty.zero(), ty, et) for j in range(n)] for i in range(n)]


def _ltvv_zero_vec(n: int, ty: Type, et: ElementType) -> VecExpr:
    return [Num(ty.zero(), ty, et) for _ in range(n)]


def _ltvv_delay_mat(D: MatExpr, k: int, ty: Type, et: ElementType) -> MatExpr:
    return [[_ltv_delay(D[i][j], k, ty, et) for j in range(len(D[0]))] for i in range(len(D))]


def _ltvv_delay_vec(v: VecExpr, k: int, ty: Type, et: ElementType) -> VecExpr:
    return [_ltv_delay(vi, k, ty, et) for vi in v]


def _ltvv_matmul(A: MatExpr, B: MatExpr, ty: Type, et: ElementType) -> MatExpr:
    n = len(A)
    result: MatExpr = _ltvv_zero_mat(n, ty, et)
    for i in range(n):
        for j in range(n):
            for kk in range(n):
                result[i][j] = _ltv_add(result[i][j], _ltv_mul(A[i][kk], B[kk][j], ty, et), ty, et)
    return result


def _ltvv_matvec(A: MatExpr, v: VecExpr, ty: Type, et: ElementType) -> VecExpr:
    n = len(A)
    result: VecExpr = _ltvv_zero_vec(n, ty, et)
    for i in range(n):
        for j in range(n):
            result[i] = _ltv_add(result[i], _ltv_mul(A[i][j], v[j], ty, et), ty, et)
    return result


def _ltvv_matadd(A: MatExpr, B: MatExpr, ty: Type, et: ElementType) -> MatExpr:
    n = len(A)
    return [[_ltv_add(A[i][j], B[i][j], ty, et) for j in range(n)] for i in range(n)]


def _ltvv_matsub(A: MatExpr, B: MatExpr, ty: Type, et: ElementType) -> MatExpr:
    n = len(A)
    return [[_ltv_sub(A[i][j], B[i][j], ty, et) for j in range(n)] for i in range(n)]


def _ltvv_vecadd(u: VecExpr, v: VecExpr, ty: Type, et: ElementType) -> VecExpr:
    return [_ltv_add(ui, vi, ty, et) for ui, vi in zip(u, v)]


def _ltvv_vecsub(u: VecExpr, v: VecExpr, ty: Type, et: ElementType) -> VecExpr:
    return [_ltv_sub(ui, vi, ty, et) for ui, vi in zip(u, v)]


def _build_ltv_tower_vec(
    A1: MatExpr,
    A2: MatExpr,
    g: VecExpr,
    ty: Type,
    et: ElementType,
    n: int,
    K: int,
    tropical: bool,
) -> Tuple[MatExpr, MatExpr, VecExpr]:
    """Build the dk/fk logarithmic tower for a 2nd-order LTV vector recurrence.

    Vector analog of _build_ltv_tower: scalars become n×n matrices (D[k])
    and n-vectors (fv[k]).  The doubling and odd-window formulas are identical
    to the scalar case with matrix multiplication replacing scalar multiply.

    Base cases: D[0]=I_n, D[1]=A1, fv[0]=0, fv[1]=g.

    Returns (D[K], D[K-1], fv[K]).
    """

    def delay_mat(D_k: MatExpr, shift: int) -> MatExpr:
        return _ltvv_delay_mat(D_k, shift, ty, et)

    def delay_vec(v: VecExpr, shift: int) -> VecExpr:
        return _ltvv_delay_vec(v, shift, ty, et)

    def matmul(A: MatExpr, B: MatExpr) -> MatExpr:
        return _ltvv_matmul(A, B, ty, et)

    def matvec(A: MatExpr, v: VecExpr) -> VecExpr:
        return _ltvv_matvec(A, v, ty, et)

    def matadd(A: MatExpr, B: MatExpr) -> MatExpr:
        return _ltvv_matadd(A, B, ty, et)

    def matsub(A: MatExpr, B: MatExpr) -> MatExpr:
        return _ltvv_matsub(A, B, ty, et)

    def vecadd(u: VecExpr, v: VecExpr) -> VecExpr:
        return _ltvv_vecadd(u, v, ty, et)

    def vecsub(u: VecExpr, v: VecExpr) -> VecExpr:
        return _ltvv_vecsub(u, v, ty, et)

    D: dict[int, MatExpr] = {0: _ltvv_identity_mat(n, ty, et), 1: A1}
    fv: dict[int, VecExpr] = {0: _ltvv_zero_vec(n, ty, et), 1: g}

    # Base case: k=1 -> 2k=2
    D[2] = matadd(
        matmul(D[1], delay_mat(D[1], 1)),
        matmul(D[0], matmul(A2, delay_mat(D[0], 2))),
    )
    fv[2] = vecadd(
        vecadd(fv[1], matvec(D[1], delay_vec(fv[1], 1))),
        matvec(D[0], matvec(A2, delay_vec(fv[0], 2))),
    )

    k = 2
    while k < K:
        D[2 * k] = matadd(
            matmul(D[k], delay_mat(D[k], k)),
            matmul(D[k - 1], matmul(delay_mat(A2, k - 1), delay_mat(D[k - 1], k + 1))),
        )

        if tropical:
            D[2 * k - 1] = matadd(
                matmul(D[k], delay_mat(D[k - 1], k)),
                matmul(D[k - 1], delay_mat(D[k], k - 1)),
            )
        else:
            A1_del_km1 = delay_mat(A1, k - 1)
            D[2 * k - 1] = matsub(
                matadd(
                    matmul(D[k], delay_mat(D[k - 1], k)),
                    matmul(D[k - 1], delay_mat(D[k], k - 1)),
                ),
                matmul(D[k - 1], matmul(A1_del_km1, delay_mat(D[k - 1], k))),
            )

        fv_bridge = fv[k] if tropical else fv[k - 1]
        fv[2 * k] = vecadd(
            vecadd(fv[k], matvec(D[k], delay_vec(fv[k], k))),
            matvec(D[k - 1], matvec(delay_mat(A2, k - 1), delay_vec(fv_bridge, k + 1))),
        )

        if not tropical:
            fv[2 * k - 1] = vecsub(
                fv[2 * k],
                matvec(D[2 * k - 1], delay_vec(g, 2 * k - 1)),
            )

        k *= 2

    return D[K], D[K - 1], fv[K]


def _dk_numeric_nonzero(A1: MatExpr, A2: MatExpr, K: int) -> bool:
    """Return True if D[K] is not the all-zero matrix, for constant-Num A1/A2.

    Only evaluates when every entry of A1 and A2 is a Num; returns True
    (don't skip) when any entry is a non-constant signal.
    """
    import numpy as np
    n = len(A1)
    try:
        A1_np = np.array([[A1[i][j].value for j in range(n)] for i in range(n)], dtype=np.float32)  # type: ignore[attr-defined]
        A2_np = np.array([[A2[i][j].value for j in range(n)] for i in range(n)], dtype=np.float32)  # type: ignore[attr-defined]
    except AttributeError:
        return True  # non-Num entry; can't determine, so don't skip
    I_np = np.eye(n)
    D: dict[int, np.ndarray] = {0: I_np, 1: A1_np}
    D[2] = A1_np @ A1_np + A2_np
    k = 2
    while k < K:
        D[2 * k] = D[k] @ D[k] + D[k - 1] @ A2_np @ D[k - 1]
        D[2 * k - 1] = D[k] @ D[k - 1] + D[k - 1] @ D[k] - D[k - 1] @ A1_np @ D[k - 1]
        k *= 2
    return not np.allclose(D[K], 0.0)


def _normalize_km_entry(
    entry: KernelExpr, ty: Type, et: ElementType
) -> Optional[Tuple[SignalExpr, SignalExpr, SignalExpr]]:
    """Normalize a TIKernel or TVKernel entry into (d0, d1, d2) as SignalExpr.

    For TIKernel, accepts length 2 (1st-order) or 3 (2nd-order); missing d2
    defaults to zero. TVKernel must be exactly length 3 — accepting length-2
    TVKernels would re-trigger on K=1 DilateLTVHalfPointsVec body outputs.
    Returns None if the entry doesn't match. d0 must be the ty zero element.
    """
    zero_val = ty.zero()
    if isinstance(entry, TIKernel):
        if len(entry.data) < 2:
            return None
        if not ty.is_zero(entry.data[0]):
            return None
        d2_val = entry.data[2] if len(entry.data) > 2 else zero_val
        return Num(zero_val, ty, et), Num(entry.data[1], ty, et), Num(d2_val, ty, et)
    elif isinstance(entry, TVKernel):
        if len(entry.data) != 3:
            return None
        if not (isinstance(entry.data[0], Num) and ty.is_zero(entry.data[0].value)):
            return None
        return entry.data[0], entry.data[1], entry.data[2]
    return None


@ArithTransform
class DilateLTVArithVec(Transform):
    """Vectorize a 2nd-order arithmetic LTV mutual recurrence via the dk/fk tower.

    Transforms::

        RecurseV(KernelMatrix([[TVKernel([0, A1_ij(x), A2_ij(x)]) ...]]), g)

    into a stride-K mutual recurrence::

        RecurseV(KernelMatrix([[TVKernel([0]*K + [D_K_ij, (D_{K-1}@A2(x-K+1))_ij])]]),
                 SignalVector([f_K_i]))

    where K = max_bits // element_type.bit_width() (rounded to power of 2).
    """

    max_bits: int

    def __init__(self, max_bits: int = 2**63) -> None:
        self.max_bits = max_bits

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        if expr.ty != Type.Arith:
            return []
        match expr:
            case RecurseV(KernelMatrix(data) as km, gvec):
                n = km.n
                ty = km.ty
                et = km.element_type

                norm: List[List[Optional[Tuple[SignalExpr, SignalExpr, SignalExpr]]]] = [
                    [_normalize_km_entry(data[i][j], ty, et) for j in range(n)]
                    for i in range(n)
                ]
                if any(norm[i][j] is None for i in range(n) for j in range(n)):
                    return []

                # Skip degenerate inputs where A1 is all-zero (e.g. K=2 half-points body output)
                if all(
                    isinstance(norm[i][j][1], Num) and ty.is_zero(norm[i][j][1].value)  # type: ignore
                    for i in range(n) for j in range(n)
                ):
                    return []

                A1: MatExpr = [[norm[i][j][1] for j in range(n)] for i in range(n)]  # type: ignore[index]
                A2: MatExpr = [[norm[i][j][2] for j in range(n)] for i in range(n)]  # type: ignore[index]
                g: VecExpr = [VectorAt(gvec, i) for i in range(n)]

                results: List[SignalVectorExpr] = [expr]
                for K in _ltv_target_K(self.max_bits, et):
                    if not _dk_numeric_nonzero(A1, A2, K):
                        continue

                    D_K, D_Km1, f_K = _build_ltv_tower_vec(A1, A2, g, ty, et, n, K, tropical=False)

                    zero = Num(ty.zero(), ty, et)
                    A2_del = _ltvv_delay_mat(A2, K - 1, ty, et)
                    D_Km1_A2 = _ltvv_matmul(D_Km1, A2_del, ty, et)

                    def make_kernel(val_K: SignalExpr, val_Kp1: SignalExpr, _K: int = K, _zero: SignalExpr = zero) -> TVKernel:
                        return TVKernel([_zero] * _K + [val_K, val_Kp1], ty, et)

                    new_km = KernelMatrix([
                        [make_kernel(D_K[i][j], D_Km1_A2[i][j]) for j in range(n)]
                        for i in range(n)
                    ])
                    results.append(RecurseV(new_km, SignalVector(f_K)))

                return results
            case _:
                return []


@TropTransform
class DilateLTVTropVec(Transform):
    """Vectorize a 2nd-order tropical LTV mutual recurrence via the dk/fk tower.

    Transforms::

        RecurseV(KernelMatrix([[TVKernel([zero, A1_ij(x), A2_ij(x)]) ...]]),
                 g)   (TropMin or TropMax type)

    into a stride-K mutual recurrence::

        RecurseV(KernelMatrix([[TVKernel([zero]*K + [D_K_ij, (D_{K-1}⊗A2(x-K+1))_ij])]]),
                 SignalVector([f_K_i]))

    The tropical version uses only 2-term odd-window formulas (idempotency
    eliminates subtraction) and requires no f[2k-1] arrays.
    """

    max_bits: int

    def __init__(self, max_bits: int = 2**63) -> None:
        self.max_bits = max_bits

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        if expr.ty not in (Type.TropMax, Type.TropMin):
            return []
        match expr:
            case RecurseV(KernelMatrix(data) as km, gvec):
                n = km.n
                ty = km.ty
                et = km.element_type

                norm: List[List[Optional[Tuple[SignalExpr, SignalExpr, SignalExpr]]]] = [
                    [_normalize_km_entry(data[i][j], ty, et) for j in range(n)]
                    for i in range(n)
                ]
                if any(norm[i][j] is None for i in range(n) for j in range(n)):
                    return []

                # Skip degenerate inputs where A1 is all-zero (e.g. K=2 half-points body output)
                if all(
                    isinstance(norm[i][j][1], Num) and ty.is_zero(norm[i][j][1].value)  # type: ignore
                    for i in range(n) for j in range(n)
                ):
                    return []

                A1: MatExpr = [[norm[i][j][1] for j in range(n)] for i in range(n)]  # type: ignore[index]
                A2: MatExpr = [[norm[i][j][2] for j in range(n)] for i in range(n)]  # type: ignore[index]
                g: VecExpr = [VectorAt(gvec, i) for i in range(n)]

                results: List[SignalVectorExpr] = [expr]
                for K in _ltv_target_K(self.max_bits, et):
                    if not _dk_numeric_nonzero(A1, A2, K):
                        continue

                    D_K, D_Km1, f_K = _build_ltv_tower_vec(A1, A2, g, ty, et, n, K, tropical=True)

                    zero = Num(ty.zero(), ty, et)
                    A2_del = _ltvv_delay_mat(A2, K - 1, ty, et)
                    D_Km1_A2 = _ltvv_matmul(D_Km1, A2_del, ty, et)

                    def make_kernel(val_K: SignalExpr, val_Kp1: SignalExpr, _K: int = K, _zero: SignalExpr = zero) -> TVKernel:
                        return TVKernel([_zero] * _K + [val_K, val_Kp1], ty, et)

                    new_km = KernelMatrix([
                        [make_kernel(D_K[i][j], D_Km1_A2[i][j]) for j in range(n)]
                        for i in range(n)
                    ])
                    results.append(RecurseV(new_km, SignalVector(f_K)))

                return results
            case _:
                return []


class DilateLTVHalfPointsVec(Transform):
    """Vectorize a 2nd-order arithmetic LTV mutual recurrence via half-points parallel scan.

    Transforms::

        RecurseV(KernelMatrix([[TVKernel([0, A1_ij(x), A2_ij(x)]) ...]]), g)

    into a WindowMapVec that processes 2K outputs per K-lane SIMD iteration.
    The body is a 2n-component RecurseV (n for odd-indexed, n for even-indexed
    outputs), computing the half-points prefix scan generalized to n×n matrix
    coefficients.  Body component 2i is the odd-indexed output for component i
    and body component 2i+1 is the even-indexed output.
    """

    max_bits: int

    def __init__(self, max_bits: int = 2**63) -> None:
        self.max_bits = max_bits

    def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
        if expr.ty != Type.Arith:
            return []
        match expr:
            case RecurseV(KernelMatrix(data) as km, gvec):
                n = km.n
                ty = km.ty
                et = km.element_type

                norm: List[List[Optional[Tuple[SignalExpr, SignalExpr, SignalExpr]]]] = [
                    [_normalize_km_entry(data[i][j], ty, et) for j in range(n)]
                    for i in range(n)
                ]
                if any(norm[i][j] is None for i in range(n) for j in range(n)):
                    return []

                # Skip degenerate inputs where A1 is all-zero (e.g. K=2 half-points body output)
                if all(
                    isinstance(norm[i][j][1], Num) and ty.is_zero(norm[i][j][1].value)  # type: ignore
                    for i in range(n) for j in range(n)
                ):
                    return []

                A1_sigs: MatExpr = [[norm[i][j][1] for j in range(n)] for i in range(n)]  # type: ignore[index]
                A2_sigs: MatExpr = [[norm[i][j][2] for j in range(n)] for i in range(n)]  # type: ignore[index]
                G_sigs: VecExpr = [VectorAt(gvec, i) for i in range(n)]

                # Inputs: A1[i][j] row-major, then A2[i][j] row-major, then G[i]
                wm_inputs_flat = (
                    [A1_sigs[i][j] for i in range(n) for j in range(n)]
                    + [A2_sigs[i][j] for i in range(n) for j in range(n)]
                    + G_sigs
                )

                results: List[SignalVectorExpr] = [expr]
                for K in [1] + list(_ltv_target_K(self.max_bits, et)):

                    def make_body(wm_inputs: List[List[WMVar]]) -> SignalVectorExpr:
                        A1_even: MatExpr = [[wm_inputs[i * n + j][0] for j in range(n)] for i in range(n)]
                        A1_odd: MatExpr = [[wm_inputs[i * n + j][1] for j in range(n)] for i in range(n)]
                        A2_even: MatExpr = [[wm_inputs[n * n + i * n + j][0] for j in range(n)] for i in range(n)]
                        A2_odd: MatExpr = [[wm_inputs[n * n + i * n + j][1] for j in range(n)] for i in range(n)]
                        G_even: VecExpr = [wm_inputs[2 * n * n + i][0] for i in range(n)]
                        G_odd: VecExpr = [wm_inputs[2 * n * n + i][1] for i in range(n)]

                        def matmul(A: MatExpr, B: MatExpr) -> MatExpr:
                            return _ltvv_matmul(A, B, ty, et)

                        def matvec(A: MatExpr, v: VecExpr) -> VecExpr:
                            return _ltvv_matvec(A, v, ty, et)

                        def matadd(A: MatExpr, B: MatExpr) -> MatExpr:
                            return _ltvv_matadd(A, B, ty, et)

                        def vecadd(u: VecExpr, v: VecExpr) -> VecExpr:
                            return _ltvv_vecadd(u, v, ty, et)

                        def delay_mat(D: MatExpr, d: int) -> MatExpr:
                            return _ltvv_delay_mat(D, d, ty, et)

                        def delay_vec(v: VecExpr, d: int) -> VecExpr:
                            return _ltvv_delay_vec(v, d, ty, et)

                        # Phase 1: M_blk(x) = M(2x+1) * M(2x), first 2 block-rows.
                        # M(t) = [[A1(t), A2(t), G(t)], [I, 0, 0], [0, 0, 1]]
                        # Row 0: A1_odd@A1_even + A2_odd, A1_odd@A2_even, A1_odd@G_even + G_odd
                        # Row 1: A1_even, A2_even, G_even
                        # M_blk[row][col<2] is n×n MatExpr; M_blk[row][2] is n-VecExpr.
                        M_blk: List[List] = [
                            [
                                matadd(matmul(A1_odd, A1_even), A2_odd),
                                matmul(A1_odd, A2_even),
                                vecadd(matvec(A1_odd, G_even), G_odd),
                            ],
                            [A1_even, A2_even, G_even],
                        ]

                        # Phase 2: log2(K) prefix scan layers in the half-rate domain.
                        for ell in range(K.bit_length() - 1):
                            d = 1 << ell
                            M_new: List[List] = [[None, None, None], [None, None, None]]
                            for row in range(2):
                                for col in range(2):
                                    M_new[row][col] = matadd(
                                        matmul(M_blk[row][0], delay_mat(M_blk[0][col], d)),
                                        matmul(M_blk[row][1], delay_mat(M_blk[1][col], d)),
                                    )
                                M_new[row][2] = vecadd(
                                    vecadd(
                                        matvec(M_blk[row][0], delay_vec(M_blk[0][2], d)),
                                        matvec(M_blk[row][1], delay_vec(M_blk[1][2], d)),
                                    ),
                                    M_blk[row][2],
                                )
                            M_blk = M_new

                        # Phase 3: 2n-component RecurseV for the stitching step.
                        # Component 2i   = M_blk[0][2][i] (odd-indexed outputs of stream i)
                        # Component 2i+1 = M_blk[1][2][i] (even-indexed outputs of stream i)
                        zero_sig = Num(ty.zero(), ty, et)

                        def make_tv_kernel(m_ij: SignalExpr, _K: int = K) -> TVKernel:
                            return TVKernel([zero_sig] * _K + [m_ij], ty, et)

                        km_rows: List[List[KernelExpr]] = []
                        for ri in range(n):
                            for block_i in range(2):
                                km_row: List[KernelExpr] = []
                                for rj in range(n):
                                    for block_j in range(2):
                                        km_row.append(make_tv_kernel(M_blk[block_i][block_j][ri][rj]))
                                km_rows.append(km_row)
                        new_km = KernelMatrix(km_rows)
                        body_data: VecExpr = []
                        for i in range(n):
                            body_data.append(M_blk[0][2][i])
                            body_data.append(M_blk[1][2][i])
                        return RecurseV(new_km, SignalVector(body_data))

                    results.append(WindowMapVec(2, n, wm_inputs_flat, make_body))

                return results
            case _:
                return []
