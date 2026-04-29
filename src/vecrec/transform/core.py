from __future__ import annotations
from functools import reduce
from typing import Generator, List, Sequence, Tuple
import typing
from vecrec.expr import *
from vecrec.expr import KernelExpr, SignalExpr, Type
from vecrec.expr.kernel import TIKernel2D
from vecrec.expr.signal_ops import WindowMap, WMVar
from vecrec.expr.signal_vec import RecurseV, SignalVector
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

    # def op_mutual(self, k: Tuple[Tuple[TIKernel]]) -> typing.Optional[Tuple[KernelMatrix, KernelMatrix]]:
    #     assert all(k0.ty == Type.Arith for row in k for k0 in row)
    #     element_type = k[0][0].element_type
    #     assert all(k0.element_type == element_type for row in k for k0 in row)
    #     max_stride = self.max_lanes(element_type)
    #     n = len(k)  # kernel matrix is n x n
    #     # taps = max(len(k[i][j]) for i in range(n) for j in range(n))

    #     stride = 2
    #     while stride <= max_stride:
    #         evens = []
    #         odds = []
    #         odd_len = 0
    #         for i in range(n):
    #             even_row = []
    #             odd_row = []
    #             for j in range(n):
    #                 taps_ij = len(k[i][j])
    #                 even = [0.0] * taps_ij
    #                 even[0:taps_ij:stride] = k[i][j].data[0::stride]
    #                 even_row.append(TIKernel(even, Type.Arith, element_type))

    #                 odd = [a - b for a, b in zip(k[i][j].data, even)]
    #                 odd_row.append(TIKernel(odd, Type.Arith, element_type))

    #                 odd_len = max(odd_len, len(odd))
    #             evens.append(even_row)
    #             odds.append(odd_row)

    #         even_t = tuple(tuple(evens[i]) for i in range(n))
    #         odd_t = tuple(tuple(odds[i]) for i in range(n))

    #         even_k = KernelMatrix(even_t)
    #         odd_k = KernelMatrix(odd_t)

    #         if odd_len == 0:
    #             stride *= 2
    #             continue

    #         f = KernelMatrix.i(n, Type.Arith, element_type) - even_k + odd_k
    #         i = -even_k @ even_k + 2 * even_k + odd_k @ odd_k
    #         return f, i
    #     return None

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

    # def apply_signal_vector(self, expr: SignalVectorExpr) -> Sequence[SignalVectorExpr]:
    #     match expr:
    #         case RecurseV(KernelMatrix(a), g) if all(isinstance(k, TIKernel) for row in a for k in row):
    #             result = self.op_mutual(a)  # type: ignore[arg-type]
    #             if not result:
    #                 return []
    #             f, i = result
    #             return [RecurseV(i, ConvolveV(f, g))]
    #         case _:
    #             return []


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


# @ArithTransform
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
