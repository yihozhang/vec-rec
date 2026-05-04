from __future__ import annotations

from typing import Callable, Optional

from vecrec.expr.base import Type


class CustomSemiring(Type):
    """A user-defined semiring with custom plus/times operators and C++ code strings."""

    cpp_zero_init: str
    cpp_mul_op: str
    cpp_add_op: str
    cpp_sadd_op: str
    cpp_pmul_op: str
    cpp_ssub_op: str

    def __init__(
        self,
        name: str,
        zero: float,
        one: float,
        plus: Callable[[float, float], float],
        times: Callable[[float, float], float],
        is_zero_fn: Optional[Callable[[float], bool]] = None,
        is_one_fn: Optional[Callable[[float], bool]] = None,
        cpp_zero_init: str = "0",
        cpp_mul_op: str = "[](auto a, auto b) { return a * b; }",
        cpp_add_op: str = "[](auto acc, auto x) { return acc + x; }",
        cpp_sadd_op: str = "[](auto left, auto right) { return left + right; }",
        cpp_pmul_op: str = "[](auto left, auto right) { return left * right; }",
        cpp_ssub_op: str = "[](auto left, auto right) { return left - right; }",
        add_idempotent: bool = False,
    ) -> None:
        _is_zero = is_zero_fn if is_zero_fn is not None else (lambda v, z=zero: bool(abs(v - z) < 1e-9))  # type: ignore[misc]
        _is_one = is_one_fn if is_one_fn is not None else (lambda v, o=one: bool(abs(v - o) < 1e-9))  # type: ignore[misc]
        super().__init__(
            name=name,
            zero_val=zero,
            one_val=one,
            is_zero_fn=_is_zero,
            is_one_fn=_is_one,
            add_fn=plus,
            mult_fn=times,
            sub_fn=lambda x, y: x - y,
            div_fn=lambda x, y: x / y,
            add_idempotent=add_idempotent,
        )
        self.cpp_zero_init = cpp_zero_init
        self.cpp_mul_op = cpp_mul_op
        self.cpp_add_op = cpp_add_op
        self.cpp_sadd_op = cpp_sadd_op
        self.cpp_pmul_op = cpp_pmul_op
        self.cpp_ssub_op = cpp_ssub_op

    def cpp_preamble(self) -> str:
        suffix = self.name
        return (
            f"// Custom semiring: {suffix}\n"
            f"SConvolveOp({suffix}, {self.cpp_zero_init}, {self.cpp_mul_op}, {self.cpp_add_op});\n"
            f"SRecurseOp({suffix}, {self.cpp_zero_init}, {self.cpp_mul_op}, {self.cpp_add_op});\n"
            f"SRecurseVOp({suffix}, {self.cpp_zero_init}, {self.cpp_mul_op}, {self.cpp_add_op});\n"
            f"BinOp(SAdd{suffix}, {self.cpp_sadd_op});\n"
            f"BinOp(PointwiseMul{suffix}, {self.cpp_pmul_op});\n"
            f"BinOp(SSub{suffix}, {self.cpp_ssub_op});\n"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CustomSemiring) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"CustomSemiring({self.name!r})"
