from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .base import *
from .base import _resolve_types, Type
from vecrec.util import ElementType

@dataclass(unsafe_hash=True)
class Num(SignalExpr):
    value: float
    ty: Type
    element_type: ElementType
    __match_args__ = ("value",)

    def __init__(self, value: float, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.value = value

@dataclass(unsafe_hash=True)
class Impulse(SignalExpr):
    """Produces `value` at time 0 and the zero element of the semiring thereafter.
    Useful for setting up initial conditions of recurrences."""
    value: float
    ty: Type
    element_type: ElementType
    __match_args__ = ("value",)

    def __init__(self, value: float, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.value = value

@dataclass(unsafe_hash=True)
class Var(SignalExpr):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.name = name

@dataclass(unsafe_hash=True)
class RVar2D(SignalExpr2D):
    name: str
    __match_args__ = ("name",)

    def __init__(self, name: str, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.name = name

class RVec2D:
    """Container for n RVar2D objects used as the prev-state argument inside RepeaterVec bodies.
    Not a RecLang node — just a plain Python helper."""

    vars: tuple[RVar2D, ...]

    def __init__(self, vars: list[RVar2D]) -> None:
        self.vars = tuple(vars)

    def at(self, i: int) -> RVar2D:
        return self.vars[i]


@dataclass(unsafe_hash=True)
class Var2D(SignalExpr2D):
    """A user-provided 2D input array. Exposes n_rows consecutive rows simultaneously.
    n_rows is inferred by InferNRows from Ith usages; do not set manually."""
    name: str
    n_rows: Optional[int]
    __match_args__ = ("name", "n_rows")

    def __init__(self, name: str, ty: Optional[Type] = None, element_type: Optional[ElementType] = None) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.name = name
        self.n_rows = None


@dataclass(unsafe_hash=True)
class ComputedSignal2D(SignalExpr2D):
    """A 2D input signal whose values are *computed on the fly* by a user-provided
    C++ struct rather than read from a materialized buffer. It behaves exactly like
    Var2D through every transform/schedule/annotation pass (a leaf exposing n_rows
    consecutive rows, inferred by InferNRows from Ith usages), differing only in
    codegen lowering.

    The signal is entirely benchmark-owned and self-describing:
      cpp_struct: name of the C++ struct/template implementing the signal interface
                  (run(reg, base) / reset_and_next_row()), instantiated by codegen as
                  cpp_struct<vec_type, n_rows, block_size, block_row>.
      prelude:    full C++ definition of that struct, appended to the generated header.
      setup_code: C++ that allocates+fills the raw buffers it reads (emitted once in
                  the driver; may reference the in-scope `N` and `row_size`).
      make_args:  constructor arguments after the resolved type, e.g.
                  "s1.data(), s2.data(), row_size, 0, 1". The driver constructs the
                  object as `cpp_struct<...>(make_args)` and passes it into the kernel
                  constructor.
    """
    name: str
    cpp_struct: str
    prelude: str
    setup_code: str
    make_args: str
    n_rows: Optional[int]
    __match_args__ = ("name", "cpp_struct", "prelude", "setup_code", "make_args", "n_rows")

    def __init__(
        self,
        name: str,
        cpp_struct: str,
        prelude: str,
        setup_code: str,
        make_args: str,
        ty: Optional[Type] = None,
        element_type: Optional[ElementType] = None,
    ) -> None:
        resolved_ty, resolved_et = _resolve_types(ty, element_type)
        super().__init__(resolved_ty, resolved_et)
        self.name = name
        self.cpp_struct = cpp_struct
        self.prelude = prelude
        self.setup_code = setup_code
        self.make_args = make_args
        self.n_rows = None
