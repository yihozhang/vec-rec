from __future__ import annotations
from abc import abstractmethod
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import numbers
import threading
from typing import Callable, ClassVar, Generator, List, Optional, Tuple, Union, overload

import numpy as np

from vecrec.util import ElementType


class Type:
    """Semiring type.

    Built-in singleton constants: ``Type.Arith``, ``Type.TropMax``, ``Type.TropMin``.
    Subclass :class:`~vecrec.semiring.CustomSemiring` for user-defined semirings.
    """

    # Class-level singleton constants (assigned after class body)
    Arith: ClassVar["Type"]
    TropMax: ClassVar["Type"]
    TropMin: ClassVar["Type"]

    _name: str
    _zero_val: float
    _one_val: float
    _is_zero_fn: Callable[[float], bool]
    _is_one_fn: Callable[[float], bool]
    _add_fn: Callable[[float, float], float]
    _mult_fn: Callable[[float, float], float]
    _sub_fn: Optional[Callable[[float, float], float]]
    _div_fn: Optional[Callable[[float, float], float]]
    _convolve_fn: Optional[Callable[[List[float], List[float]], List[float]]]

    def __init__(
        self,
        name: str,
        zero_val: float,
        one_val: float,
        is_zero_fn: Callable[[float], bool],
        is_one_fn: Callable[[float], bool],
        add_fn: Callable[[float, float], float],
        mult_fn: Callable[[float, float], float],
        sub_fn: Optional[Callable[[float, float], float]] = None,
        div_fn: Optional[Callable[[float, float], float]] = None,
        convolve_fn: Optional[Callable[[List[float], List[float]], List[float]]] = None,
    ) -> None:
        self._name = name
        self._zero_val = zero_val
        self._one_val = one_val
        self._is_zero_fn = is_zero_fn
        self._is_one_fn = is_one_fn
        self._add_fn = add_fn
        self._mult_fn = mult_fn
        self._sub_fn = sub_fn
        self._div_fn = div_fn
        self._convolve_fn = convolve_fn

    @property
    def name(self) -> str:
        return self._name

    def is_zero(self, value: float) -> bool:
        return self._is_zero_fn(value)

    def is_one(self, value: float) -> bool:
        return self._is_one_fn(value)

    def zero(self) -> float:
        return self._zero_val

    def one(self) -> float:
        return self._one_val

    def _apply(
        self,
        f: Callable[[float, float], float],
        a: Union[float, List[float]],
        b: Union[float, List[float]],
    ) -> Union[float, List[float]]:
        if isinstance(a, list) and isinstance(b, list):
            return [f(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) and isinstance(b, numbers.Number):
            return [f(x, b) for x in a]
        elif isinstance(a, numbers.Number) and isinstance(b, list):
            return [f(a, y) for y in b]
        elif isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return f(a, b)
        assert False, "unreachable"

    @overload
    def add(self, a: float, b: float) -> float: ...
    @overload
    def add(self, a: List[float], b: Union[List[float], float]) -> List[float]: ...
    @overload
    def add(self, a: float, b: List[float]) -> List[float]: ...

    def add(self, a: Union[float, List[float]], b: Union[float, List[float]]) -> Union[float, List[float]]:
        return self._apply(self._add_fn, a, b)

    @overload
    def sub(self, a: List[float], b: List[float]) -> List[float]: ...
    @overload
    def sub(self, a: List[float], b: float) -> List[float]: ...
    @overload
    def sub(self, a: float, b: List[float]) -> List[float]: ...

    def sub(self, a: Union[float, List[float]], b: Union[float, List[float]]) -> Union[float, List[float]]:
        assert self._sub_fn is not None, f"Subtraction is not defined for {self._name}"
        return self._apply(self._sub_fn, a, b)

    @overload
    def mult(self, a: float, b: float) -> float: ...
    @overload
    def mult(self, a: List[float], b: Union[List[float], float]) -> List[float]: ...
    @overload
    def mult(self, a: float, b: List[float]) -> List[float]: ...

    def mult(self, a: Union[float, List[float]], b: Union[float, List[float]]) -> Union[float, List[float]]:
        return self._apply(self._mult_fn, a, b)

    @overload
    def div(self, a: float, b: float) -> float: ...
    @overload
    def div(self, a: List[float], b: Union[List[float], float]) -> List[float]: ...
    @overload
    def div(self, a: float, b: List[float]) -> List[float]: ...

    def div(self, a: Union[float, List[float]], b: Union[float, List[float]]) -> Union[float, List[float]]:
        assert self._div_fn is not None, f"Division is not defined for {self._name}"
        return self._apply(self._div_fn, a, b)

    def has_additive_inverse(self) -> bool:
        """Return True if every element has an additive inverse (i.e. subtraction/negation is defined)."""
        return self._sub_fn is not None

    def convolve(self, a: List[float], b: List[float]) -> List[float]:
        if self._convolve_fn is not None:
            return self._convolve_fn(a, b)
        zero = self._zero_val
        max_len = len(a) + len(b) - 1
        result: List[float] = [zero] * max_len
        for i in range(len(a)):
            for j in range(len(b)):
                result[i + j] = self._add_fn(result[i + j], self._mult_fn(a[i], b[j]))
        return result

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"Type.{self._name}"


Type.Arith = Type(
    name="Arith",
    zero_val=0.0,
    one_val=1.0,
    is_zero_fn=lambda v: bool(np.isclose(v, 0.0)),
    is_one_fn=lambda v: bool(np.isclose(v, 1.0)),
    add_fn=lambda x, y: x + y,
    mult_fn=lambda x, y: x * y,
    sub_fn=lambda x, y: x - y,
    div_fn=lambda x, y: x / y,
    convolve_fn=lambda a, b: np.convolve(a, b).tolist(),
)
Type.TropMax = Type(
    name="TropMax",
    zero_val=-np.inf,
    one_val=0.0,
    is_zero_fn=lambda v: bool(np.isclose(v, -np.inf)),
    is_one_fn=lambda v: bool(np.isclose(v, 0.0)),
    add_fn=lambda x, y: max(x, y),
    mult_fn=lambda x, y: x + y,
)
Type.TropMin = Type(
    name="TropMin",
    zero_val=np.inf,
    one_val=0.0,
    is_zero_fn=lambda v: bool(np.isclose(v, np.inf)),
    is_one_fn=lambda v: bool(np.isclose(v, 0.0)),
    add_fn=lambda x, y: min(x, y),
    mult_fn=lambda x, y: x + y,
)


@dataclass(kw_only=True, match_args=False)
class RecLang:
    lanes: Optional[int] = None

    ty: Type
    element_type: ElementType

    def __init__(self, lanes: Optional[int] = None) -> None:
        self.lanes = lanes

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False

        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.__dict__.items()))))

    def children(
        self,
    ) -> List[KernelExpr | SignalExpr | KernelExpr2D | SignalExpr2D | KernelMatrixExpr | SignalVectorExpr | SignalExpr2DVec]:
        """Return the children of the expression."""
        return [
            v
            for v in self.__dict__.values()
            if isinstance(v, (KernelExpr, SignalExpr, KernelExpr2D, SignalExpr2D, KernelMatrixExpr, SignalVectorExpr, SignalExpr2DVec))
        ]

    def is_leaf(self) -> bool:
        """Return whether the expression is a leaf node."""
        for v in self.__dict__.values():
            if isinstance(v, (KernelExpr, SignalExpr, KernelExpr2D, SignalExpr2D, KernelMatrixExpr, SignalVectorExpr, SignalExpr2DVec)):
                return False
        return True

    def with_lanes(self, lanes: Optional[int]) -> RecLang:
        """Return a copy of the expression with the given number of lanes."""
        assert self.lanes is None, "Lanes already set"
        new_expr = copy.copy(self)
        new_expr.lanes = lanes
        return new_expr


@dataclass(match_args=False)
class SignalExpr(RecLang):
    ty: Type
    element_type: ElementType

    def with_lanes(self, lanes: Optional[int]) -> SignalExpr:
        """Return a copy of the expression with the given number of lanes."""
        return super().with_lanes(lanes)  # type: ignore

    def cache_rows(self) -> SignalExpr2D:
        from vecrec.expr.signal_ops import (
            Repeater,
        )  # local import to avoid circular dependency

        return Repeater(lambda _: self, self.ty, self.element_type).with_lanes(self.lanes)


@dataclass(match_args=False)
class SignalExpr2D(RecLang):
    ty: Type
    element_type: ElementType

    def at(self, r: int) -> SignalExpr:
        from vecrec.expr.signal_ops import Ith

        return Ith(self, r)

    def with_lanes(self, lanes: Optional[int]) -> SignalExpr2D:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class KernelExpr(RecLang):
    ty: Type
    element_type: ElementType

    @abstractmethod
    def time_delay(self, max_delay: int) -> Tuple[int, KernelExpr]:
        """Return the time delay of the kernel and the kernel without the delay."""
        ...

    @abstractmethod
    def to_sparse_repr(self) -> Tuple[int, List[int], List[float] | List[SignalExpr]]: ...

    def with_lanes(self, lanes: Optional[int]) -> KernelExpr:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class KernelExpr2D(RecLang):
    ty: Type
    element_type: ElementType

    def with_lanes(self, lanes: Optional[int]) -> KernelExpr2D:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class KernelMatrixExpr(RecLang):
    """Base class for an n×n matrix of kernel expressions."""

    ty: Type
    element_type: ElementType
    n: int

    def __init__(self, ty: Type, element_type: ElementType, n: int, lanes: Optional[int] = None) -> None:
        super().__init__(lanes)
        self.ty = ty
        self.element_type = element_type
        self.n = n

    def with_lanes(self, lanes: Optional[int]) -> KernelMatrixExpr:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class SignalVectorExpr(RecLang):
    """Base class for an n-vector of signal streams (each run() produces n vec_type values)."""

    ty: Type
    element_type: ElementType
    n: int

    def __init__(self, ty: Type, element_type: ElementType, n: int, lanes: Optional[int] = None) -> None:
        super().__init__(lanes)
        self.ty = ty
        self.element_type = element_type
        self.n = n

    def at(self, i: int) -> SignalExpr:
        """Extract component i as a 1D SignalExpr."""
        from vecrec.expr.signal_vec import VectorAt  # local import to avoid circular dependency

        return VectorAt(self, i)

    def with_lanes(self, lanes: Optional[int]) -> SignalVectorExpr:
        return super().with_lanes(lanes)  # type: ignore


@dataclass(match_args=False)
class SignalExpr2DVec(RecLang):
    """Base class for a vector of n 2D signal streams (each component is a SignalExpr2D)."""

    ty: Type
    element_type: ElementType
    n: int

    def __init__(self, ty: Type, element_type: ElementType, n: int, lanes: Optional[int] = None) -> None:
        super().__init__(lanes)
        self.ty = ty
        self.element_type = element_type
        self.n = n

    def at(self, i: int) -> SignalExpr2D:
        """Extract component i as a SignalExpr2D."""
        from vecrec.expr.signal_ops import Signal2DAt  # local import to avoid circular dependency

        return Signal2DAt(self, i)

    def with_lanes(self, lanes: Optional[int]) -> SignalExpr2DVec:
        return super().with_lanes(lanes)  # type: ignore


# ---------------------------------------------------------------------------
# Default-type context manager
# ---------------------------------------------------------------------------

_type_context: threading.local = threading.local()


def _resolve_types(ty: Optional[Type], element_type: Optional[ElementType]) -> Tuple[Type, ElementType]:
    """Resolve ty/element_type, falling back to the current with_types() context."""
    resolved_ty = ty if ty is not None else getattr(_type_context, "ty", None)
    resolved_et = element_type if element_type is not None else getattr(_type_context, "element_type", None)
    if resolved_ty is None:
        raise ValueError("Type must be specified explicitly or via the with_types() context manager")
    if resolved_et is None:
        raise ValueError("ElementType must be specified explicitly or via the with_types() context manager")
    return resolved_ty, resolved_et


@contextmanager
def with_types(ty: Type, element_type: ElementType) -> Generator[None, None, None]:
    """Context manager that sets default Type and ElementType for leaf AST nodes.

    Example::

        with with_types(Type.Arith, ElementType.Float):
            kernel = TIKernel([0, 1.8, -0.9])
            signal = Var("x")
            expr = Recurse(kernel, signal)
    """
    old_ty: Optional[Type] = getattr(_type_context, "ty", None)
    old_et: Optional[ElementType] = getattr(_type_context, "element_type", None)
    _type_context.ty = ty
    _type_context.element_type = element_type
    try:
        yield
    finally:
        _type_context.ty = old_ty
        _type_context.element_type = old_et
