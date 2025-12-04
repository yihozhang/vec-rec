from enum import Enum
from typing import Optional
from vecrec.expr import *
import numpy as np

from importlib.resources import files

TEMPLATE = files("vecrec.templates") / "common.h"


class ElementType(Enum):
    Float = 1
    I32 = 2

    def bit_width(self) -> int:
        match self:
            case ElementType.Float:
                return 32
            case ElementType.I32:
                return 32

    def to_str(self) -> str:
        match self:
            case ElementType.Float:
                return "float"
            case ElementType.I32:
                return "int32_t"


class Code:
    text: str
    element_type: ElementType
    lanes: int
    taps: int | None

    def __init__(self, text, element_type, lanes, taps=None):
        self.text = text
        self.element_type = element_type
        self.lanes = lanes
        self.taps = taps

    def to_str(self, path: str) -> str:
        text = TEMPLATE.read_text() + "\n" + self.text
        return text

    def to_file(self, path: str):
        text = self.to_str(path)
        with open(path, "w") as f:
            f.write(text)


class CodeGen:
    # Hardware-dependent maximum
    max_bits: int

    def __init__(self, max_bits: int):
        assert max_bits in [64, 128, 256, 512]
        self.max_bits = max_bits

    def max_lanes(self, element_type: ElementType) -> int:
        """Get the maximal number of lanes possible given the type"""
        return self.max_bits // element_type.bit_width()

    def get_vec_type(self, element_type: ElementType, lanes: int = -1) -> str:
        """
        Get the C++ type of the given element type and lanes.
        Use maximum number of lanes if not provided.
        """
        if lanes == -1:
            lanes = self.max_bits // element_type.bit_width()

        match element_type:
            case ElementType.Float:
                return f"float_vec{lanes}"
            case ElementType.I32:
                raise NotImplementedError

    def generate(self, expr: SignalExpr, name: str) -> Code:
        code = self.generate_signal(expr)
        vars = self.collect_variables(expr)
        args = ", ".join(f"const {ElementType.Float.to_str()} *{var}" for var in vars)
        text = "\n".join(
            [
                f"auto make_{name}({args}) {{",
                f"    return {code.text};",
                "}",
            ]
        )
        return Code(
            text,
            code.element_type,
            code.lanes,
        )

    def generate_signal(self, expr: SignalExpr) -> Code:
        match expr:
            case Num(value):
                return Code(
                    f"Signal1DConstant<{self.get_vec_type(ElementType.Float)}>({value})",
                    ElementType.Float,
                    self.max_lanes(ElementType.Float),
                )
            case SignalExprBinOp(a, b):
                code_a = self.generate_signal(a)
                code_b = self.generate_signal(b)

                assert code_a.element_type == code_b.element_type
                element_type = code_a.element_type

                code_a = self.enforce_lanes(code_a)
                code_b = self.enforce_lanes(code_b)

                constructor = type(expr).__name__
                program = f"{constructor}({code_a.text}, {code_b.text})"
                return Code(program, element_type, self.max_lanes(element_type))

            case Var(name):
                # TODO: It is not optimal to assume input signal can always be produced
                # in groups of `max_lanes`, since it might be immediately followed by a
                # Recurse that requires a smaller lanes
                return Code(
                    f"Signal1D<{self.get_vec_type(ElementType.Float)}>({name})",
                    ElementType.Float,
                    self.max_lanes(ElementType.Float),
                )
            case Convolve(a, f):
                code_a = self.generate_kernel(a)
                code_f = self.generate_signal(f)

                assert code_a.element_type == code_f.element_type
                element_type = code_a.element_type

                code_a = self.enforce_lanes(code_a)
                code_f = self.enforce_lanes(code_f)
                vec_type = self.get_vec_type(element_type)

                program = f"make_s_convolve<{code_a.taps}, {vec_type}>({code_a.text}, {code_f.text})"
                return Code(program, element_type, code_a.lanes)
            case Recurse(a, g):
                time_delay, a = a.time_delay()
                code_a = self.generate_kernel(a)
                code_g = self.generate_signal(g)

                assert code_a.element_type == code_g.element_type
                element_type = code_a.element_type

                # Lanes of an IIR depends on its time delay (and hardware limit)
                lanes = min(self.max_lanes(code_a.element_type), time_delay)

                code_a = self.enforce_lanes(code_a, lanes)
                code_g = self.enforce_lanes(code_g, lanes)
                vec_type = self.get_vec_type(element_type, lanes)

                program = f"make_s_recurse<{code_a.taps}, {vec_type}>({code_a.text}, {code_g.text})"
                return Code(program, element_type, lanes)

        assert False

    def generate_kernel(self, expr: KernelExpr) -> Code:
        match expr:
            case TIKernel(_):
                vec_type = self.get_vec_type(ElementType.Float)

                taps, indices, values = expr.to_sparse_repr()
                index_args = ", ".join(map(str, indices))
                value_args = ", ".join(map(str, values))
                return Code(
                    f"TimeInvariantKernel<{taps}, {vec_type}>({{{index_args}}}, {{{value_args}}})",
                    ElementType.Float,
                    self.max_lanes(ElementType.Float),
                    taps=taps,
                )
            case TVKernel(signals):
                signal_codes = [self.generate_signal(signal) for signal in signals]
                signal_codes = [self.enforce_lanes(code) for code in signal_codes]
                element_type = signal_codes[0].element_type
                vec_type = self.get_vec_type(element_type)
                taps = len(signal_codes)

                arguments = ", ".join(c.text for c in signal_codes)

                return Code(
                    f"TimeVaryingKernel<{taps}, {vec_type}>({arguments})",
                    element_type,
                    self.max_lanes(element_type),
                    taps=taps,
                )
            case KAdd(_, _) | KSub(_, _) | KNeg(_) | KConvolve(_, _):
                raise NotImplementedError("These apps should have been eliminated by constant folding")
            case _:
                assert False


    def enforce_lanes(self, code: Code, lanes: int = -1) -> Code:
        """
        Return the code adapted to produce a vector of maximum lanes possible.
        This is a no-op if the code is already producing maximum lanes
        """
        if lanes == -1:
            lanes = self.max_lanes(code.element_type)

        if code.lanes == lanes:
            return code

        vec_type_in = self.get_vec_type(code.element_type, code.lanes)
        vec_type_out = self.get_vec_type(code.element_type, lanes)

        if code.lanes < lanes:
            # upscale lanes
            return Code(
                f"ConvertN2One<{vec_type_in}, {vec_type_out}>({code.text})",
                code.element_type,
                lanes,
                taps=code.taps,
            )
        else:  # code.lanes > lanes
            # downscale lanes
            return Code(
                f"ConvertOne2N<{vec_type_in}, {vec_type_out}>({code.text})",
                code.element_type,
                lanes,
                taps=code.taps,
            )

    def collect_variables(
        self, expr: RecLang, vars: Optional[set[str]] = None
    ) -> set[str]:
        if vars is None:
            vars = set()

        match expr:
            case Var(name):
                # TODO: variables need have type information
                vars.add(name)
            case SignalExprBinOp(a, b):
                self.collect_variables(a, vars)
                self.collect_variables(b, vars)
            case Convolve(a, f):
                self.collect_variables(a, vars)
                self.collect_variables(f, vars)
            case Recurse(a, g):
                self.collect_variables(a, vars)
                self.collect_variables(g, vars)
            case KAdd(a, b) | KSub(a, b) | KConvolve(a, b):
                self.collect_variables(a, vars)
                self.collect_variables(b, vars)
            case KNeg(a):
                self.collect_variables(a, vars)
            case TIKernel() | TVKernel() | Num():
                pass

        return vars
