from enum import Enum
from typing import Optional
from vecrec.expr import *


class ElementType(Enum):
    Float = 1
    I32 = 2

    def bit_width(self) -> int:
        match self:
            case ElementType.Float:
                return 32
            case ElementType.I32:
                return 32


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
        args = ", ".join(
            f"const {self.get_vec_type(ElementType.Float)}& {var}" for var in vars
        )
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

                program = f"SConvolve<{code_a.taps}, {vec_type}>({code_a.text}, {code_f.text})"
                return Code(program, element_type, code_a.lanes)
            case Recurse(a, g):
                time_delay = a.time_delay()
                code_a = self.generate_kernel(a)
                code_g = self.generate_signal(g)

                assert code_a.element_type == code_g.element_type
                element_type = code_a.element_type

                # Lanes of an IIR depends on its time delay (and hardware limit)
                lanes = min(self.max_lanes(code_a.element_type), time_delay)

                code_a = self.enforce_lanes(code_a, lanes)
                code_g = self.enforce_lanes(code_g, lanes)
                vec_type = self.get_vec_type(element_type, lanes)

                program = (
                    f"SRecurse<{code_a.taps}, {vec_type}>({code_a.text}, {code_g.text})"
                )
                return Code(program, element_type, lanes)

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
            )
        else:  # code.lanes > lanes
            # downscale lanes
            return Code(
                f"ConvertOne2N<{vec_type_in}, {vec_type_out}>({code.text})",
                code.element_type,
                lanes,
            )

    def generate_kernel(self, expr: KernelExpr) -> Code:
        match expr:
            case TIKernel(data):
                vec_type = self.get_vec_type(ElementType.Float)
                taps = len(data)
                arguments = ", ".join(map(str, data))
                Code(
                    f"TimeInvariantKernel<{taps}, {vec_type}>({{{arguments}}})",
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
            case KAdd(a, b):
                code_a = self.generate_kernel(a)
                code_b = self.generate_kernel(b)

                assert code_a.element_type == code_b.element_type
                element_type = code_a.element_type
                vec_type = self.get_vec_type(element_type)

                assert code_a.taps and code_b.taps and code_a.taps == code_b.taps
                taps = code_a.taps

                return Code(
                    f"KAdd<{taps}, {vec_type}>({code_a.text}, {code_b.text})",
                    element_type,
                    self.max_lanes(element_type),
                    taps=taps,
                )
            case KSub(a, b):
                code_a = self.generate_kernel(a)
                code_b = self.generate_kernel(b)

                assert code_a.element_type == code_b.element_type
                element_type = code_a.element_type
                vec_type = self.get_vec_type(element_type)

                assert code_a.taps and code_b.taps and code_a.taps == code_b.taps
                taps = code_a.taps

                return Code(
                    f"KSub<{taps}, {vec_type}>({code_a.text}, {code_b.text})",
                    element_type,
                    self.max_lanes(element_type),
                    taps=taps,
                )
            case KNeg(a):
                code_a = self.generate_kernel(a)

                element_type = code_a.element_type
                vec_type = self.get_vec_type(element_type)

                assert code_a.taps
                taps = code_a.taps

                return Code(
                    f"KNeg<{taps}, {vec_type}>({code_a.text})",
                    element_type,
                    self.max_lanes(element_type),
                    taps=taps,
                )
            case KConvolve(k1, k2):
                code_k1 = self.generate_kernel(k1)
                code_k2 = self.generate_kernel(k2)

                assert code_k1.element_type == code_k2.element_type
                element_type = code_k1.element_type
                vec_type = self.get_vec_type(element_type)

                assert code_k1.taps and code_k2.taps

                return Code(
                    f"KConvolve<{code_k1.taps}, {code_k2.taps}, {vec_type}>({code_k1.text}, {code_k2.text})",
                    element_type,
                    self.max_lanes(element_type),
                    taps=code_k1.taps + code_k2.taps - 1,
                )

        assert False

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
