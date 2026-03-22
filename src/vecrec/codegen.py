from typing import List, Optional, Sequence, Dict, Tuple, Union, cast
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D, KernelMatrixExpr, SignalVectorExpr
from vecrec.expr.signal import RVar2D, Var2D
from vecrec.expr.signal_ops import Repeater, Ith
from vecrec.expr.kernel import TIKernel, TVKernel
from vecrec.expr.kernel_matrix import KernelMatrix
from vecrec.expr.signal_vec import SignalVector, ConvolveV, RecurseV, VectorAt
from vecrec.util import ElementType
import json
import numpy as np
import os
import subprocess
import tempfile
from importlib.resources import files

__all__ = [
    "CodeGen",
    "generate_benchmark",
    "instantiate_kernels",
    "generate_kernel_executable",
    "generate_and_run_benchmark",
]

TEMPLATE = files("vecrec.templates") / "common.h"


class Code:
    text: str
    ty: Type
    element_type: ElementType
    lanes: int
    taps: int | None

    def __init__(self, text, ty, element_type, lanes, taps=None):
        self.text = text
        self.ty = ty
        self.element_type = element_type
        self.lanes = lanes
        self.taps = taps


class CodeGen:
    # Hardware-dependent maximum
    counter: int
    prologue: list[str]
    # Maps RVar2D names to their context variable names (str) and n_rows (int)
    var2d_context: Dict[str, Union[str, int]]
    # Maps expressions (by structural equality) to Code object for caching identical AST structures
    expr_cache: Optional[Dict]

    def __init__(self, cache_cse: bool = True):
        self.counter = 0
        self.prologue = []
        self.var2d_context = {}
        self.expr_cache = {} if cache_cse else None

    def get_new_var(self) -> str:
        self.counter += 1
        return f"v{self.counter}"

    def get_vec_type(self, element_type: ElementType, lanes: int) -> str:
        """
        Get the C++ type of the given element type and lanes.
        Use maximum number of lanes if not provided.
        """
        match element_type:
            case ElementType.Float:
                return f"float_vec{lanes}"
            case ElementType.I32:
                return f"int32_vec{lanes}"
            case ElementType.I64:
                return f"int64_vec{lanes}"

    def clear(self):
        self.counter = 0
        self.prologue = []
        self.var2d_context = {}
        self.expr_cache = {} if self.expr_cache is not None else None

    def get_sem_suffix(self, ty: Type) -> str:
        match ty:
            case Type.Arith:
                return ""
            case Type.TropMax:
                return "TropMax"
            case Type.TropMin:
                return "TropMin"

    def _is_trivial_expr(self, expr: SignalExpr | SignalExpr2D | KernelExpr) -> bool:
        """Check if an expression is trivial and should be inlined rather than cached."""
        return isinstance(expr, (Num, Impulse, Var, TIKernel, RVar2D, Var2D))

    def _generate_input_signal(self, expr: SignalExpr | SignalExpr2D) -> Code:
        if isinstance(expr, Var):
            assert expr.lanes is not None
            return Code(
                f"Signal1D<{self.get_vec_type(expr.element_type, expr.lanes)}>({expr.name})",
                expr.ty,
                expr.element_type,
                expr.lanes,
            )

        if isinstance(expr, Var2D):
            assert expr.lanes is not None
            assert expr.n_rows is not None, (
                f"Var2D('{expr.name}') has n_rows=None at codegen time. "
                "Did you forget to include InferNRows in your schedule before AnnotateLanes?"
            )
            vec_type = self.get_vec_type(expr.element_type, expr.lanes)
            return Code(
                f"make_input_signal2d<{vec_type}, {expr.n_rows}>({expr.name}, {expr.name}_stride)",
                expr.ty,
                expr.element_type,
                expr.lanes,
            )

        raise TypeError(f"Expected Var or Var2D, got {type(expr)}")

    def _signal2d_n_rows(self, signal2d: SignalExpr2D) -> int:
        if isinstance(signal2d, RVar2D):
            if signal2d.name not in self.var2d_context:
                raise ValueError(f"RVar2D '{signal2d.name}' used outside of Repeater context")
            return cast(int, self.var2d_context[f"{signal2d.name}_n_rows"])

        if isinstance(signal2d, (Repeater, Var2D)):
            assert signal2d.n_rows is not None, (
                f"{type(signal2d).__name__} has n_rows=None at codegen time. "
                "Did you forget to include InferNRows in your schedule before AnnotateLanes?"
            )
            return signal2d.n_rows

        raise ValueError(f"Ith expects RVar2D, Repeater, or Var2D as signal2d, got {type(signal2d)}")

    def collect_input_variables(
        self,
        expr: RecLang,
        vars_1d: Optional[Dict[str, ElementType]] = None,
        vars_2d: Optional[Dict[str, ElementType]] = None,
    ) -> Tuple[Dict[str, ElementType], Dict[str, ElementType]]:
        if vars_1d is None:
            vars_1d = {}
        if vars_2d is None:
            vars_2d = {}

        match expr:
            case Var(name):
                if name in vars_1d and vars_1d[name] != expr.element_type:
                    raise ValueError(
                        f"Variable {name} has inconsistent element types: {vars_1d[name]} vs {expr.element_type}"
                    )
                vars_1d[name] = expr.element_type
            case Var2D(name, _):
                if name in vars_2d and vars_2d[name] != expr.element_type:
                    raise ValueError(f"Var2D '{name}' has inconsistent element_type")
                vars_2d[name] = expr.element_type
            case _:
                for child in expr.children():
                    self.collect_input_variables(child, vars_1d, vars_2d)

        return vars_1d, vars_2d

    def generate(self, expr: SignalExpr | SignalExpr2D, name: str) -> Code:
        self.clear()
        code = self.generate_signal(expr)
        vars, vars2d = self.collect_input_variables(expr)
        args_1d = [f"const {elt_type.to_str()} *{var}" for var, elt_type in sorted(vars.items())]
        args_2d = [f"const {elt_type.to_str()} *{var}, int {var}_stride" for var, elt_type in sorted(vars2d.items())]
        all_args = args_1d + args_2d + ["int row_size"]
        args = ", ".join(all_args)
        text = "\n".join(
            [
                f"auto make_{name}({args}) {{",
                *map(lambda x: "    " + x, self.prologue),
                f"    return {code.text};",
                "}",
                f"using {name}_vector_type = {self.get_vec_type(code.element_type, code.lanes)};",
            ]
        )
        return Code(
            text,
            code.ty,
            code.element_type,
            code.lanes,
        )

    def generate_signal(self, expr: SignalExpr | SignalExpr2D) -> Code:
        # Check cache first (using structural equality)
        if self.expr_cache is not None and expr in self.expr_cache:
            return self.expr_cache[expr]

        assert expr.lanes is not None

        # Generate code for each case
        match expr:
            case Num(value):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                value_str = expr.element_type.val_to_str(value, expr.ty)
                code = Code(
                    f"Signal1DConstant<{vec_type}>({value_str})",
                    expr.ty,
                    expr.element_type,
                    expr.lanes,
                )
            case Impulse(value):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                value_str = expr.element_type.val_to_str(value, expr.ty)
                zero_str = expr.element_type.val_to_str(expr.ty.zero(), expr.ty)
                code = Code(
                    f"Signal1DImpulse<{vec_type}>({value_str}, {zero_str})",
                    expr.ty,
                    expr.element_type,
                    expr.lanes,
                )
            case SignalExprBinOp(a, b):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                code_a = self.generate_signal(a)
                code_b = self.generate_signal(b)

                assert code_a.element_type == code_b.element_type
                assert code_a.ty == code_b.ty
                element_type = code_a.element_type
                ty = code_a.ty

                constructor = "make_" + type(expr).__name__
                suffix = self.get_sem_suffix(ty)
                program = f"{constructor}{suffix}<{vec_type}>({code_a.text}, {code_b.text})"
                code = Code(program, ty, element_type, expr.lanes)

            case Var():
                assert expr.lanes is not None
                code = Code(
                    f"Signal1D<{self.get_vec_type(expr.element_type, expr.lanes)}>({expr.name})",
                    expr.ty,
                    expr.element_type,
                    expr.lanes,
                )
            case Var2D():
                assert expr.lanes is not None
                assert expr.n_rows is not None, (
                    f"Var2D('{expr.name}') has n_rows=None at codegen time. "
                    "Did you forget to include InferNRows in your schedule before AnnotateLanes?"
                )
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                code = Code(
                    f"make_input_signal2d<{vec_type}, {expr.n_rows}>({expr.name}, {expr.name}_stride)",
                    expr.ty,
                    expr.element_type,
                    expr.lanes,
                )
            case RVar2D(name):
                # RVar2D should only appear within a Repeater context
                if name not in self.var2d_context:
                    raise ValueError(f"RVar2D '{name}' used outside of Repeater context")

                context_var = self.var2d_context[name]
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                n_rows = cast(int, self.var2d_context[f"{name}_n_rows"])

                code = Code(
                    f"make_signal2d<{vec_type}, {n_rows}>({context_var})",
                    expr.ty,
                    expr.element_type,
                    expr.lanes,
                )
            case Repeater(a, n_rows, prev_rows_var):
                assert n_rows is not None, (
                    f"Repeater with prev_rows_var '{prev_rows_var.name}' has n_rows=None at codegen time. "
                    "Did you forget to include InferNRows in your schedule before AnnotateLanes?"
                )
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)

                # Create a context variable
                context_var = self.get_new_var()
                self.prologue.append(
                    f"RepeaterContext<{vec_type}, {n_rows}>* {context_var} = new RepeaterContext<{vec_type}, {n_rows}>(row_size);"
                )

                # Add the RVar2D to our context so inner signal can reference it
                old_context = self.var2d_context.copy()
                self.var2d_context[prev_rows_var.name] = f"{context_var}"
                # n_rows includes the current row, so prev_rows has n_rows - 1 rows
                self.var2d_context[f"{prev_rows_var.name}_n_rows"] = n_rows - 1

                # Generate code for the inner signal (which may reference prev_rows_var)
                code_a = self.generate_signal(a)

                # Restore context
                self.var2d_context = old_context

                # Generate the Repeater construction
                program = f"make_repeater<{vec_type}, {n_rows}>({context_var}, {code_a.text})"
                code = Code(program, expr.ty, expr.element_type, expr.lanes)
            case Convolve(a, f):
                code_a = self.generate_kernel(a)
                code_f = self.generate_signal(f)

                assert code_a.element_type == code_f.element_type
                assert code_a.ty == code_f.ty
                element_type = code_a.element_type
                ty = code_a.ty

                vec_type = self.get_vec_type(element_type, expr.lanes)
                suffix = self.get_sem_suffix(ty)
                program = f"make_s_convolve{suffix}<{vec_type}>({code_a.text}, {code_f.text})"
                code = Code(program, ty, element_type, expr.lanes)
            case Recurse(a, g):
                time_delay, a = a.time_delay(expr.lanes)
                assert time_delay == a.lanes
                code_a = self.generate_kernel(a)
                code_g = self.generate_signal(g)

                assert code_a.element_type == code_g.element_type
                assert code_a.ty == code_g.ty
                element_type = code_a.element_type
                ty = code_a.ty

                vec_type = self.get_vec_type(element_type, expr.lanes)

                suffix = self.get_sem_suffix(ty)
                program = f"make_s_recurse{suffix}<{vec_type}>({code_a.text}, {code_g.text})"
                code = Code(program, ty, element_type, a.lanes)
            case Ith(signal2d, i):
                # Ith extracts the ith row from a 2D signal
                # Generate: make_ith_row<vec_type, n_rows, i>(signal2d_code)
                element_type = expr.element_type
                ty = expr.ty

                # Generate code for the 2D signal (typically RVar2D -> Signal2D)
                code_signal2d = self.generate_signal(signal2d)

                # IthRow's vec_type must match the Signal2D's vec_type (code_signal2d.lanes),
                # not necessarily Ith's target lanes. After PushDownConvertLanes, Ith.lanes
                # may be narrowed while signal2d.lanes stays at max_lanes, so we generate
                # IthRow using signal2d_lanes and wrap with a lane converter if needed.
                signal2d_lanes = code_signal2d.lanes
                ith_vec_type = self.get_vec_type(element_type, signal2d_lanes)

                n_rows = self._signal2d_n_rows(signal2d)

                # Generate make_ith_row call using signal2d_lanes to keep C++ types consistent
                program = f"make_ith_row<{ith_vec_type}, {n_rows}, {i}>({code_signal2d.text})"
                code = Code(program, ty, element_type, signal2d_lanes)

                # Wrap with ConvertOne2N (downscale) or ConvertN2One (upscale) if needed
                if signal2d_lanes != expr.lanes:
                    code = self.enforce_lanes(code, expr.lanes)
            case ConvertLanes(a):
                code_a = self.generate_signal(a)
                code = self.enforce_lanes(code_a, expr.lanes)
            case VectorAt(a, i):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                if isinstance(a, ConvolveV):
                    # FIR: out[i] = sum_j K[i][j] * g[j]
                    # Decompose into SConvolve + SAdd — no shared state between rows.
                    assert isinstance(a.a, KernelMatrix) and isinstance(a.b, SignalVector)
                    terms = []
                    for j in range(a.n):
                        k = a.a.data[i][j]
                        assert isinstance(k, TIKernel)
                        if len(k.data) == 0:
                            continue  # zero kernel: no contribution
                        code_k = self.generate_kernel(k)
                        code_s = self.generate_signal(a.b.data[j])
                        terms.append(f"make_s_convolve<{vec_type}>({code_k.text}, {code_s.text})")
                    if not terms:
                        program = f"Signal1DConstant<{vec_type}>(0)"
                    elif len(terms) == 1:
                        program = terms[0]
                    else:
                        program = f"make_s_add<{vec_type}>({terms[0]}, {terms[1]})"
                else:
                    # IIR (RecurseV): generate the shared vector object, then extract component i
                    code_vec = self.generate_signal_vector(a)
                    n = a.n
                    program = f"make_vector_at<{n}, {i}, {vec_type}>({code_vec.text})"
                code = Code(program, expr.ty, expr.element_type, expr.lanes)
            case _:
                assert False

        # Cache and return at the end
        if not self._is_trivial_expr(expr):
            var = self.get_new_var()
            self.prologue.append(f"auto {var} = {code.text};")
            code = Code(var, code.ty, code.element_type, code.lanes, code.taps)
        if self.expr_cache is not None:
            self.expr_cache[expr] = code
        return code

    def generate_kernel(self, expr: KernelExpr) -> Code:
        # Check cache first (using structural equality)
        if self.expr_cache is not None and expr in self.expr_cache:
            return self.expr_cache[expr]

        assert expr.lanes is not None

        # Generate code for each case
        match expr:
            case TIKernel(_):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)

                taps, indices, values = expr.to_sparse_repr()

                if taps == 0:
                    code = Code(
                        f"ZeroKernel<{vec_type}>()",
                        expr.ty,
                        expr.element_type,
                        expr.lanes,
                        taps=0,
                    )
                else:
                    index_args = ", ".join(map(str, indices))
                    value_args = ", ".join(map(lambda v: f"{expr.element_type.val_to_str(v, expr.ty)}", values))

                    index_var = self.get_new_var()
                    value_var = self.get_new_var()

                    self.prologue.append(
                        f"constexpr static int {index_var}[{taps}] = {{{index_args}}};",
                    )
                    self.prologue.append(
                        f"constexpr static {expr.element_type.to_str()} {value_var}[{taps}] = {{{value_args}}};",
                    )

                    code = Code(
                        f"TimeInvariantKernel<{taps}, {vec_type}, {index_var}, {value_var}>()",
                        expr.ty,
                        expr.element_type,
                        expr.lanes,
                        taps=taps,
                    )
            case TVKernel(_):
                taps, indices, signals = expr.to_sparse_repr()
                signal_codes = [self.generate_signal(signal) for signal in signals]
                element_type = signal_codes[0].element_type
                ty = signal_codes[0].ty
                vec_type = self.get_vec_type(element_type, expr.lanes)
                taps = len(signal_codes)

                arguments = ", ".join(c.text for c in signal_codes)

                index_args = ", ".join(map(str, indices))
                index_var = self.get_new_var()
                self.prologue.append(
                    f"constexpr static int {index_var}[{taps}] = {{{index_args}}};",
                )

                code = Code(
                    f"make_time_varying_kernel<{taps}, {vec_type}, {index_var}>({arguments})",
                    ty,
                    element_type,
                    expr.lanes,
                    taps=taps,
                )
            case KConvertLanes(a):
                code_a = self.generate_kernel(a)
                code = self.enforce_lanes(code_a, expr.lanes)
            case KAdd(_, _) | KSub(_, _) | KNeg(_) | KConvolve(_, _):
                raise NotImplementedError("These apps should have been eliminated by constant folding")
            case _:
                assert False

        # Cache and return at the end
        if not self._is_trivial_expr(expr):
            var = self.get_new_var()
            self.prologue.append(f"auto {var} = {code.text};")
            code = Code(var, code.ty, code.element_type, code.lanes, code.taps)
        if self.expr_cache is not None:
            self.expr_cache[expr] = code
        return code

    def generate_signal_vector(self, expr: SignalVectorExpr) -> Code:
        """Generate code for a SignalVectorExpr.

        Returns a Code whose C++ object has a run(vec_type* out) that fills
        out[0..n-1].  The Code is cached in the prologue.
        """
        if self.expr_cache is not None and expr in self.expr_cache:
            return self.expr_cache[expr]

        assert expr.lanes is not None

        match expr:
            case SignalVector(data):
                # Individual signals: code is just comma-separated, not a single object.
                # We don't have a C++ struct for this directly; it's consumed by RecurseV/ConvolveV.
                raise NotImplementedError(
                    "SignalVector has no standalone C++ object; it must appear as the input to RecurseV or ConvolveV"
                )

            case RecurseV(a, g):
                assert isinstance(a, KernelMatrix), f"RecurseV kernel must be KernelMatrix, got {type(a)}"
                assert isinstance(g, SignalVector), f"RecurseV input must be SignalVector, got {type(g)}"
                n = a.n
                assert n == 2, f"RecurseV currently only supports n=2, got n={n}"
                lanes = expr.lanes
                vec_type = self.get_vec_type(expr.element_type, lanes)

                # Strip time delay from each kernel and assert delay == lanes
                stripped: List[List[TIKernel]] = []
                for i in range(n):
                    row: List[TIKernel] = []
                    for j in range(n):
                        k = a.data[i][j]
                        assert isinstance(k, TIKernel), f"RecurseV kernels must be TIKernel, got {type(k)}"
                        if len(k.data) == 0:
                            # Zero kernel – no feedback on this path
                            row.append(TIKernel([], k.ty, k.element_type).with_lanes(lanes))  # type: ignore[arg-type]
                        else:
                            delay, stripped_k = k.time_delay(lanes)
                            assert delay == lanes, (
                                f"RecurseV kernel A[{i}][{j}] has time delay {delay}, expected {lanes}. "
                                "Each kernel must have a causal delay equal to the number of lanes."
                            )
                            row.append(cast(TIKernel, stripped_k))
                    stripped.append(row)

                # Generate C++ code for each kernel
                kernel_codes: List[List[Code]] = [
                    [self.generate_kernel(stripped[i][j]) for j in range(n)]
                    for i in range(n)
                ]

                # Generate C++ code for each input signal
                signal_codes = [self.generate_signal(g.data[i]) for i in range(n)]

                # Build factory call for n=2
                k00, k01, k10, k11 = (
                    kernel_codes[0][0].text, kernel_codes[0][1].text,
                    kernel_codes[1][0].text, kernel_codes[1][1].text,
                )
                s0, s1 = signal_codes[0].text, signal_codes[1].text

                program = f"make_s_recurse_v2<{vec_type}>({k00}, {k01}, {k10}, {k11}, {s0}, {s1})"
                code = Code(program, expr.ty, expr.element_type, lanes)

            case _:
                assert False, f"Unsupported SignalVectorExpr: {type(expr)}"

        # Cache the vector signal in the prologue
        var = self.get_new_var()
        self.prologue.append(f"auto {var} = {code.text};")
        code = Code(var, code.ty, code.element_type, code.lanes)
        if self.expr_cache is not None:
            self.expr_cache[expr] = code
        return code

    def enforce_lanes(self, code: Code, lanes: int) -> Code:
        """
        Return the code adapted to produce a vector of maximum lanes possible.
        This is a no-op if the code is already producing maximum lanes
        """
        if code.lanes == lanes:
            return code

        vec_type_in = self.get_vec_type(code.element_type, code.lanes)
        vec_type_out = self.get_vec_type(code.element_type, lanes)

        function_name = {
            (True, True): "make_convert_n2one",
            (True, False): "make_convert_one2n",
            (False, True): "make_k_convert_n2one",
            (False, False): "make_k_convert_one2n",
        }[(code.taps is None, code.lanes < lanes)]

        if code.lanes < lanes:
            # upscale lanes
            return Code(
                f"{function_name}<{vec_type_in}, {vec_type_out}>({code.text})",
                code.ty,
                code.element_type,
                lanes,
                taps=code.taps,
            )
        else:  # code.lanes > lanes
            # downscale lanes
            return Code(
                f"{function_name}<{vec_type_in}, {vec_type_out}>({code.text})",
                code.ty,
                code.element_type,
                lanes,
                taps=code.taps,
            )


def instantiate_kernels(path: str, codes: List[Code]) -> None:
    text = TEMPLATE.read_text()
    for code in codes:
        text += "\n" + code.text
    with open(path, "w") as f:
        f.write(text)


def generate_benchmark(
    codegen: CodeGen,
    exprs: Sequence[SignalExpr | SignalExpr2D],
    kernel_names: Sequence[str],
    kernel_header_path: str,
    output_path: str,
    include_correctness_check: bool = False,
    correctness_tolerance: float = 1e-3,
    input_size: int = (1 << 20),
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    row_size: int = 1024,
) -> None:
    """
    Generate a C++ benchmarking program for the given kernels.

    Args:
        codegen: CodeGen instance used to generate the kernels
        exprs: List of signal expressions corresponding to each kernel
        kernel_names: List of names corresponding to each kernel
        output_path: Path where the benchmark C++ file will be written
        include_correctness_check: If True, adds correctness checking code
        correctness_tolerance: Tolerance for floating point comparisons in correctness checking
        input_size: Size of input data for benchmarking
        warmup_iterations: Number of warmup iterations before timing
        benchmark_iterations: Number of iterations for timing
        row_size: Number of elements per row (used to call reset_and_next_row at row boundaries)
    """
    assert len(exprs) == len(kernel_names), "Number of expressions and kernel names must match"
    assert len(exprs) > 0, "At least one expression must be provided"

    # Collect variables from all expressions
    all_vars: Dict[str, ElementType] = {}
    all_vars2d: Dict[str, ElementType] = {}
    for expr in exprs:
        vars_dict, vars2d_dict = codegen.collect_input_variables(expr)
        all_vars.update(vars_dict)
        all_vars2d.update(vars2d_dict)

    # Generate the benchmark program
    benchmark_code = _generate_benchmark_code(
        all_vars,
        all_vars2d,
        exprs[0].element_type,
        kernel_names,
        kernel_header_path,
        include_correctness_check,
        correctness_tolerance,
        input_size,
        warmup_iterations,
        benchmark_iterations,
        row_size,
    )

    with open(output_path, "w") as f:
        f.write(benchmark_code)


def _generate_benchmark_code(
    all_vars: Dict[str, ElementType],
    all_vars2d: Dict[str, ElementType],
    output_type: ElementType,
    kernel_names: Sequence[str],
    header_path: str,
    include_correctness_check: bool,
    correctness_tolerance: float,
    input_size: int,
    warmup_iterations: int,
    benchmark_iterations: int,
    row_size: int = 1024,
) -> str:
    """Generate the complete C++ benchmark code."""

    # Start with includes
    benchmark_text = f'#include "{header_path}"\n'
    benchmark_text += "#include <iostream>\n"
    benchmark_text += "#include <chrono>\n"
    benchmark_text += "#include <vector>\n"
    benchmark_text += "#include <cmath>\n"
    benchmark_text += "#include <random>\n\n"

    if include_correctness_check:
        benchmark_text += _generate_correctness_check_helper(correctness_tolerance)

    # Generate main function
    benchmark_text += "int main() {\n"
    benchmark_text += f"    const int N = {input_size};\n"
    benchmark_text += f"    const int row_size = {row_size};\n"
    benchmark_text += f"    const int warmup = {warmup_iterations};\n"
    benchmark_text += f"    const int iterations = {benchmark_iterations};\n\n"

    bytes_per_element_terms = [
        f"sizeof({elt_type.to_str()})" for _, elt_type in sorted(all_vars.items()) + sorted(all_vars2d.items())
    ]
    bytes_per_element_terms.append(f"sizeof({output_type.to_str()})")
    bytes_per_iteration_expr = " + ".join(bytes_per_element_terms)
    benchmark_text += (
        "    const double bytes_per_iteration = "
        f"static_cast<double>(N) * static_cast<double>({bytes_per_iteration_expr});\n"
    )
    benchmark_text += "\n"

    # Generate input data initialization
    benchmark_text += "    // Initialize input data\n"
    benchmark_text += "    std::random_device rd;\n"
    benchmark_text += "    std::mt19937 gen(rd());\n"

    # Determine which distributions are needed
    all_elt_types = list(all_vars.values()) + list(all_vars2d.values())
    needs_float = any(elt_type == ElementType.Float for elt_type in all_elt_types)
    needs_int32 = any(elt_type == ElementType.I32 for elt_type in all_elt_types)
    needs_int64 = any(elt_type == ElementType.I64 for elt_type in all_elt_types)

    if needs_float:
        benchmark_text += "    std::uniform_real_distribution<float> float_dis(-1.0f, 1.0f);\n"
    if needs_int32:
        benchmark_text += "    std::uniform_int_distribution<int32_t> int32_dis(-1000, 1000);\n"
    if needs_int64:
        benchmark_text += "    std::uniform_int_distribution<int64_t> int64_dis(-1000000, 1000000);\n"
    benchmark_text += "\n"

    for var, elt_type in sorted(all_vars.items()) + sorted(all_vars2d.items()):
        cpp_type = elt_type.to_str()
        if elt_type == ElementType.Float:
            dist = "float_dis"
        elif elt_type == ElementType.I32:
            dist = "int32_dis"
        else:  # I64 or U64
            dist = "int64_dis"

        benchmark_text += f"    std::vector<{cpp_type}> {var}(N);\n"
        benchmark_text += "    for (int i = 0; i < N; i++) {\n"
        benchmark_text += f"        {var}[i] = {dist}(gen);\n"
        benchmark_text += "    }\n\n"

    # Allocate 2D input arrays. The benchmark row width matches RepeaterContext::width (1024).
    # With causal (look-behind) semantics, N output elements need exactly N input elements.
    _benchmark_row_width = 1024

    # Generate output buffers for each kernel
    benchmark_text += "    // Output buffers for each kernel\n"
    for i, name in enumerate(kernel_names):
        benchmark_text += f"    std::vector<{output_type.to_str()}> output_{name}(N);\n"
    benchmark_text += "\n"

    # Generate benchmark code for each kernel
    var_args_1d = [f"{var}.data()" for var in sorted(all_vars)]
    var_args_2d = [f"{var}.data(), row_size" for var in sorted(all_vars2d)]
    var_args_list = var_args_1d + var_args_2d + ["row_size"]
    var_args = ", ".join(var_args_list)

    benchmark_text += '    std::cout << "{\\n";'

    for i, name in enumerate(kernel_names):
        benchmark_text += f"    // Benchmark {name}\n"
        benchmark_text += "    {\n"

        benchmark_text += f"        auto kernel = make_{name}({var_args});\n"
        benchmark_text += "        \n"
        benchmark_text += "        // Warmup\n"
        benchmark_text += "        for (int i = 0; i < warmup; i++) {\n"
        benchmark_text += f"            {name}_vector_type out;\n"
        benchmark_text += "            kernel.run(&out);\n"
        benchmark_text += "        }\n\n"

        benchmark_text += "        // Reset kernel state\n"
        benchmark_text += f"        kernel = make_{name}({var_args});\n\n"

        benchmark_text += "        // Timed run\n"
        benchmark_text += "        auto start = std::chrono::high_resolution_clock::now();\n"
        benchmark_text += "        for (int iter = 0; iter < iterations; iter++) {\n"
        benchmark_text += "            // Recreate kernel for each iteration\n"
        benchmark_text += f"            kernel = make_{name}({var_args});\n"
        benchmark_text += "            int pos = 0;\n"
        benchmark_text += "            while (pos < N) {\n"
        benchmark_text += f"                {name}_vector_type out;\n"
        # TODO: 16 only for avx-512 with float32.
        # This piece of code should be generalized.
        benchmark_text += "#pragma unroll\n"
        benchmark_text += f"                for (int i = 0; i < 16 / vec_lanes_of({name}_vector_type{{}}); i++) {{\n"
        benchmark_text += "                    kernel.run(&out);\n"
        benchmark_text += f"                    memcpy(&output_{name}[pos], &out, sizeof(out));\n"
        benchmark_text += "                    pos += vec_lanes_of(out);\n"

        benchmark_text += "                    if (pos % row_size == 0) {\n"
        benchmark_text += "                        kernel.reset_and_next_row();\n"
        benchmark_text += "                    }\n"

        benchmark_text += "                }\n"
        benchmark_text += "            }\n"
        benchmark_text += "        }\n"
        benchmark_text += "        auto end = std::chrono::high_resolution_clock::now();\n"
        benchmark_text += "        \n"
        benchmark_text += "        // XOR sum to prevent optimization of output array\n"
        benchmark_text += f"        uint64_t xor_sum = 0;\n"
        benchmark_text += f"        for (int i = 0; i < N; i++) {{\n"
        benchmark_text += f"            xor_sum ^= static_cast<uint64_t>(output_{name}[i]);\n"
        benchmark_text += f"        }}\n"
        benchmark_text += f"        volatile uint64_t xor_result_{name} = xor_sum;  // Prevent optimization\n"
        benchmark_text += "        \n"
        benchmark_text += "        const double elapsed_seconds = std::chrono::duration<double>(end - start).count();\n"
        benchmark_text += "        const double throughput_gbps = (bytes_per_iteration * static_cast<double>(iterations)) / elapsed_seconds / 1e6;\n"
        # Deciding if we need to generate the trailing comma
        if i == len(kernel_names) - 1 and not (include_correctness_check and len(kernel_names) > 1):
            benchmark_text += f'        std::cout << "  \\"{name}\\": " << throughput_gbps << "\\n";\n'
        else:
            benchmark_text += f'        std::cout << "  \\"{name}\\": " << throughput_gbps << ",\\n";\n'
        benchmark_text += "    }\n\n"

    # Add correctness checking if requested
    if include_correctness_check and len(kernel_names) > 1:
        benchmark_text += _generate_correctness_check_code(kernel_names, input_size)

    benchmark_text += '    std::cout << "}\\n";'

    benchmark_text += "    return 0;\n"
    benchmark_text += "}\n"

    return benchmark_text


def _generate_correctness_check_helper(tolerance: float = 1e-3) -> str:
    """
    Generate helper function for correctness checking.

    Returns:
        C++ code for arrays_equal function with trailing newlines for proper spacing.
    """
    return f"""// Helper function for comparing floating point arrays
int arrays_equal(const std::vector<float>& a, const std::vector<float>& b, float tolerance = {tolerance}) {{
    if (a.size() != b.size()) return -1;
    for (size_t i = 0; i < a.size(); i++) {{
        if (std::abs(a[i] - b[i]) > tolerance) {{
            return i;
        }}
    }}
    return a.size();
}}

int arrays_equal(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {{
    if (a.size() != b.size()) return -1;
    for (size_t i = 0; i < a.size(); i++) {{
        if (a[i] != b[i]) {{
            return i;
        }}
    }}
    return a.size();
}}

int arrays_equal(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {{
    if (a.size() != b.size()) return -1;
    for (size_t i = 0; i < a.size(); i++) {{
        if (a[i] != b[i]) {{
            return i;
        }}
    }}
    return a.size();
}}

"""


def _generate_correctness_check_code(kernel_names: Sequence[str], input_size: int) -> str:
    """
    Generate correctness checking code comparing all kernels.

    Compares each kernel's output against the first kernel (reference).

    Returns:
        C++ code for correctness checking with trailing newlines for proper spacing.
    """
    code = "    // Correctness checking\n"
    code += '    std::cout << "  \\"validation\\":{\\n";\n'
    code += "    int idx;"

    reference = kernel_names[0]
    for i in range(1, len(kernel_names)):
        name = kernel_names[i]
        code += f"    if ((idx=arrays_equal(output_{reference}, output_{name})) == output_{reference}.size()) {{\n"
        code += f'        std::cout << "    \\"{name}\\": true";\n'
        code += "    } else {\n"
        code += f'        std::cout << "    \\"{name}\\":" << idx;\n'
        code += "    }\n"
        if i == len(kernel_names) - 1:
            code += '    std::cout << "\\n";\n'
        else:
            code += '    std::cout << ",\\n";\n'

    code += '    std::cout << "  }\\n";\n'
    code += "\n"
    return code


def _numpy_dtype_for_element_type(element_type: ElementType) -> np.dtype:
    match element_type:
        case ElementType.Float:
            return np.dtype(np.float32)
        case ElementType.I32:
            return np.dtype(np.int32)
        case ElementType.I64:
            return np.dtype(np.int64)


class KernelExecutable:
    executable_path: str
    variable_order: List[str]
    variable_types: Dict[str, ElementType]
    vars2d_info: Dict[str, ElementType]
    vars2d_order: List[str]
    output_type: ElementType
    row_size: int

    def __init__(
        self,
        executable_path: str,
        variable_order: List[str],
        variable_types: Dict[str, ElementType],
        output_type: ElementType,
        vars2d_info: Dict[str, ElementType],
        row_size: int = 1024,
    ):
        self.executable_path = executable_path
        self.variable_order = variable_order
        self.variable_types = variable_types
        self.vars2d_info = vars2d_info
        self.vars2d_order = sorted(self.vars2d_info.keys())
        self.output_type = output_type
        self.row_size = row_size

    def _normalize_inputs(
        self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Return (inputs_1d, inputs_2d) dicts."""
        all_1d = set(self.variable_order)
        all_2d = set(self.vars2d_order)

        if isinstance(inputs, np.ndarray):
            total = len(all_1d) + len(all_2d)
            if total != 1:
                raise ValueError("Expected a dictionary of named arrays for multi-input kernels")
            if all_1d:
                return {self.variable_order[0]: inputs}, {}
            else:
                return {}, {self.vars2d_order[0]: inputs}

        if not isinstance(inputs, dict):
            raise TypeError("inputs must be a numpy array or a dict[str, numpy.ndarray]")

        expected = all_1d | all_2d
        actual = set(inputs.keys())
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Input variable mismatch: missing={missing}, extra={extra}")

        inputs_1d = {k: v for k, v in inputs.items() if k in all_1d}
        inputs_2d = {k: v for k, v in inputs.items() if k in all_2d}
        return inputs_1d, inputs_2d

    def run(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> tuple[np.ndarray, float]:
        inputs_1d, inputs_2d = self._normalize_inputs(inputs)

        n: Optional[int] = None

        # Prepare 1D inputs
        prepared_1d: Dict[str, np.ndarray] = {}
        for var in self.variable_order:
            expected_dtype = _numpy_dtype_for_element_type(self.variable_types[var])
            arr = np.asarray(inputs_1d[var], dtype=expected_dtype)
            if arr.ndim != 1:
                raise ValueError(f"Input '{var}' must be 1D, got shape {arr.shape}")
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            if n is None:
                n = int(arr.shape[0])
            elif int(arr.shape[0]) != n:
                raise ValueError(f"All input arrays must have the same length, got {n} and {arr.shape[0]}")
            prepared_1d[var] = arr

        # Prepare 2D inputs.
        prepared_2d: Dict[str, Tuple[np.ndarray, int]] = {}
        for var in self.vars2d_order:
            elt_type = self.vars2d_info[var]
            expected_dtype = _numpy_dtype_for_element_type(elt_type)
            arr = np.asarray(inputs_2d[var], dtype=expected_dtype)
            if arr.ndim != 2:
                raise ValueError(f"Input '{var}' must be 2D, got shape {arr.shape}")
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            H, W = arr.shape
            stride = W
            if n is None:
                # Infer N: with causal (look-behind) semantics all H rows produce output,
                # so N = H * stride.
                n = H * stride
            prepared_2d[var] = (arr, stride)

        assert n is not None

        with tempfile.TemporaryDirectory(prefix="vecrec_run_", dir="/tmp") as run_dir:
            # Write 1D input files
            input_paths_1d: List[str] = []
            for var in self.variable_order:
                input_path = os.path.join(run_dir, f"{var}.bin")
                prepared_1d[var].tofile(input_path)
                input_paths_1d.append(input_path)

            # Write 2D input files and collect path+stride CLI args
            input_args_2d: List[str] = []
            for var in self.vars2d_order:
                arr, stride = prepared_2d[var]
                input_path = os.path.join(run_dir, f"{var}.bin")
                arr.tofile(input_path)  # write entire 2D array as flat binary
                input_args_2d.append(input_path)
                input_args_2d.append(str(stride))

            output_path = os.path.join(run_dir, "output.bin")

            cmd = [
                self.executable_path,
                str(n),
                str(self.row_size),
                *input_paths_1d,
                *input_args_2d,
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Kernel execution failed with code {result.returncode}: {result.stderr}")

            try:
                metadata = json.loads(result.stdout.strip())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse runner output as JSON: {result.stdout}") from exc

            elapsed_us = float(metadata["elapsed_us"])

            output_dtype = _numpy_dtype_for_element_type(self.output_type)
            output = np.fromfile(output_path, dtype=output_dtype, count=n)
            if int(output.shape[0]) != n:
                raise RuntimeError(f"Output size mismatch: expected {n}, got {output.shape[0]}")

            return output, elapsed_us

    def __call__(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> tuple[np.ndarray, float]:
        return self.run(inputs)


def _generate_kernel_runner_code(
    all_vars: Dict[str, ElementType],
    all_vars2d: Dict[str, ElementType],
    output_type: ElementType,
    kernel_name: str,
    header_path: str,
) -> str:
    var_items = sorted(all_vars.items())
    var2d_items = sorted(all_vars2d.items())

    text = f'#include "{header_path}"\n'
    text += "#include <chrono>\n"
    text += "#include <cstdint>\n"
    text += "#include <cstdlib>\n"
    text += "#include <cstring>\n"
    text += "#include <fstream>\n"
    text += "#include <iostream>\n"
    text += "#include <vector>\n\n"

    text += "template <typename T>\n"
    text += "bool read_binary(const char* path, T* out, int n) {\n"
    text += "    std::ifstream in(path, std::ios::binary);\n"
    text += "    if (!in) return false;\n"
    text += "    in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(sizeof(T)) * n);\n"
    text += "    return in.good() || in.eof();\n"
    text += "}\n\n"

    text += "template <typename T>\n"
    text += "bool write_binary(const char* path, const T* data, int n) {\n"
    text += "    std::ofstream out(path, std::ios::binary);\n"
    text += "    if (!out) return false;\n"
    text += "    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(sizeof(T)) * n);\n"
    text += "    return static_cast<bool>(out);\n"
    text += "}\n\n"

    text += "int main(int argc, char** argv) {\n"
    # argc: exe + N + row_size + n_1d + 2*n_2d (path + stride per 2D var) + output
    expected_argc = len(var_items) + 2 * len(var2d_items) + 4
    text += f"    if (argc != {expected_argc}) {{\n"
    text += '        std::cerr << "Usage: <exe> <N> <row_size>'
    for name, _ in var_items:
        text += f" <{name}_path>"
    for name, _ in var2d_items:
        text += f" <{name}_path> <{name}_stride>"
    text += ' <output_path>" << std::endl;\n'
    text += "        return 1;\n"
    text += "    }\n\n"

    text += "    const int N = std::atoi(argv[1]);\n"
    text += "    if (N < 0) {\n"
    text += '        std::cerr << "N must be non-negative" << std::endl;\n'
    text += "        return 1;\n"
    text += "    }\n"
    text += "    const int row_size = std::atoi(argv[2]);\n"
    text += "    if (row_size <= 0) {\n"
    text += '        std::cerr << "row_size must be positive" << std::endl;\n'
    text += "        return 1;\n"
    text += "    }\n\n"

    # 1D vars: argv[3 .. 3+n_1d-1]
    for idx, (name, elt_type) in enumerate(var_items):
        cpp_type = elt_type.to_str()
        arg_idx = idx + 3
        text += f"    std::vector<{cpp_type}> {name}(N);\n"
        text += f"    if (!read_binary<{cpp_type}>(argv[{arg_idx}], {name}.data(), N)) {{\n"
        text += f'        std::cerr << "Failed to read input for {name}" << std::endl;\n'
        text += "        return 2;\n"
        text += "    }\n"
    text += "\n"

    # 2D vars: argv[3+n_1d ..] in pairs (path, stride)
    # With causal (look-behind) semantics, N output elements consume exactly N input elements
    # (rows before row 0 are zero-padded by InputSignal2D, so nelem = N).
    base_2d_arg = len(var_items) + 3
    for idx, (name, elt_type) in enumerate(var2d_items):
        cpp_type = elt_type.to_str()
        path_arg_idx = base_2d_arg + 2 * idx
        stride_arg_idx = base_2d_arg + 2 * idx + 1
        text += f"    const int stride_{name} = std::atoi(argv[{stride_arg_idx}]);\n"
        text += f"    const int nelem_{name} = N;\n"
        text += f"    std::vector<{cpp_type}> {name}(nelem_{name});\n"
        text += f"    if (!read_binary<{cpp_type}>(argv[{path_arg_idx}], {name}.data(), nelem_{name})) {{\n"
        text += f'        std::cerr << "Failed to read 2D input for {name}" << std::endl;\n'
        text += "        return 2;\n"
        text += "    }\n"
    text += "\n"

    # Build var_args: 1D args first (sorted), then 2D args (sorted), each as "name.data(), stride_name"
    var_args_1d = [f"{name}.data()" for name, _ in var_items]
    var_args_2d = [f"{name}.data(), stride_{name}" for name, _ in var2d_items]
    var_args = ", ".join(var_args_1d + var_args_2d + ["row_size"])
    output_cpp_type = output_type.to_str()

    text += f"    auto kernel = make_{kernel_name}({var_args});\n"
    text += f"    std::vector<{output_cpp_type}> output(N);\n\n"

    text += "    auto start = std::chrono::high_resolution_clock::now();\n"
    text += "    int pos = 0;\n"
    text += "    while (pos < N) {\n"
    text += f"        {kernel_name}_vector_type out;\n"
    text += "        kernel.run(&out);\n"
    text += "        const int lanes = vec_lanes_of(out);\n"
    text += "        const int copy_count = (N - pos) < lanes ? (N - pos) : lanes;\n"
    text += f"        std::memcpy(output.data() + pos, &out, sizeof({output_cpp_type}) * copy_count);\n"
    text += "        pos += lanes;\n"
    text += "        if (pos % row_size == 0) {\n"
    text += "            kernel.reset_and_next_row();\n"
    text += "        }\n"
    text += "    }\n"
    text += "    auto end = std::chrono::high_resolution_clock::now();\n"
    text += "\n"

    output_arg_idx = len(var_items) + 2 * len(var2d_items) + 3
    text += f"    if (!write_binary<{output_cpp_type}>(argv[{output_arg_idx}], output.data(), N)) {{\n"
    text += '        std::cerr << "Failed to write output" << std::endl;\n'
    text += "        return 3;\n"
    text += "    }\n"
    text += "\n"

    text += "    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();\n"
    text += '    std::cout << "{\\"elapsed_us\\":" << elapsed_us << "}" << std::endl;\n'
    text += "    return 0;\n"
    text += "}\n"

    return text


def generate_kernel_executable(
    codegen: CodeGen,
    expr: SignalExpr | SignalExpr2D,
    header_path: Optional[str] = None,
    runner_path: Optional[str] = None,
    executable_path: Optional[str] = None,
    compiler: str = "clang++",
    compiler_flags: Optional[List[str]] = None,
    row_size: int = 1024,
) -> KernelExecutable:
    """
    Generate and compile a single-kernel executable for Python interop.

    The generated executable consumes raw binary input arrays and writes a raw
    binary output array. The returned KernelExecutable handles Python/NumPy
    translation and subprocess execution.
    """
    if compiler_flags is None:
        compiler_flags = ["-std=c++20", "-O3", "-march=native", "-I", "."]

    if header_path is None:
        fd, header_path = tempfile.mkstemp(suffix=".h", prefix="header_", dir="/tmp")
        os.close(fd)

    if runner_path is None:
        fd, runner_path = tempfile.mkstemp(suffix=".cpp", prefix="kernel_runner_", dir="/tmp")
        os.close(fd)

    if executable_path is None:
        fd, executable_path = tempfile.mkstemp(prefix="kernel_runner_", dir="/tmp")
        os.close(fd)

    kernel_name = "k0"
    code = codegen.generate(expr, kernel_name)
    instantiate_kernels(header_path, [code])

    all_vars, all_vars2d = codegen.collect_input_variables(expr)
    runner_code = _generate_kernel_runner_code(
        all_vars=all_vars,
        all_vars2d=all_vars2d,
        output_type=expr.element_type,
        kernel_name=kernel_name,
        header_path=header_path,
    )
    with open(runner_path, "w") as f:
        f.write(runner_code)

    compile_cmd = [compiler] + compiler_flags + [runner_path, "-o", executable_path]
    compile_result = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
    )
    if compile_result.returncode != 0:
        raise RuntimeError(f"Compilation failed: {compile_result.stderr}")

    return KernelExecutable(
        executable_path=executable_path,
        variable_order=sorted(all_vars.keys()),
        variable_types=all_vars,
        vars2d_info=all_vars2d,
        output_type=expr.element_type,
        row_size=row_size,
    )


def generate_and_run_benchmark(
    codegen: CodeGen,
    exprs: Sequence[SignalExpr | SignalExpr2D],
    include_correctness_check: bool = False,
    header_path: Optional[str] = None,
    benchmark_path: Optional[str] = None,
    executable_path: Optional[str] = None,
    correctness_tolerance: float = 1e-3,
    input_size: int = (1 << 20),
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    compiler: str = "clang++",
    compiler_flags: Optional[List[str]] = None,
    row_size: int = 1024,
) -> dict:
    """
    Generate, compile, and run a benchmark for the given kernels.

    Args:
        codegen: CodeGen instance used to generate the kernels
        exprs: List of signal expressions corresponding to each kernel
        header_path: Path where the kernel header file will be written
        benchmark_path: Path where the benchmark C++ file will be written (default: random file in /tmp/)
        executable_path: Path for the compiled benchmark executable (default: random file in /tmp/)
        include_correctness_check: If True, adds correctness checking code
        correctness_tolerance: Tolerance for floating point comparisons in correctness checking
        input_size: Size of input data for benchmarking
        warmup_iterations: Number of warmup iterations before timing
        benchmark_iterations: Number of iterations for timing
        compiler: Compiler to use (default: clang++)
        compiler_flags: Additional compiler flags (default: ["-std=c++20", "-O2", "-march=native"])
        row_size: Number of elements per row (used to call reset_and_next_row at row boundaries)

    Returns:
        A dictionary containing:
            - 'output': stdout from benchmark execution
            - 'error': stderr if any errors occurred
            - 'return_code': return code from benchmark execution
    """
    if compiler_flags is None:
        compiler_flags = ["-std=c++20", "-O3", "-march=native", "-I", "."]

    if header_path is None:
        fd, header_path = tempfile.mkstemp(suffix=".h", prefix="header_", dir="/tmp")
        print(header_path)
        os.close(fd)  # Close the file descriptor, we'll write to it later

    # Generate unique random file names if not provided
    if benchmark_path is None:
        fd, benchmark_path = tempfile.mkstemp(suffix=".cpp", prefix="benchmark_", dir="/tmp")
        print(benchmark_path)
        os.close(fd)  # Close the file descriptor, we'll write to it later

    if executable_path is None:
        fd, executable_path = tempfile.mkstemp(prefix="benchmark_", dir="/tmp")
        os.close(fd)  # Close the file descriptor

    kernel_names = [f"k{i}" for i in range(len(exprs))]

    # Generate kernel code
    codes = [codegen.generate(expr, name) for expr, name in zip(exprs, kernel_names)]
    instantiate_kernels(header_path, codes)

    # Generate benchmark program
    generate_benchmark(
        codegen=codegen,
        exprs=exprs,
        kernel_names=kernel_names,
        kernel_header_path=header_path,
        output_path=benchmark_path,
        include_correctness_check=include_correctness_check,
        correctness_tolerance=correctness_tolerance,
        input_size=input_size,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        row_size=row_size,
    )

    # Compile benchmark
    compile_cmd = [compiler] + compiler_flags + [benchmark_path, "-o", executable_path]
    compile_result = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
    )

    if compile_result.returncode != 0:
        result = {
            "output": compile_result.stdout,
            "error": f"Compilation failed: {compile_result.stderr}",
            "return_code": compile_result.returncode,
        }
    else:

        # Run benchmark
        run_result = subprocess.run(
            [os.path.abspath(executable_path)],
            capture_output=True,
            text=True,
        )

        print(executable_path)
        print("result:", run_result.stdout)
        json_output = json.loads(run_result.stdout)
        validation = json_output.pop("validation", None)
        # Since Python 3.7, Python dicts maintain insertion order.
        json_output = dict(sorted(json_output.items(), key=lambda item: item[1]))
        if validation is not None:
            json_output["validation"] = validation

        result = {
            "output": json_output,
            "error": run_result.stderr if run_result.returncode != 0 else "",
            "return_code": run_result.returncode,
        }

    if result["return_code"] != 0:
        raise RuntimeError(f"Benchmark execution failed: {result['error']}")
    return result
