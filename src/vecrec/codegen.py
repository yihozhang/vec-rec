from typing import List, Optional, Sequence, Dict, Union
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.expr.signal import Var2D
from vecrec.expr.signal_ops import Repeater, Ith
from vecrec.expr.kernel import TIKernel, TVKernel
from vecrec.util import ElementType
import json
from importlib.resources import files

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
    # Maps Var2D names to their context variable names (str) and n_rows (int)
    var2d_context: Dict[str, Union[str, int]]

    def __init__(self):
        self.counter = 0
        self.prologue = []
        self.var2d_context = {}

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

    def get_sem_suffix(self, ty: Type) -> str:
        match ty:
            case Type.Arith:
                return ""
            case Type.TropMax:
                return "TropMax"
            case Type.TropMin:
                return "TropMin"

    def generate(self, expr: SignalExpr | SignalExpr2D, name: str) -> Code:
        self.clear()
        code = self.generate_signal(expr)
        vars = self.collect_variables(expr)
        args = ", ".join(f"const {elt_type.to_str()} *{var}" for var, elt_type in sorted(vars.items()))
        text = "\n".join(
            [
                f"auto make_{name}({args}) {{",
                *map(lambda x: "    " + x, self.prologue),
                f"    return {code.text};",
                "}",
                f"using {name}_vector_type = {self.get_vec_type(code.element_type, code.lanes)};"
            ]
        )
        return Code(
            text,
            code.ty,
            code.element_type,
            code.lanes,
        )

    def generate_signal(self, expr: SignalExpr | SignalExpr2D) -> Code:
        assert expr.lanes is not None
        match expr:
            case Num(value):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                value_str = expr.element_type.val_to_str(value, expr.ty)
                return Code(
                    f"Signal1DConstant<{vec_type}>({value_str})",
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
                return Code(program, ty, element_type, expr.lanes)

            case Var(name):
                return Code(
                    f"Signal1D<{self.get_vec_type(expr.element_type, expr.lanes)}>({name})",
                    expr.ty,
                    expr.element_type,
                    expr.lanes,
                )
            case Var2D(name):
                # Var2D should only appear within a Repeater context
                if name not in self.var2d_context:
                    raise ValueError(f"Var2D '{name}' used outside of Repeater context")
                
                context_var = self.var2d_context[name]
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                n_rows = self.var2d_context[f"{name}_n_rows"]
                
                return Code(
                    f"make_signal2d<{vec_type}, {n_rows}>({context_var})",
                    expr.ty,
                    expr.element_type,
                    expr.lanes,
                )
            case Repeater(a, n_rows, prev_rows_var):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                
                # Create a context variable
                context_var = self.get_new_var()
                self.prologue.append(
                    f"RepeaterContext<{vec_type}, {n_rows}>* {context_var} = new RepeaterContext<{vec_type}, {n_rows}>();"
                )
                
                # Add the Var2D to our context so inner signal can reference it
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
                
                return Code(program, expr.ty, expr.element_type, expr.lanes)
            case Convolve(a, f):
                code_a = self.generate_kernel(a)
                code_f = self.generate_signal(f)

                assert code_a.element_type == code_f.element_type
                assert code_a.ty == code_f.ty
                element_type = code_a.element_type
                ty = code_a.ty

                vec_type = self.get_vec_type(element_type, expr.lanes)
                program = f"make_s_convolve<{vec_type}>({code_a.text}, {code_f.text})"
                return Code(program, ty, element_type, expr.lanes)
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

                program = f"make_s_recurse<{vec_type}>({code_a.text}, {code_g.text})"
                return Code(program, ty, element_type, a.lanes)
            case Ith(signal2d, i):
                # Ith extracts the ith row from a 2D signal
                # Generate: make_ith_row<vec_type, n_rows, i>(signal2d_code)
                element_type = expr.element_type
                ty = expr.ty
                vec_type = self.get_vec_type(element_type, expr.lanes)
                
                # Generate code for the 2D signal (typically Var2D -> Signal2D)
                code_signal2d = self.generate_signal(signal2d)
                
                # Get n_rows from context (signal2d should be Var2D with context)
                # There are two Signal2D types right now: Var2D and Repeater
                if isinstance(signal2d, Var2D):
                    if signal2d.name not in self.var2d_context:
                        raise ValueError(f"Var2D '{signal2d.name}' used outside of Repeater context")
                    n_rows = self.var2d_context[f"{signal2d.name}_n_rows"]
                elif isinstance(signal2d, Repeater):
                    n_rows = signal2d.n_rows
                else:
                    raise ValueError(f"Ith expects Var2D or Repeater as signal2d, got {type(signal2d)}")
                
                # Generate make_ith_row call
                program = f"make_ith_row<{vec_type}, {n_rows}, {i}>({code_signal2d.text})"
                return Code(program, ty, element_type, expr.lanes)
            case ConvertLanes(a):
                code_a = self.generate_signal(a)
                code_a = self.enforce_lanes(code_a, expr.lanes)
                return code_a
        assert False

    def generate_kernel(self, expr: KernelExpr) -> Code:
        assert expr.lanes is not None
        match expr:
            case TIKernel(_):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)

                taps, indices, values = expr.to_sparse_repr()
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

                return Code(
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

                return Code(
                    f"make_time_varying_kernel<{taps}, {vec_type}, {index_var}>({arguments})",
                    ty,
                    element_type,
                    expr.lanes,
                    taps=taps,
                )
            case KConvertLanes(a):
                code_a = self.generate_kernel(a)
                code_a = self.enforce_lanes(code_a, expr.lanes)
                return code_a
            case KAdd(_, _) | KSub(_, _) | KNeg(_) | KConvolve(_, _):
                raise NotImplementedError("These apps should have been eliminated by constant folding")
            case _:
                assert False


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

    def collect_variables(
        self, expr: RecLang, vars: Optional[Dict[str, ElementType]] = None
    ) -> Dict[str, ElementType]:
        if vars is None:
            vars = {}

        match expr:
            case Var(name):
                if name in vars and vars[name] != expr.element_type:
                    raise ValueError(f"Variable {name} has inconsistent element types: {vars[name]} vs {expr.element_type}")
                vars[name] = expr.element_type
            case _:
                for c in expr.children():
                    self.collect_variables(c, vars)
            # case SignalExprBinOp(a, b):
            #     self.collect_variables(a, vars)
            #     self.collect_variables(b, vars)
            # case Convolve(a, f):
            #     self.collect_variables(a, vars)
            #     self.collect_variables(f, vars)
            # case Recurse(a, g):
            #     self.collect_variables(a, vars)
            #     self.collect_variables(g, vars)
            # case KAdd(a, b) | KSub(a, b) | KConvolve(a, b):
            #     self.collect_variables(a, vars)
            #     self.collect_variables(b, vars)
            # case KNeg(a):
            #     self.collect_variables(a, vars)
            # case TIKernel() | TVKernel() | Num():
            #     pass

        return vars

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
    """
    assert len(exprs) == len(kernel_names), "Number of expressions and kernel names must match"
    assert len(exprs) > 0, "At least one expression must be provided"

    # Collect variables from all expressions
    all_vars: Dict[str, ElementType] = {}
    for expr in exprs:
        vars_dict = codegen.collect_variables(expr)
        all_vars.update(vars_dict)

    # Generate the benchmark program
    benchmark_code = _generate_benchmark_code(
        all_vars,
        exprs[0].element_type,
        kernel_names,
        kernel_header_path,
        include_correctness_check,
        correctness_tolerance,
        input_size,
        warmup_iterations,
        benchmark_iterations,
    )
    
    with open(output_path, "w") as f:
        f.write(benchmark_code)


def _generate_benchmark_code(
    all_vars: Dict[str, ElementType],
    output_type: ElementType,
    kernel_names: Sequence[str],
    header_path: str,
    include_correctness_check: bool,
    correctness_tolerance: float,
    input_size: int,
    warmup_iterations: int,
    benchmark_iterations: int,
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
    benchmark_text += f"    const int warmup = {warmup_iterations};\n"
    benchmark_text += f"    const int iterations = {benchmark_iterations};\n\n"

    # Generate input data initialization
    benchmark_text += "    // Initialize input data\n"
    benchmark_text += "    std::random_device rd;\n"
    benchmark_text += "    std::mt19937 gen(rd());\n"

    # Determine which distributions are needed
    needs_float = any(elt_type == ElementType.Float for elt_type in all_vars.values())
    needs_int32 = any(elt_type == ElementType.I32 for elt_type in all_vars.values())
    needs_int64 = any(elt_type == ElementType.I64 for elt_type in all_vars.values())

    if needs_float:
        benchmark_text += "    std::uniform_real_distribution<float> float_dis(-1.0f, 1.0f);\n"
    if needs_int32:
        benchmark_text += "    std::uniform_int_distribution<int32_t> int32_dis(-1000, 1000);\n"
    if needs_int64:
        benchmark_text += "    std::uniform_int_distribution<int64_t> int64_dis(-1000000, 1000000);\n"
    benchmark_text += "\n"

    for var, elt_type in sorted(all_vars.items()):
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
    
    # Generate output buffers for each kernel
    benchmark_text += "    // Output buffers for each kernel\n"
    for i, name in enumerate(kernel_names):
        benchmark_text += f"    std::vector<{output_type.to_str()}> output_{name}(N);\n"
    benchmark_text += "\n"
    
    # Generate benchmark code for each kernel
    var_args = ", ".join(f"{var}.data()" for var in sorted(all_vars))
    
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
        
        benchmark_text += "                    if (pos % 1024 == 0) {\n"
        benchmark_text += "                        kernel.reset_and_next_row();\n"
        benchmark_text += "                    }\n"
        
        benchmark_text += "                }\n"
        benchmark_text += "            }\n"
        benchmark_text += "        }\n"
        benchmark_text += "        auto end = std::chrono::high_resolution_clock::now();\n"
        benchmark_text += "        \n"
        benchmark_text += "        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();\n"
        # Deciding if we need to generate the trailing comma
        if i == len(kernel_names) - 1 and not (include_correctness_check and len(kernel_names) > 1):
            benchmark_text += f'        std::cout << "  \\"{name}\\": " << duration / static_cast<double>(iterations) << "\\n";\n'
        else:
            benchmark_text += f'        std::cout << "  \\"{name}\\": " << duration / static_cast<double>(iterations) << ",\\n";\n'
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
    code += '    int idx;'
    
    reference = kernel_names[0]
    for i in range(1, len(kernel_names)):
        name = kernel_names[i]
        code += f'    if ((idx=arrays_equal(output_{reference}, output_{name})) == output_{reference}.size()) {{\n'
        code += f'        std::cout << "    \\"{name}\\": true";\n'
        code += '    } else {\n'
        code += f'        std::cout << "    \\"{name}\\":" << idx;\n'
        code += '    }\n'
        if i == len(kernel_names) - 1:
            code += '    std::cout << "\\n";\n'
        else:
            code += '    std::cout << ",\\n";\n'
    
    code += '    std::cout << "  }\\n";\n'
    code += "\n"
    return code


def generate_and_run_benchmark(
    codegen: CodeGen,
    exprs: Sequence[SignalExpr | SignalExpr2D],
    kernel_names: Sequence[str],
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
) -> dict:
    """
    Generate, compile, and run a benchmark for the given kernels.
    
    Args:
        codegen: CodeGen instance used to generate the kernels
        exprs: List of signal expressions corresponding to each kernel
        kernel_names: List of names corresponding to each kernel
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
    
    Returns:
        A dictionary containing:
            - 'output': stdout from benchmark execution
            - 'error': stderr if any errors occurred
            - 'return_code': return code from benchmark execution
    """
    import subprocess
    import os
    import tempfile
    
    if compiler_flags is None:
        compiler_flags = ["-std=c++20", "-O3", "-march=native", "-I", "."]
    
    if header_path is None:
        fd, header_path= tempfile.mkstemp(suffix='.h', prefix='header_', dir='/tmp')
        print(header_path)
        os.close(fd)  # Close the file descriptor, we'll write to it later

    # Generate unique random file names if not provided
    if benchmark_path is None:
        fd, benchmark_path = tempfile.mkstemp(suffix='.cpp', prefix='benchmark_', dir='/tmp')
        print(benchmark_path)
        os.close(fd)  # Close the file descriptor, we'll write to it later
    
    if executable_path is None:
        fd, executable_path = tempfile.mkstemp(prefix='benchmark_', dir='/tmp')
        os.close(fd)  # Close the file descriptor
    
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
    )
    
    # Compile benchmark
    compile_cmd = [compiler] + compiler_flags + [benchmark_path, "-o", executable_path]
    compile_result = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
    )
    
    if compile_result.returncode != 0:
        return {
            'output': compile_result.stdout,
            'error': f"Compilation failed: {compile_result.stderr}",
            'return_code': compile_result.returncode,
        }
    
    # Run benchmark
    run_result = subprocess.run(
        [os.path.abspath(executable_path)],
        capture_output=True,
        text=True,
    )

    print(executable_path)
    print('result:', run_result.stdout)    
    json_output = json.loads(run_result.stdout)
    validation = json_output.pop("validation", None)
    # Since Python 3.7, Python dicts maintain insertion order.
    json_output = dict(sorted(json_output.items(), key=lambda item: item[1]))
    if validation is not None:
        json_output["validation"] = validation

    return {
        'output': json_output,
        'error': run_result.stderr if run_result.returncode != 0 else "",
        'return_code': run_result.returncode,
    }
