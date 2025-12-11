from enum import Enum
from typing import List, Optional, TYPE_CHECKING
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

class CodeGen:
    # Hardware-dependent maximum
    max_bits: int
    counter: int
    prologue: list[str]

    def __init__(self, max_bits: int):
        assert max_bits in [64, 128, 256, 512]
        self.max_bits = max_bits
        self.counter = 0
        self.prologue = []

    def get_new_var(self) -> str:
        self.counter += 1
        return f"v{self.counter}"

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

    def clear(self):
        self.counter = 0
        self.prologue = []

    def generate(self, expr: SignalExpr, name: str) -> Code:
        self.clear()
        code = self.generate_signal(expr)
        vars = self.collect_variables(expr)
        args = ", ".join(f"const {ElementType.Float.to_str()} *{var}" for var in vars)
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
                program = f"make_s_convolve<{vec_type}>({code_a.text}, {code_f.text})"
                return Code(program, element_type, code_a.lanes)
            case Recurse(a, g):
                time_delay, a = a.time_delay(self.max_lanes(ElementType.Float))
                code_a = self.generate_kernel(a)
                code_g = self.generate_signal(g)

                assert code_a.element_type == code_g.element_type
                element_type = code_a.element_type

                # Lanes of an IIR depends on its time delay (and hardware limit)
                lanes = time_delay

                code_a = self.enforce_lanes(code_a, lanes)
                code_g = self.enforce_lanes(code_g, lanes)
                vec_type = self.get_vec_type(element_type, lanes)

                program = f"make_s_recurse<{vec_type}>({code_a.text}, {code_g.text})"
                return Code(program, element_type, lanes)

        assert False

    def generate_kernel(self, expr: KernelExpr) -> Code:
        match expr:
            case TIKernel(_):
                vec_type = self.get_vec_type(ElementType.Float)

                taps, indices, values = expr.to_sparse_repr()
                index_args = ", ".join(map(str, indices))
                value_args = ", ".join(map(str, values))

                index_var = self.get_new_var()
                value_var = self.get_new_var()

                self.prologue.append(
                    f"constexpr static int {index_var}[{taps}] = {{{index_args}}};",
                )
                self.prologue.append(
                    f"constexpr static {ElementType.Float.to_str()} {value_var}[{taps}] = {{{value_args}}};",
                )

                return Code(
                    f"TimeInvariantKernel<{taps}, {vec_type}, {index_var}, {value_var}>()",
                    ElementType.Float,
                    self.max_lanes(ElementType.Float),
                    taps=taps,
                )
            case TVKernel(_):
                taps, indices, signals = expr.to_sparse_repr()
                signal_codes = [self.generate_signal(signal) for signal in signals]
                signal_codes = [self.enforce_lanes(code) for code in signal_codes]
                element_type = signal_codes[0].element_type
                vec_type = self.get_vec_type(element_type)
                taps = len(signal_codes)

                arguments = ", ".join(c.text for c in signal_codes)

                index_args = ", ".join(map(str, indices))
                index_var = self.get_new_var()
                self.prologue.append(
                    f"constexpr static int {index_var}[{taps}] = {{{index_args}}};",
                )

                return Code(
                    f"TimeVaryingKernel<{taps}, {vec_type}, {index_var}>({arguments})",
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
                code.element_type,
                lanes,
                taps=code.taps,
            )
        else:  # code.lanes > lanes
            # downscale lanes
            return Code(
                f"{function_name}<{vec_type_in}, {vec_type_out}>({code.text})",
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

def instantiate_kernels(path: str, codes: List[Code]) -> None:
    text = TEMPLATE.read_text()
    for code in codes:
        text += "\n" + code.text
    with open(path, "w") as f:
        f.write(text)


def generate_benchmark(
    codegen: CodeGen,
    exprs: List[SignalExpr],
    kernel_names: List[str],
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
    
    # Collect variables from all expressions
    all_vars = set()
    for expr in exprs:
        vars = codegen.collect_variables(expr)
        all_vars.update(vars)
    
    # Generate the benchmark program
    benchmark_code = _generate_benchmark_code(
        list(all_vars),
        kernel_names,
        include_correctness_check,
        correctness_tolerance,
        input_size,
        warmup_iterations,
        benchmark_iterations,
    )
    
    with open(output_path, "w") as f:
        f.write(benchmark_code)


def _generate_benchmark_code(
    all_vars: List[str],
    kernel_names: List[str],
    include_correctness_check: bool,
    correctness_tolerance: float,
    input_size: int,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> str:
    """Generate the complete C++ benchmark code."""
    
    # Start with includes
    benchmark_text = '#include "output.h"\n'
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
    benchmark_text += "    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);\n\n"
    
    for var in sorted(all_vars):
        benchmark_text += f"    std::vector<float> {var}(N);\n"
        benchmark_text += f"    for (int i = 0; i < N; i++) {{\n"
        benchmark_text += f"        {var}[i] = dis(gen);\n"
        benchmark_text += f"    }}\n\n"
    
    # Generate output buffers for each kernel
    benchmark_text += "    // Output buffers for each kernel\n"
    for i, name in enumerate(kernel_names):
        benchmark_text += f"    std::vector<float> output_{name}(N);\n"
    benchmark_text += "\n"
    
    # Generate benchmark code for each kernel
    var_args = ", ".join(f"{var}.data()" for var in sorted(all_vars))
    
    for i, name in enumerate(kernel_names):
        benchmark_text += f"    // Benchmark {name}\n"
        benchmark_text += f"    {{\n"
        
        benchmark_text += f"        auto kernel = make_{name}({var_args});\n"
        benchmark_text += f"        \n"
        benchmark_text += f"        // Warmup\n"
        benchmark_text += f"        for (int i = 0; i < warmup; i++) {{\n"
        benchmark_text += f"            {name}_vector_type out;\n"
        benchmark_text += f"            kernel.run(&out);\n"
        benchmark_text += f"        }}\n\n"
        
        benchmark_text += f"        // Reset kernel state\n"
        benchmark_text += f"        kernel = make_{name}({var_args});\n\n"
        
        benchmark_text += f"        // Timed run\n"
        benchmark_text += f"        auto start = std::chrono::high_resolution_clock::now();\n"
        benchmark_text += f"        for (int iter = 0; iter < iterations; iter++) {{\n"
        benchmark_text += f"            // Recreate kernel for each iteration\n"
        benchmark_text += f"            kernel = make_{name}({var_args});\n"
        benchmark_text += f"            int pos = 0;\n"
        benchmark_text += f"            while (pos < N) {{\n"
        benchmark_text += f"                {name}_vector_type out;\n"
        benchmark_text += f"                kernel.run(&out);\n"
        benchmark_text += f"                memcpy(&output_{name}[pos], &out, sizeof(out));\n"
        benchmark_text += f"                pos += vec_lanes_of(out);\n"
        benchmark_text += f"            }}\n"
        benchmark_text += f"        }}\n"
        benchmark_text += f"        auto end = std::chrono::high_resolution_clock::now();\n"
        benchmark_text += f"        \n"
        benchmark_text += f"        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();\n"
        benchmark_text += f'        std::cout << "{name}: " << duration / static_cast<double>(iterations) << " us per iteration\\n";\n'
        benchmark_text += f"    }}\n\n"
    
    # Add correctness checking if requested
    if include_correctness_check and len(kernel_names) > 1:
        benchmark_text += _generate_correctness_check_code(kernel_names, input_size)
    
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
bool arrays_equal(const std::vector<float>& a, const std::vector<float>& b, float tolerance = {tolerance}) {{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {{
        if (std::abs(a[i] - b[i]) > tolerance) {{
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << "\\n";
            return false;
        }}
    }}
    return true;
}}

"""


def _generate_correctness_check_code(kernel_names: List[str], input_size: int) -> str:
    """
    Generate correctness checking code comparing all kernels.
    
    Compares each kernel's output against the first kernel (reference).
    
    Returns:
        C++ code for correctness checking with trailing newlines for proper spacing.
    """
    code = "    // Correctness checking\n"
    code += '    std::cout << "\\nCorrectness checking:\\n";\n'
    
    reference = kernel_names[0]
    for i in range(1, len(kernel_names)):
        name = kernel_names[i]
        code += f'    if (arrays_equal(output_{reference}, output_{name})) {{\n'
        code += f'        std::cout << "{name} matches {reference}: PASS\\n";\n'
        code += f'    }} else {{\n'
        code += f'        std::cout << "{name} matches {reference}: FAIL\\n";\n'
        code += f'    }}\n'
    
    code += "\n"
    return code
