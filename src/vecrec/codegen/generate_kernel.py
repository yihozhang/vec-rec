from typing import Any, List, Optional, Sequence, Dict, Tuple, Union
from vecrec.codegen.codegen import CodeGen, KernelProgram, ComputedSignalDecl
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D
from vecrec.util import ElementType
import concurrent.futures
import json
import numpy as np
import os
import subprocess
import tempfile
import warnings
from importlib.resources import files

__all__ = [
    "generate_benchmark",
    "instantiate_kernels",
    "generate_kernel_executable",
    "generate_and_run_benchmark",
    "sweep_codegen_setup",
    "find_best_schedule_and_emit_header",
    "MAX_SWEEP_SCHEDULES",
]
TEMPLATE = files("vecrec.templates") / "common.h"

# Upper bound on the number of non-baseline schedules fed into the (epi, rpi)
# sweep. The sweep takes the top ~10% of candidate schedules by default, but
# each schedule fans out over a grid of codegen configs, so a large candidate
# count makes the sweep's compile budget blow up. Cap the non-baseline count so
# at most this many optimized schedules are swept (the reference schedule k0 is
# always kept in addition, as the baseline).
MAX_SWEEP_SCHEDULES = 40


def instantiate_kernels(path: str, codes: List[KernelProgram] | List[str]) -> None:
    text = TEMPLATE.read_text()
    # Emit each custom semiring's preamble exactly once, regardless of how many kernels use it.
    seen_semirings: set = set()
    for code in codes:
        if isinstance(code, KernelProgram):
            for sem in code.custom_semirings:
                if sem not in seen_semirings:
                    seen_semirings.add(sem)
                    text += "\n" + sem.cpp_preamble()
    # Emit each CustomPointwiseOp struct definition (the user-provided operator
    # struct) exactly once, before the kernel structs that reference it.
    seen_op_preludes: set = set()
    for code in codes:
        if isinstance(code, KernelProgram):
            for prelude in code.custom_op_preludes:
                if prelude not in seen_op_preludes:
                    seen_op_preludes.add(prelude)
                    text += "\n" + prelude
    # Emit each ComputedSignal2D prelude (the user-provided signal struct definition)
    # exactly once, before the kernel structs that reference it.
    seen_preludes: set = set()
    for code in codes:
        if isinstance(code, KernelProgram):
            for decl in code.computed_signals:
                if decl.prelude not in seen_preludes:
                    seen_preludes.add(decl.prelude)
                    text += "\n" + decl.prelude
    for code in codes:
        text += "\n" + (code.text if isinstance(code, KernelProgram) else code)
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
    ele_per_iter_list: Optional[List[int]] = None,
    rows_per_iter_list: Optional[List[int]] = None,
    computed_signals_list: Optional[List[List[ComputedSignalDecl]]] = None,
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
    if ele_per_iter_list is None:
        assert all(
            expr.lanes is not None for expr in exprs
        ), "If ele_per_iter_list is not provided, all expressions must have a defined lane count"
        ele_per_iter_list = [max(16, expr.lanes) for expr in exprs]  # type: ignore

    if rows_per_iter_list is None:
        rows_per_iter_list = [1] * len(exprs)
    assert all(rpi >= 1 for rpi in rows_per_iter_list), "rows_per_iter must be >= 1"

    if computed_signals_list is None:
        computed_signals_list = [[] for _ in exprs]

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
        ele_per_iter_list,
        rows_per_iter_list,
        computed_signals_list,
        codegen.band_width,
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
    row_size: int,
    ele_per_iter_list: List[int],
    rows_per_iter_list: List[int],
    computed_signals_list: List[List[ComputedSignalDecl]],
    band_width: int = 0,
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
    needs_double = any(elt_type == ElementType.Double for elt_type in all_elt_types)
    needs_int8 = any(elt_type == ElementType.I8 for elt_type in all_elt_types)
    needs_int16 = any(elt_type == ElementType.I16 for elt_type in all_elt_types)
    needs_int32 = any(elt_type == ElementType.I32 for elt_type in all_elt_types)
    needs_int64 = any(elt_type == ElementType.I64 for elt_type in all_elt_types)

    if needs_float:
        benchmark_text += "    std::uniform_real_distribution<float> float_dis(-1.0f, 1.0f);\n"
    if needs_double:
        benchmark_text += "    std::uniform_real_distribution<double> double_dis(-1.0, 1.0);\n"
    if needs_int8:
        benchmark_text += "    std::uniform_int_distribution<int32_t> int8_dis(-1, 1);\n"
    if needs_int16:
        benchmark_text += "    std::uniform_int_distribution<int32_t> int16_dis(-10, 10);\n"
    if needs_int32:
        benchmark_text += "    std::uniform_int_distribution<int32_t> int32_dis(-1000, 1000);\n"
    if needs_int64:
        benchmark_text += "    std::uniform_int_distribution<int64_t> int64_dis(-1000000, 1000000);\n"
    benchmark_text += "\n"

    for var, elt_type in sorted(all_vars.items()) + sorted(all_vars2d.items()):
        cpp_type = elt_type.to_str()
        benchmark_text += f"    std::vector<{cpp_type}> {var}(N);\n"
        benchmark_text += "    for (int i = 0; i < N; i++) {\n"
        if elt_type == ElementType.Float:
            benchmark_text += f"        {var}[i] = float_dis(gen);\n"
        elif elt_type == ElementType.Double:
            benchmark_text += f"        {var}[i] = double_dis(gen);\n"
        elif elt_type == ElementType.I8:
            benchmark_text += f"        {var}[i] = static_cast<int8_t>(int8_dis(gen));\n"
        elif elt_type == ElementType.I16:
            benchmark_text += f"        {var}[i] = static_cast<int16_t>(int16_dis(gen));\n"
        elif elt_type == ElementType.I32:
            benchmark_text += f"        {var}[i] = int32_dis(gen);\n"
        else:  # I64
            benchmark_text += f"        {var}[i] = int64_dis(gen);\n"
        benchmark_text += "    }\n\n"

    # Allocate 2D input arrays. The benchmark row width matches RepeaterContext::width (1024).
    # With causal (look-behind) semantics, N output elements need exactly N input elements.
    _benchmark_row_width = 1024

    # Emit each unique ComputedSignal2D setup block once. These allocate+fill the raw
    # buffers the computed signals read; all kernels in this binary share them, so the
    # cross-schedule correctness check compares kernels on identical data.
    benchmark_text += "    // ComputedSignal2D buffer setup\n"
    seen_setup: set = set()
    for sig_list in computed_signals_list:
        for decl in sig_list:
            if decl.setup_code not in seen_setup:
                seen_setup.add(decl.setup_code)
                for line in decl.setup_code.splitlines():
                    benchmark_text += f"    {line}\n"
    benchmark_text += "\n"

    # Generate output buffers for each kernel
    benchmark_text += "    // Output buffers for each kernel\n"
    for i, name in enumerate(kernel_names):
        benchmark_text += f"    std::vector<{output_type.to_str()}> output_{name}(N);\n"
    benchmark_text += "\n"

    # Generate benchmark code for each kernel. ComputedSignal2D objects are constructed
    # inline (per kernel, since their resolved C++ type depends on the kernel's lanes)
    # and prepended to the raw input pointers, matching the constructor signature order.
    var_args_1d = [f"{var}.data()" for var in sorted(all_vars)]
    var_args_2d = [f"{var}.data(), row_size" for var in sorted(all_vars2d)]
    var_args_base = var_args_1d + var_args_2d + ["row_size"]

    def kernel_ctor_args(idx: int) -> str:
        # Buffer-less computed signals (empty make_args) are default-constructed as
        # kernel members, so they are not passed as constructor arguments.
        sig_args = [f"{d.resolved_type}({d.make_args})" for d in computed_signals_list[idx] if d.make_args.strip()]
        return ", ".join(sig_args + var_args_base)

    benchmark_text += '    std::cout << "{\\n";'

    for i, name in enumerate(kernel_names):
        epi = ele_per_iter_list[i]
        rpi = rows_per_iter_list[i]
        benchmark_text += f"    // Benchmark {name}\n"
        benchmark_text += "    {\n"

        benchmark_text += "        // Warmup\n"
        benchmark_text += "        {\n"
        benchmark_text += f"            auto kernel = {name}({kernel_ctor_args(i)});\n"
        benchmark_text += f"            using elt_type = typename ElementType<{name}_vector_type>::type;\n"
        benchmark_text += f"            constexpr int vec_lanes = sizeof({name}_vector_type) / sizeof(elt_type);\n"
        # Reset at row boundaries, exactly as the timed loop does. Stateful operators
        # (Repeater/computed signals) advance an internal column cursor by ele_per_iter
        # each run() and rely on reset_and_next_row() to wrap it back to 0 at the end of
        # a row; their buffers are sized for one row. Calling run() warmup times without
        # this discipline lets the cursor run past row_size and overrun those buffers
        # (only observable when warmup*ele_per_iter > row_size, e.g. narrow element types).
        benchmark_text += "            int warm_col = 0;\n"
        benchmark_text += "            for (int i = 0; i < warmup; i++) {\n"
        benchmark_text += f"                {name}_vector_type out_buf[{rpi}][{epi}/vec_lanes];\n"
        benchmark_text += f"                {name}_vector_type* out_per_row[{rpi}] = {{\n"
        for r in range(rpi):
            benchmark_text += f"                    out_buf[{r}],\n"
        benchmark_text += "                };\n"
        benchmark_text += "                kernel.run(out_per_row);\n"
        benchmark_text += f"                warm_col += {epi};\n"
        benchmark_text += "                if (warm_col >= row_size) {\n"
        benchmark_text += f"                    for (int r = 0; r < {rpi}; r++) kernel.reset_and_next_row();\n"
        benchmark_text += "                    warm_col = 0;\n"
        benchmark_text += "                }\n"
        benchmark_text += "            }\n"
        benchmark_text += "        }\n\n"

        benchmark_text += "        // Timed run (one wall-clock sample per iteration)\n"
        benchmark_text += "        std::vector<double> samples_us;\n"
        benchmark_text += "        samples_us.reserve(static_cast<std::size_t>(iterations));\n"
        benchmark_text += "        for (int iter = 0; iter < iterations; iter++) {\n"
        benchmark_text += "            auto start = std::chrono::high_resolution_clock::now();\n"
        benchmark_text += f"            auto kernel = {name}({kernel_ctor_args(i)});\n"
        benchmark_text += f"            using elt_type = typename ElementType<{name}_vector_type>::type;\n"
        benchmark_text += f"            constexpr int vec_lanes = sizeof({name}_vector_type) / sizeof(elt_type);\n"
        benchmark_text += "            int row_base = 0;\n"
        benchmark_text += "            while (row_base < N) {\n"
        if band_width > 0:
            bw = band_width
            benchmark_text += f"                int _log_row = row_base / row_size;\n"
            # Left side always 0 (no left restriction — avoids diagonal carry bug).
            benchmark_text += f"                int col = 0;\n"
            benchmark_text += f"                int col_end = std::min(row_size, ((_log_row + {rpi} + {bw} + {epi} - 1) / {epi}) * {epi});\n"
            benchmark_text += "                while (col < col_end) {\n"
        else:
            benchmark_text += "                int col = 0;\n"
            benchmark_text += "                while (col < row_size) {\n"
        benchmark_text += f"                    {name}_vector_type out_buf[{rpi}][{epi}/vec_lanes];\n"
        benchmark_text += f"                    {name}_vector_type* out_per_row[{rpi}] = {{\n"
        for r in range(rpi):
            benchmark_text += f"                        out_buf[{r}],\n"
        benchmark_text += "                    };\n"
        benchmark_text += "                    kernel.run(out_per_row);\n"
        benchmark_text += f"                    for (int r = 0; r < {rpi}; r++) {{\n"
        benchmark_text += "                        int dst = row_base + r * row_size + col;\n"
        benchmark_text += "                        if (dst >= N) break;\n"
        benchmark_text += f"                        int copy_count = ((N - dst) < {epi}) ? (N - dst) : {epi};\n"
        benchmark_text += f"                        memcpy(&output_{name}[dst], out_buf[r], sizeof({output_type.to_str()}) * copy_count);\n"
        benchmark_text += "                    }\n"
        benchmark_text += f"                    col += {epi};\n"
        benchmark_text += "                }\n"
        benchmark_text += f"                for (int r = 0; r < {rpi}; r++) kernel.reset_and_next_row();\n"
        benchmark_text += f"                row_base += {rpi} * row_size;\n"
        benchmark_text += "            }\n"
        benchmark_text += "            auto end = std::chrono::high_resolution_clock::now();\n"
        benchmark_text += "            samples_us.push_back(static_cast<double>(\n"
        benchmark_text += "                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));\n"
        benchmark_text += "        }\n"
        benchmark_text += "        \n"
        benchmark_text += "        // XOR sum to prevent optimization of output array\n"
        benchmark_text += "        uint64_t xor_sum = 0;\n"
        benchmark_text += "        for (int i = 0; i < N; i++) {\n"
        benchmark_text += f"            xor_sum ^= static_cast<uint64_t>(output_{name}[i]);\n"
        benchmark_text += "        }\n"
        benchmark_text += f"        volatile uint64_t xor_result_{name} = xor_sum;  // Prevent optimization\n"
        benchmark_text += f"        (void)xor_result_{name};\n"
        benchmark_text += "        \n"
        benchmark_text += "        double elapsed_seconds = 0.0;\n"
        benchmark_text += "        for (double sample_us : samples_us) elapsed_seconds += sample_us / 1e6;\n"
        benchmark_text += "        const double throughput_gbps = elapsed_seconds > 0.0\n"
        benchmark_text += "            ? (bytes_per_iteration * static_cast<double>(iterations)) / elapsed_seconds / 1e6\n"
        benchmark_text += "            : 0.0;\n"
        # Deciding if we need to generate the trailing comma
        if i == len(kernel_names) - 1 and not (include_correctness_check and len(kernel_names) > 1):
            benchmark_text += f'        std::cout << "  \\"{name}\\": {{\\"throughput_mbs\\":" << throughput_gbps << ",\\"samples_us\\":[";\n'
        else:
            benchmark_text += f'        std::cout << "  \\"{name}\\": {{\\"throughput_mbs\\":" << throughput_gbps << ",\\"samples_us\\":[";\n'
        benchmark_text += "        for (std::size_t si = 0; si < samples_us.size(); ++si) {\n"
        benchmark_text += '            if (si) std::cout << ",";\n'
        benchmark_text += "            std::cout << samples_us[si];\n"
        benchmark_text += "        }\n"
        if i == len(kernel_names) - 1 and not (include_correctness_check and len(kernel_names) > 1):
            benchmark_text += '        std::cout << "]}" << "\\n";\n'
        else:
            benchmark_text += '        std::cout << "]}" << ",\\n";\n'
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

int arrays_equal(const std::vector<double>& a, const std::vector<double>& b, double tolerance = {tolerance}) {{
    if (a.size() != b.size()) return -1;
    for (size_t i = 0; i < a.size(); i++) {{
        if (std::abs(a[i] - b[i]) > tolerance) {{
            return i;
        }}
    }}
    return a.size();
}}

int arrays_equal(const std::vector<int8_t>& a, const std::vector<int8_t>& b) {{
    if (a.size() != b.size()) return -1;
    for (size_t i = 0; i < a.size(); i++) {{
        if (a[i] != b[i]) {{
            return i;
        }}
    }}
    return a.size();
}}

int arrays_equal(const std::vector<int16_t>& a, const std::vector<int16_t>& b) {{
    if (a.size() != b.size()) return -1;
    for (size_t i = 0; i < a.size(); i++) {{
        if (a[i] != b[i]) {{
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
    code += "    size_t idx;"

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
        case ElementType.Double:
            return np.dtype(np.float64)
        case ElementType.I32:
            return np.dtype(np.int32)
        case ElementType.I64:
            return np.dtype(np.int64)
        case ElementType.I8:
            return np.dtype(np.int8)
        case ElementType.I16:
            return np.dtype(np.int16)


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

    def run(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        n: Optional[int] = None,
    ) -> tuple[np.ndarray, float, float]:
        """Run the compiled kernel.

        `n` is the output length. It is normally inferred from the input arrays
        (1D length, or H*stride for 2D), but must be supplied explicitly when the
        kernel has no materialized inputs — e.g. a ComputedSignal2D whose buffers
        are owned by the C++ setup code (pass `inputs={}` and an explicit `n`)."""
        inputs_1d, inputs_2d = self._normalize_inputs(inputs)

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
                # print(result.stdout)
                # print(result.stderr)
                metadata = json.loads(result.stdout.strip())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse runner output as JSON: {result.stdout}") from exc

            elapsed_us = float(metadata["elapsed_us"])
            throughput_mbs = float(metadata.get("throughput_mbs", 0.0))

            output_dtype = _numpy_dtype_for_element_type(self.output_type)
            output = np.fromfile(output_path, dtype=output_dtype, count=n)
            if int(output.shape[0]) != n:
                raise RuntimeError(f"Output size mismatch: expected {n}, got {output.shape[0]}")

            return output, elapsed_us, throughput_mbs

    def __call__(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> tuple[np.ndarray, float, float]:
        return self.run(inputs)


def _generate_kernel_runner_code(
    all_vars: Dict[str, ElementType],
    all_vars2d: Dict[str, ElementType],
    output_type: ElementType,
    kernel_name: str,
    header_path: str,
    ele_per_iter: int,
    rows_per_iter: int = 1,
    computed_signals: Optional[List[ComputedSignalDecl]] = None,
) -> str:
    assert rows_per_iter >= 1, "rows_per_iter must be >= 1"
    computed_signals = computed_signals or []
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

    # Emit each unique ComputedSignal2D setup block once (allocates+fills its raw
    # buffers from N / row_size), then construct each signal object inline as the
    # leading kernel-constructor arguments (matching the constructor signature order).
    seen_setup: set = set()
    for decl in computed_signals:
        if decl.setup_code not in seen_setup:
            seen_setup.add(decl.setup_code)
            for line in decl.setup_code.splitlines():
                text += f"    {line}\n"
    text += "\n"

    # Build var_args: computed-signal objects first, then 1D args (sorted), then 2D args (sorted).
    # Buffer-less computed signals (empty make_args) are default-constructed kernel members.
    sig_args = [f"{decl.resolved_type}({decl.make_args})" for decl in computed_signals if decl.make_args.strip()]
    var_args_1d = [f"{name}.data()" for name, _ in var_items]
    var_args_2d = [f"{name}.data(), stride_{name}" for name, _ in var2d_items]
    var_args = ", ".join(sig_args + var_args_1d + var_args_2d + ["row_size"])
    output_cpp_type = output_type.to_str()

    text += f"    auto kernel = {kernel_name}({var_args});\n"
    text += f"    std::vector<{output_cpp_type}> output(N);\n\n"

    text += "    auto start = std::chrono::high_resolution_clock::now();\n"
    # Process rows_per_iter rows per kernel.run call.
    # `row_base` is the element index of the first row in the current block;
    # `col` scans the columns of one row; we advance `row_base` by k*row_size
    # after each block of k rows is fully scanned.
    k = rows_per_iter
    text += f"    using elt_type = typename ElementType<{kernel_name}_vector_type>::type;\n"
    text += f"    constexpr int vec_lanes = sizeof({kernel_name}_vector_type) / sizeof(elt_type);\n"
    text += "    int row_base = 0;\n"
    text += "    while (row_base < N) {\n"
    text += "        int col = 0;\n"
    text += "        while (col < row_size) {\n"
    text += f"            {kernel_name}_vector_type out_buf[{k}][{ele_per_iter}/vec_lanes];\n"
    text += f"            {kernel_name}_vector_type* out_per_row[{k}] = {{\n"
    for r in range(k):
        text += f"                out_buf[{r}],\n"
    text += "            };\n"
    text += "            kernel.run(out_per_row);\n"
    text += f"            for (int r = 0; r < {k}; r++) {{\n"
    text += "                int dst = row_base + r * row_size + col;\n"
    text += "                if (dst >= N) break;\n"
    text += f"                int copy_count = ((N - dst) < {ele_per_iter}) ? (N - dst) : {ele_per_iter};\n"
    text += (
        f"                std::memcpy(output.data() + dst, out_buf[r], sizeof({output_cpp_type}) * copy_count);\n"
    )
    text += "            }\n"
    text += f"            col += {ele_per_iter};\n"
    text += "        }\n"
    text += f"        for (int r = 0; r < {k}; r++) kernel.reset_and_next_row();\n"
    text += f"        row_base += {k} * row_size;\n"
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
    # Throughput in MB/s = (bytes touched per kernel invocation) / elapsed_seconds / 1e6.
    # Bytes touched = N elements * (sum of sizeof for every 1D input, every 2D input, and the output).
    sizeof_terms = [f"sizeof({elt.to_str()})" for _, elt in var_items + var2d_items]
    sizeof_terms.append(f"sizeof({output_cpp_type})")
    bytes_per_iter_expr = " + ".join(sizeof_terms)
    text += (
        "    const double bytes_per_iter = "
        f"static_cast<double>(N) * static_cast<double>({bytes_per_iter_expr});\n"
    )
    text += "    const double elapsed_seconds = static_cast<double>(elapsed_us) / 1e6;\n"
    text += "    const double throughput_mbs = elapsed_seconds > 0.0 ? bytes_per_iter / elapsed_seconds / 1e6 : 0.0;\n"
    text += (
        '    std::cout << "{\\"elapsed_us\\":" << elapsed_us'
        ' << ",\\"throughput_mbs\\":" << throughput_mbs << "}" << std::endl;\n'
    )
    text += "    return 0;\n"
    text += "}\n"

    return text


def _build_kernel_executable(
    code: KernelProgram,
    all_vars: Dict[str, ElementType],
    all_vars2d: Dict[str, ElementType],
    output_type: ElementType,
    kernel_name: str = "k0",
    compiler: str = "clang++",
    compiler_flags: Optional[List[str]] = None,
    row_size: int = 1024,
    quiet: bool = False,
) -> KernelExecutable:
    """Workhorse for turning an already-generated KernelProgram into a runnable
    KernelExecutable. Writes the header, emits the runner C++ source, compiles
    it, and returns the wrapper. The caller is responsible for invoking
    `codegen.generate(...)` and `codegen.collect_input_variables(...)` to
    produce `code`, `all_vars`, and `all_vars2d`.
    """
    compiler_flags = ["-std=c++20", "-O3", "-march=native", "-I", "."] + (compiler_flags or [])

    fd, header_path = tempfile.mkstemp(suffix=".h", prefix="header_", dir="/tmp")
    os.close(fd)

    fd, runner_path = tempfile.mkstemp(suffix=".cpp", prefix="kernel_runner_", dir="/tmp")
    os.close(fd)

    fd, executable_path = tempfile.mkstemp(prefix="kernel_runner_", dir="/tmp")
    os.close(fd)

    instantiate_kernels(header_path, [code])

    runner_code = _generate_kernel_runner_code(
        all_vars=all_vars,
        all_vars2d=all_vars2d,
        output_type=output_type,
        kernel_name=kernel_name,
        header_path=header_path,
        ele_per_iter=code.element_per_iter,
        rows_per_iter=code.rows_per_iter,
        computed_signals=code.computed_signals,
    )
    with open(runner_path, "w") as f:
        f.write(runner_code)
    if not quiet:
        print(runner_path)

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
        output_type=output_type,
        row_size=row_size,
    )


def generate_kernel_executable(
    codegen: CodeGen,
    expr: SignalExpr | SignalExpr2D,
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
    kernel_name = "k0"
    code = codegen.generate(expr, kernel_name)
    all_vars, all_vars2d = codegen.collect_input_variables(expr)
    return _build_kernel_executable(
        code=code,
        all_vars=all_vars,
        all_vars2d=all_vars2d,
        output_type=expr.element_type,
        kernel_name=kernel_name,
        compiler=compiler,
        compiler_flags=compiler_flags,
        row_size=row_size,
    )


def generate_and_run_benchmark(
    codegen: Union[CodeGen, Sequence[CodeGen]],
    exprs: Sequence[SignalExpr | SignalExpr2D],
    include_correctness_check: bool = True,
    correctness_tolerance: float = 1e-3,
    input_size: int = (1 << 20),
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    compiler: str = "clang++",
    compiler_flags: Optional[List[str]] = None,
    row_size: int = 1024,
    schedules_per_runner: int = 5,
    parallel_compile: Optional[bool] = None,
) -> dict:
    """
    Generate, compile, and run a benchmark for the given kernels.

    ``codegen`` may be either a single ``CodeGen`` (applied to every expression)
    or a sequence of ``CodeGen`` instances parallel to ``exprs`` (used by the
    sweep to vary ``(ele_per_iter_multiple, rows_per_iter)`` per kernel).

    The schedules are split into batches of size ``schedules_per_runner``
    (each batch always includes the reference schedule k0). Every batch is
    compiled into its own runner; runners are compiled in parallel. After
    running, each batch's reported throughputs are normalized so that all
    batches agree on a common k0 throughput, anchored to batch 0's k0.

    Args:
        codegen: CodeGen instance, or one per expression. The first kernel (k0)
            is the cross-schedule correctness reference.
        exprs: List of signal expressions corresponding to each kernel
        include_correctness_check: If True, adds correctness checking code
        correctness_tolerance: Tolerance for floating point comparisons in correctness checking
        input_size: Size of input data for benchmarking
        warmup_iterations: Number of warmup iterations before timing
        benchmark_iterations: Number of iterations for timing
        compiler: Compiler to use (default: clang++)
        compiler_flags: Additional compiler flags (default: ["-std=c++20", "-O3", "-march=native", "-I", "."])
        row_size: Number of elements per row (used to call reset_and_next_row at row boundaries)
        schedules_per_runner: Maximum schedules per runner (k0 always included)
        parallel_compile: If True, compile batches in parallel. If False, compile
            sequentially. If None (default), reads env var VECREC_PARALLEL_COMPILE
            (set to "0" to disable); defaults to parallel. Sequential mode is
            useful when many benchmarks already run concurrently (e.g. pytest -n),
            to avoid oversubscribing the CPU.

    Returns:
        A dictionary containing:
            - 'output': dict mapping kernel name -> throughput, plus optional 'validation'
            - 'error': stderr if any errors occurred
            - 'return_code': return code from benchmark execution
    """
    compiler_flags = ["-std=c++20", "-O3", "-march=native", "-I", "."] + (compiler_flags or [])

    assert len(exprs) > 0, "At least one expression must be provided"
    assert schedules_per_runner >= 1, "schedules_per_runner must be at least 1"

    n_exprs = len(exprs)
    if isinstance(codegen, CodeGen):
        codegens: List[CodeGen] = [codegen] * n_exprs
    else:
        codegens = list(codegen)
        assert len(codegens) == n_exprs, (
            f"codegens length {len(codegens)} != exprs length {n_exprs}"
        )
    kernel_names_global = [f"k{i}" for i in range(n_exprs)]
    codes_global = [
        cg.generate(expr, name) for cg, expr, name in zip(codegens, exprs, kernel_names_global)
    ]

    # Build batches. Each batch holds at most ``schedules_per_runner`` kernels,
    # and always includes k0 so its measured throughput can anchor normalization.
    batches: List[List[int]] = []
    if n_exprs <= schedules_per_runner:
        batches.append(list(range(n_exprs)))
    else:
        # First batch already contains k0 by virtue of starting at 0.
        batches.append(list(range(schedules_per_runner)))
        new_per_batch = schedules_per_runner - 1
        i = schedules_per_runner
        while i < n_exprs:
            batches.append([0] + list(range(i, min(i + new_per_batch, n_exprs))))
            i += new_per_batch

    # Per-batch artifact paths.
    batch_paths: List[Tuple[str, str, str]] = []
    for b_idx in range(len(batches)):
        fd, h_path = tempfile.mkstemp(suffix=".h", prefix=f"header_b{b_idx}_", dir="/tmp")
        os.close(fd)
        fd, b_path = tempfile.mkstemp(suffix=".cpp", prefix=f"benchmark_b{b_idx}_", dir="/tmp")
        os.close(fd)
        fd, e_path = tempfile.mkstemp(prefix=f"benchmark_b{b_idx}_", dir="/tmp")
        os.close(fd)
        print(h_path)
        print(b_path)
        print(e_path)
        batch_paths.append((h_path, b_path, e_path))

    # Generate header + benchmark .cpp for every batch.
    for b_idx, batch in enumerate(batches):
        h_path, b_path, _e_path = batch_paths[b_idx]
        instantiate_kernels(h_path, [codes_global[i] for i in batch])
        generate_benchmark(
            # collect_input_variables doesn't depend on per-instance codegen state,
            # so picking the codegen of the batch's first kernel is fine.
            codegen=codegens[batch[0]],
            exprs=[exprs[i] for i in batch],
            kernel_names=[kernel_names_global[i] for i in batch],
            kernel_header_path=h_path,
            output_path=b_path,
            include_correctness_check=include_correctness_check,
            correctness_tolerance=correctness_tolerance,
            input_size=input_size,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            row_size=row_size,
            ele_per_iter_list=[codes_global[i].element_per_iter for i in batch],
            rows_per_iter_list=[codes_global[i].rows_per_iter for i in batch],
            computed_signals_list=[codes_global[i].computed_signals for i in batch],
        )

    # Compile every batch — in parallel by default, sequentially when requested.
    def _compile(idx: int) -> Tuple[int, subprocess.CompletedProcess]:
        h_path, b_path, e_path = batch_paths[idx]
        cmd = [compiler] + compiler_flags + [b_path, "-o", e_path]
        return idx, subprocess.run(cmd, capture_output=True, text=True)

    if parallel_compile is None:
        parallel_compile = os.environ.get("VECREC_PARALLEL_COMPILE", "1") != "0"

    compile_results: List[Optional[subprocess.CompletedProcess]] = [None] * len(batches)
    if parallel_compile:
        max_workers = min(len(batches), os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_compile, i) for i in range(len(batches))]
            for fut in concurrent.futures.as_completed(futures):
                idx, res = fut.result()
                compile_results[idx] = res
    else:
        for i in range(len(batches)):
            idx, res = _compile(i)
            compile_results[idx] = res

    for idx, maybe_res in enumerate(compile_results):
        assert maybe_res is not None
        if maybe_res.returncode != 0:
            raise RuntimeError(f"Benchmark execution failed: Compilation failed for batch {idx}: {maybe_res.stderr}")

    # Run runners sequentially so concurrent execution doesn't skew timing.
    parsed_outputs: List[Dict[str, Any]] = []
    parsed_validations: List[Dict[str, Any]] = []
    for idx, batch in enumerate(batches):
        _h_path, _b_path, e_path = batch_paths[idx]
        run_result = subprocess.run([os.path.abspath(e_path)], capture_output=True, text=True)
        if run_result.returncode != 0:
            raise RuntimeError(f"Benchmark execution failed for batch {idx}: {run_result.stderr}")
        parsed = json.loads(run_result.stdout)
        validation = parsed.pop("validation", None)
        parsed_outputs.append(parsed)
        parsed_validations.append(validation if validation is not None else {})

    def _kernel_throughput_mbs(value: Any) -> float:
        if isinstance(value, dict):
            return float(value["throughput_mbs"])
        return float(value)

    def _kernel_samples_seconds(value: Any) -> List[float]:
        if not isinstance(value, dict):
            return []
        return [float(s) / 1e6 for s in value.get("samples_us", [])]

    # Normalize: anchor every batch's reported throughputs to batch 0's k0,
    # so that all batches agree on the same k0 number and the rest scale with it.
    anchor_k0 = _kernel_throughput_mbs(parsed_outputs[0]["k0"])
    aggregated: Dict[str, float] = {"k0": anchor_k0}
    aggregated_timing: Dict[str, Dict[str, Any]] = {}
    aggregated_validation: Dict[str, Any] = {}
    for parsed, validation in zip(parsed_outputs, parsed_validations):
        local_k0 = _kernel_throughput_mbs(parsed["k0"])
        scale = (anchor_k0 / local_k0) if local_k0 != 0.0 else 1.0
        for kname, value in parsed.items():
            if kname == "k0":
                samples = _kernel_samples_seconds(value)
                if samples:
                    aggregated_timing.setdefault(kname, {"samples": samples})
                continue
            aggregated[kname] = _kernel_throughput_mbs(value) * scale
            samples = _kernel_samples_seconds(value)
            if samples:
                aggregated_timing[kname] = {"samples": samples}
        for kname, val in validation.items():
            aggregated_validation.setdefault(kname, val)

    # k0 timing comes from batch 0 only (anchor batch).
    k0_samples = _kernel_samples_seconds(parsed_outputs[0]["k0"])
    if k0_samples:
        aggregated_timing["k0"] = {"samples": k0_samples}

    aggregated_sorted: Dict[str, Any] = dict(sorted(aggregated.items(), key=lambda item: item[1]))
    if aggregated_validation:
        aggregated_sorted["validation"] = aggregated_validation

    return {
        "output": aggregated_sorted,
        "timing": aggregated_timing,
        "error": "",
        "return_code": 0,
    }


def sweep_codegen_setup(
    exprs: Sequence[Union[SignalExpr, SignalExpr2D]],
    ele_per_iter_multiples: Sequence[int] = (1, 2, 4),
    rows_per_iter_options: Sequence[int] = (1, 2, 4),
    compiler: str = "clang++",
    compiler_flags: Optional[List[str]] = None,
    row_size: int = 1024,
    input_size: int = 1 << 20,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    schedules_per_runner: int = 5,
    parallel_compile: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Sweep ``CodeGen.ele_per_iter_multiple`` and ``rows_per_iter`` per schedule.

    For each input expression we time a grid of ``(epi, rpi)`` configurations
    using the same ``generate_and_run_benchmark`` machinery as the headline
    benchmark numbers — so the swept throughputs are directly comparable to the
    default-codegen number. Within each expression's grid, the ``(2, 1)`` cell
    is placed first and used as the in-C++ cross-schedule correctness
    reference (k0); any other cell whose output diverges from k0 is dropped
    and cannot win the "best" selection.

    Returns:
        A list (one entry per input expression) of dicts with keys:
          - "best_ele_per_iter_multiple": int
          - "best_rows_per_iter": int
          - "best_throughput_mbs": float (``0`` if every combination failed)
          - "throughputs_mbs": dict[(epi, rpi)] -> Optional[float] MB/s
          - "validations": dict[(epi, rpi)] -> True | int (mismatch idx) | None
    """
    assert len(exprs) > 0, "At least one expression must be provided"

    # Per-expression, decide which rpi values are even worth trying. A "truly
    # 1D" expression (one with any 1D Var leaf) does not support row blocking:
    # Signal1D has no per-row offset, so rpi > 1 would silently produce wrong
    # output and CodeGen.generate refuses to emit it. Skip those combos so
    # they don't pollute the result with FAIL cells.
    rpi_options_per_expr: List[Sequence[int]] = []
    for expr in exprs:
        try:
            vars_1d_probe, _ = CodeGen().collect_input_variables(expr)
        except Exception:
            vars_1d_probe = {}
        rpi_options_per_expr.append(
            rows_per_iter_options if len(vars_1d_probe) == 0 else (1,)
        )

    # k0 is the (epi=2, rpi=1) cell — the default CodeGen() — so it carries
    # the same configuration the rest of the pipeline already verifies as
    # correct. Build the ordered cell list per expression with k0 first.
    def _ordered_cells(expr_rpis: Sequence[int]) -> List[Tuple[int, int]]:
        k0 = (2, 1)
        rest = [
            (epi, rpi)
            for epi in ele_per_iter_multiples
            for rpi in expr_rpis
            if (epi, rpi) != k0
        ]
        return [k0] + rest if 1 in expr_rpis else rest

    results: List[Dict[str, Any]] = []
    for expr_idx, expr in enumerate(exprs):
        expr_rpis = rpi_options_per_expr[expr_idx]
        cells = _ordered_cells(expr_rpis)

        # Build a CodeGen per cell. Skip cells whose codegen errors (e.g.,
        # rpi-vs-1D-Var check inside CodeGen.generate).
        good_cells: List[Tuple[int, int]] = []
        good_codegens: List[CodeGen] = []
        for epi, rpi in cells:
            try:
                cg = CodeGen(ele_per_iter_multiple=epi, rows_per_iter=rpi)
                cg.generate(expr, "k_probe")  # cheap test compile of the AST
            except Exception:
                continue
            # Re-instantiate so each cell has a fresh codegen (`generate` is
            # stateful — it mutates internal counters/mappings).
            good_codegens.append(CodeGen(ele_per_iter_multiple=epi, rows_per_iter=rpi))
            good_cells.append((epi, rpi))

        throughputs: Dict[Tuple[int, int], Optional[float]] = {
            cell: None for cell in cells
        }
        validations: Dict[Tuple[int, int], Any] = {cell: None for cell in cells}

        if not good_cells:
            results.append(
                {
                    "best_ele_per_iter_multiple": ele_per_iter_multiples[0],
                    "best_rows_per_iter": expr_rpis[0],
                    "best_throughput_mbs": 0.0,
                    "throughputs_mbs": throughputs,
                    "validations": validations,
                }
            )
            continue

        # Run the bundled benchmark. The first cell (k0) acts as the in-C++
        # cross-schedule reference. include_correctness_check=True emits the
        # validation pairs we need to enforce strict matching below.
        try:
            raw = generate_and_run_benchmark(
                codegen=good_codegens,
                exprs=[expr] * len(good_codegens),
                include_correctness_check=True,
                input_size=input_size,
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
                compiler=compiler,
                compiler_flags=compiler_flags,
                row_size=row_size,
                schedules_per_runner=schedules_per_runner,
                parallel_compile=parallel_compile,
            )
        except Exception as exc:
            warnings.warn(
                f"sweep_codegen_setup: bundled benchmark failed for expression "
                f"{expr_idx}: {exc}. All sweep cells for this expression will be "
                "marked as failed."
            )
            results.append(
                {
                    "best_ele_per_iter_multiple": ele_per_iter_multiples[0],
                    "best_rows_per_iter": expr_rpis[0],
                    "best_throughput_mbs": 0.0,
                    "throughputs_mbs": throughputs,
                    "validations": validations,
                }
            )
            continue

        output = raw.get("output", {})
        cell_validations = output.get("validation", {}) if isinstance(output, dict) else {}
        for i, cell in enumerate(good_cells):
            kname = f"k{i}"
            thr = output.get(kname)
            # k0 is the reference — never has its own validation entry.
            if i == 0:
                val: Any = True
            else:
                val = cell_validations.get(kname)
            validations[cell] = val
            if thr is None:
                continue
            # Strict: any cell whose output disagrees with k0 (val is not True)
            # is dropped from the throughput map and cannot win.
            if i != 0 and val is not True:
                warnings.warn(
                    f"sweep_codegen_setup: cell {cell} for expression {expr_idx} "
                    f"diverged from k0 at index {val}; dropping its throughput."
                )
                continue
            throughputs[cell] = float(thr)

        finite = [(k, v) for k, v in throughputs.items() if v is not None]
        if finite:
            best_key, best_mbs_val = max(finite, key=lambda kv: kv[1])
            best_epi, best_rpi = best_key
            best_throughput = float(best_mbs_val)
        else:
            best_epi, best_rpi = ele_per_iter_multiples[0], expr_rpis[0]
            best_throughput = 0.0

        results.append(
            {
                "best_ele_per_iter_multiple": best_epi,
                "best_rows_per_iter": best_rpi,
                "best_throughput_mbs": best_throughput,
                "throughputs_mbs": throughputs,
                "validations": validations,
            }
        )

    return results


def find_best_schedule_and_emit_header(
    expr: Union[SignalExpr, SignalExpr2D],
    header_path: str,
    kernel_name: str,
    baseline_kernel_name: Optional[str] = None,
    max_bits: Optional[int] = None,
    ele_per_iter_multiples: Sequence[int] = (1, 2, 4),
    rows_per_iter_options: Sequence[int] = (1, 2, 4),
    debug: bool = False,
) -> Dict[str, Any]:
    """Find the best schedule + codegen config for ``expr`` and emit a single
    self-contained C++ header for it at ``header_path`` with struct ``kernel_name``.

    This mirrors the ranking logic in ``benchmarks/benchmark.py::Benchmark.run()``:
    enumerate candidate schedules, time them all (with a cross-schedule
    correctness check), sweep ``(ele_per_iter_multiple, rows_per_iter)`` over the
    top ~10% (always including the reference schedule k0), and pick the
    highest-throughput ``(schedule, epi, rpi)``. The winning kernel is written as
    a standalone header via :func:`instantiate_kernels` (which inlines
    ``templates/common.h``), so the result compiles with only a C++ compiler.

    It also emits a *baseline* kernel: the reference (un-optimized) schedule
    ``schedules[0]`` — the minimal-lowering branch with no dilation/factorization —
    with its own best ``(epi, rpi)`` found by the same sweep. This is the baseline
    ``vec-rec/benchmarks/*`` always reports; it is a stronger comparison point than a
    naive serial loop because it still benefits from codegen's register tiling and
    unrolling, isolating the gain attributable to the optimization itself. The baseline
    struct is written into the same header as ``baseline_kernel_name`` (default
    ``<kernel_name>_baseline``).

    Returns a dict with the best schedule's ``kernel_name`` / ``best_*`` / ``ele_per_iter``
    / ``rows_per_iter`` and the baseline's ``baseline_kernel_name`` / ``baseline_*``, plus
    ``header_path`` and ``num_schedules``. It also returns the chosen schedule ASTs as
    ``best_expr`` / ``baseline_expr`` (VecRec IR objects, not JSON-serializable).
    """
    # Imported lazily to avoid a circular import (optimize -> codegen).
    import math
    from vecrec.optimize import get_candidate_schedules
    from vecrec.util import get_native_bits

    if max_bits is None:
        max_bits = get_native_bits()

    schedules = get_candidate_schedules(expr, max_bits, debug=debug)
    assert len(schedules) > 0, "No candidate schedules were generated"

    # Time every schedule with the default codegen, cross-checking correctness.
    raw = generate_and_run_benchmark(CodeGen(), schedules, include_correctness_check=True)
    output = raw["output"]
    timing_keys = [k for k in output if k != "validation"]

    # Sweep the top ~10% schedules (always include the reference schedule k0),
    # but never more than MAX_SWEEP_SCHEDULES non-baseline schedules.
    n = len(schedules)
    n_top = min(n, max(2, math.ceil(0.1 * n)))
    sorted_keys = sorted(timing_keys, key=lambda k: output[k], reverse=True)
    top_indices = [int(k[1:]) for k in sorted_keys[:n_top]]
    if 0 not in top_indices:
        top_indices.append(0)
    # Cap non-baseline schedules at MAX_SWEEP_SCHEDULES, keeping the
    # highest-throughput ones (top_indices is throughput-sorted). Schedule 0
    # (the baseline) is always retained regardless of the cap.
    non_baseline = [i for i in top_indices if i != 0]
    if len(non_baseline) > MAX_SWEEP_SCHEDULES:
        kept = set(non_baseline[:MAX_SWEEP_SCHEDULES]) | {0}
        top_indices = [i for i in top_indices if i in kept]
    top_exprs = [schedules[i] for i in top_indices]

    sweep_results = sweep_codegen_setup(
        top_exprs,
        ele_per_iter_multiples=ele_per_iter_multiples,
        rows_per_iter_options=rows_per_iter_options,
    )
    best_local_idx, best_result = max(
        enumerate(sweep_results),
        key=lambda kv: kv[1]["best_throughput_mbs"],
    )
    best_idx = top_indices[best_local_idx]
    best_expr = schedules[best_idx]
    best_epi = best_result["best_ele_per_iter_multiple"]
    best_rpi = best_result["best_rows_per_iter"]

    # Baseline = reference schedule (schedules[0]) with its own best (epi, rpi).
    # It is always present in top_indices, so the sweep already measured it.
    if baseline_kernel_name is None:
        baseline_kernel_name = kernel_name + "_baseline"
    base_local_idx = top_indices.index(0)
    base_result = sweep_results[base_local_idx]
    base_epi = base_result["best_ele_per_iter_multiple"]
    base_rpi = base_result["best_rows_per_iter"]
    base_expr = schedules[0]

    # Emit both kernels into one self-contained header.
    best_kp = CodeGen(ele_per_iter_multiple=best_epi, rows_per_iter=best_rpi).generate(best_expr, kernel_name)
    base_kp = CodeGen(ele_per_iter_multiple=base_epi, rows_per_iter=base_rpi).generate(base_expr, baseline_kernel_name)
    instantiate_kernels(header_path, [best_kp, base_kp])

    return {
        "kernel_name": kernel_name,
        "baseline_kernel_name": baseline_kernel_name,
        "header_path": header_path,
        # The chosen schedule ASTs (VecRec IR). Not JSON-serializable; callers that
        # persist the metadata should render them (e.g. via ``vecrec.expr.pp``) first.
        "best_expr": best_expr,
        "baseline_expr": base_expr,
        "best_ele_per_iter_multiple": best_epi,
        "best_rows_per_iter": best_rpi,
        "best_throughput_mbs": float(best_result["best_throughput_mbs"]),
        "ele_per_iter": best_kp.element_per_iter,
        "rows_per_iter": best_kp.rows_per_iter,
        "baseline_ele_per_iter_multiple": base_epi,
        "baseline_rows_per_iter": base_rpi,
        "baseline_throughput_mbs": float(base_result["best_throughput_mbs"]),
        "baseline_ele_per_iter": base_kp.element_per_iter,
        "baseline_is_best": best_idx == 0,
        "num_schedules": n,
    }
