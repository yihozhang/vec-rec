from typing import List, Optional, Dict, Set, Tuple, cast
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D, SignalVectorExpr, KernelMatrixExpr, Type
from vecrec.expr.signal import RVar2D, Var2D, ComputedSignal2D
from vecrec.expr.signal_ops import (
    Repeater,
    Ith,
    Signal2DAt,
    RepeaterVec,
    WindowMap,
    WindowMapVec,
    WMVar,
    ChangeSemiring,
    CustomPointwiseOp,
)
from vecrec.expr.kernel import TIKernel, TVKernel
from vecrec.expr.kernel_matrix import KernelMatrix
from vecrec.expr.signal_vec import SignalVector, ConvolveV, RecurseV, ConvertLanesV, VectorAt
from vecrec.semiring import CustomSemiring
from vecrec.transform import Transform
from vecrec.util import ElementType
from importlib.resources import files

TEMPLATE = files("vecrec.templates") / "common.h"


class ComputedSignalDecl:
    """Descriptor for a ComputedSignal2D lowered into a kernel. The signal struct is
    constructed in driver scope (from `setup_code`'s buffers) and passed into the
    kernel constructor by value via the `param_name` parameter."""

    param_name: str
    resolved_type: str
    prelude: str
    setup_code: str
    make_args: str

    def __init__(self, param_name: str, resolved_type: str, prelude: str, setup_code: str, make_args: str):
        self.param_name = param_name
        self.resolved_type = resolved_type
        self.prelude = prelude
        self.setup_code = setup_code
        self.make_args = make_args


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


class KernelProgram:
    text: str
    element_per_iter: int
    rows_per_iter: int
    custom_semirings: Set[CustomSemiring]
    computed_signals: List[ComputedSignalDecl]
    # C++ struct definitions for CustomPointwiseOp operators used by this kernel.
    # Emitted (deduplicated) into the header before the kernel structs.
    custom_op_preludes: List[str]

    def __init__(
        self,
        text: str,
        element_per_iter: int,
        custom_semirings: Optional[Set[CustomSemiring]] = None,
        rows_per_iter: int = 1,
        computed_signals: Optional[List[ComputedSignalDecl]] = None,
        custom_op_preludes: Optional[List[str]] = None,
    ):
        assert rows_per_iter >= 1, "rows_per_iter must be >= 1"
        self.text = text
        self.element_per_iter = element_per_iter
        self.rows_per_iter = rows_per_iter
        self.custom_semirings = custom_semirings if custom_semirings is not None else set()
        self.computed_signals = computed_signals if computed_signals is not None else []
        self.custom_op_preludes = custom_op_preludes if custom_op_preludes is not None else []


class TopoSort(Transform):
    nodes: List[RecLang]
    visited: Set[RecLang]

    def __init__(self):
        self.nodes = []
        self.visited = set()
        self.apply_kernel = self.apply_signal = self.apply_signal2d = self.apply_kernel2d = self.apply_generic  # type: ignore

    def apply_generic(self, expr):
        if expr in self.visited:
            return []
        self.visited.add(expr)
        for child in expr.children():
            self.apply_generic(child)
        self.nodes.append(expr)
        return [expr]


class Node:
    class_name: str
    var_name: str
    constructor_args: str
    run_args_list: List[List[str]]
    input_type: str
    output_type: str
    is_pointer: bool

    def __init__(
        self,
        class_name: str,
        var_name: str,
        constructor_args: str,
        run_args_list: List[List[str]],
        input_type: str,
        output_type: str,
        is_pointer: bool = False,
    ):
        self.class_name = class_name
        self.var_name = var_name
        self.constructor_args = constructor_args
        self.run_args_list = run_args_list
        self.input_type = input_type
        self.output_type = output_type
        self.is_pointer = is_pointer


class CodeGen:
    """ele_per_iter determines the number of elements to be processed within each element."""

    ele_per_iter: int
    # Block size: kernel.run() emits `rows_per_iter` rows per call, and operators
    # (other than Repeater/RepeaterVec) are replicated per block-row. Always >= 1.
    rows_per_iter: int
    # mapping from a node to its register(s) and code information.
    # Cache key includes block_row so different replicas memoise independently.
    mapping: Dict[Tuple[RecLang, int, int], Tuple[List[int], Code]]
    # All the vector types used in the generated code. Useful for determining which registers to declare.
    vec_types: Set[str]
    # Map RVar2D names to their context variables and signal2d n_rows (= owning Repeater n_rows - 1).
    var2d_context: Dict[str, Tuple[str, int]]
    # Map RVar2D names to per-block-row inner-register lists (one entry per slice).
    # Populated when the enclosing Repeater/RepeaterVec is codegened for that block_row;
    # consumed when Signal2D (RVar2D case) emits register-reuse Mov overrides.
    rvar_inners: Dict[str, Dict[int, List[int]]]
    # Map vector types to their current register count, used for generating new registers
    reg_counter: Dict[str, int]
    var_counter: int

    # The list of AST declaration
    # (classname, variable_name, constructor_args)
    program: List[Node]
    # The lsit of constant declaration
    prologue: List[str]

    rvec_comp_regs: Dict[Tuple[RepeaterVec, int], List[List[int]]]
    # Map WMVar names to their per-slice register indices (set during _generate_windowmap)
    _wm_var_map: Dict[str, List[int]]
    # Active block_row for the current generate_signal recursion. Set by callers
    # (generate() loops over block-rows; Repeater/RepeaterVec also set this when
    # generating their bodies). Read by generate_signal to key the cache.
    current_block_row: int
    # Shared RepeaterContext per Repeater AST node (id-keyed): created once and
    # reused across all block-rows so they share the same ring buffer.
    shared_repeater_contexts: Dict[int, str]
    # Shared per-component RepeaterContext per RepeaterVec AST node (id-keyed).
    shared_repeatervec_contexts: Dict[int, List[str]]

    def __init__(self, ele_per_iter_multiple: int = 2, rows_per_iter: Optional[int] = None):
        assert ele_per_iter_multiple >= 1, "ele_per_iter_multiple must be >= 1"
        if rows_per_iter is not None:
            assert rows_per_iter >= 1, "rows_per_iter must be >= 1"
        self._ele_per_iter_multiple = ele_per_iter_multiple
        self._rows_per_iter_override = rows_per_iter
        self.current_block_row = 0
        self.reg_counter = {}
        self.var_counter = 0
        self.vec_types = set()
        self.mapping = {}
        self.program = []
        self.prologue = []
        self.var2d_context = {}
        self.rvar_inners = {}
        self.shared_repeater_contexts = {}
        self.shared_repeatervec_contexts = {}
        self.rvec_comp_regs = {}
        self._wm_var_map = {}
        self.custom_semirings: Set[CustomSemiring] = set()
        self.computed_signals: List[ComputedSignalDecl] = []
        self.custom_op_preludes: List[str] = []

    def clear(self):
        self.current_block_row = 0
        self.reg_counter = {}
        self.var_counter = 0
        self.vec_types = set()
        self.mapping = {}
        self.program = []
        self.prologue = []
        self.var2d_context = {}
        self.rvar_inners = {}
        self.shared_repeater_contexts = {}
        self.shared_repeatervec_contexts = {}
        self.rvec_comp_regs = {}
        self._wm_var_map = {}
        self.custom_semirings = set()
        self.computed_signals = []
        self.custom_op_preludes = []

    def _max_lanes_in_tree(self, expr: RecLang, visited: Optional[set] = None) -> int:
        """Find the maximum lane width across all nodes in the expression tree."""
        if visited is None:
            visited = set()
        if id(expr) in visited:
            return 0
        visited.add(id(expr))
        # WindowMap outputs 2 K-lane vectors per iteration, so report 2*lanes
        if isinstance(expr, WindowMap):
            assert expr.lanes is not None
            return 2 * expr.lanes
        # WindowMapVec outputs n_interleave K-lane vectors per iteration
        if isinstance(expr, WindowMapVec):
            assert expr.lanes is not None
            return expr.n_interleave * expr.lanes
        max_l = expr.lanes or 0
        for child in expr.children():
            max_l = max(max_l, self._max_lanes_in_tree(child, visited))
        return max_l

    def generate(self, expr: SignalExpr | SignalExpr2D, name: str) -> KernelProgram:

        self.clear()
        assert expr.lanes is not None
        # ele_per_iter must be divisible by every lane width in the tree.
        # Sub-expressions (e.g. Var inside TVKernel) may have wider lanes than the
        # top-level expression, so we scan the whole tree for the maximum.
        # The multiple controls how many lane-widths we pack per iteration; the
        # default of 2 matches WindowMap's 2K-per-run output.
        self.ele_per_iter = self._max_lanes_in_tree(expr) * self._ele_per_iter_multiple

        # Decide rows_per_iter (block size). User override > default.
        # Default is 1 (single-row blocks). Row blocking (rpi > 1) is an opt-in
        # optimization the sweep tries explicitly for expressions whose inputs
        # are all 2D; we no longer auto-enable it based on Repeater presence
        # alone, because a Repeater with a 1D Var leaf is also unsafe at rpi>1.
        if self._rows_per_iter_override is not None:
            self.rows_per_iter = self._rows_per_iter_override
        else:
            self.rows_per_iter = 1

        # Row blocking (rpi > 1) with any 1D Var input is unsafe — even inside
        # a Repeater. The Signal1D reader has no block-row offset and its
        # reset_and_next_row is a no-op, so the k replicas all start at &data[0]
        # and advance identically. Each pair of consecutive logical rows ends
        # up reading the same physical chunk of data, producing silently wrong
        # output. Use Var2D (which lowers to the block-row-aware InputSignal2D)
        # if you need per-row inputs at rpi > 1.
        if self.rows_per_iter > 1:
            vars_1d_check, _ = self.collect_input_variables(expr)
            if vars_1d_check:
                raise ValueError(
                    f"rows_per_iter={self.rows_per_iter} with 1D Var inputs "
                    f"{sorted(vars_1d_check.keys())} is unsafe: Signal1D is not "
                    "block-row aware, so the k replicas would re-read the same "
                    "offsets and silently produce wrong output. Convert the 1D Vars "
                    "to Var2D, or use rows_per_iter=1."
                )
            # ComputedSignal2D is block-row aware (like InputSignal2D): codegen replicates
            # it per block-row, baking block_row into the resolved type, and the driver
            # constructs one signal object per block-row. So rpi > 1 is supported.

        n_block_rows = self.rows_per_iter
        per_row_results: List[Tuple[List[int], Code]] = []
        for br in range(n_block_rows):
            self.current_block_row = br
            per_row_results.append(self.generate_signal(expr))

        vars, vars2d = self.collect_input_variables(expr)
        # ComputedSignal2D objects that read external buffers are constructed in driver
        # scope and passed in by value, preceding the raw input pointers. Buffer-less
        # signals (empty make_args) are default-constructed as members, not parameters.
        args_computed = [
            f"{d.resolved_type} {d.param_name}" for d in self.computed_signals if d.make_args.strip()
        ]
        args_1d = [f"const {elt_type.to_str()} *{var}" for var, elt_type in sorted(vars.items())]
        args_2d = [f"const {elt_type.to_str()} *{var}, int {var}_stride" for var, elt_type in sorted(vars2d.items())]
        all_args = args_computed + args_1d + args_2d + ["int row_size"]
        args = ", ".join(all_args)
        new_line = "\n            "
        assert expr.lanes is not None
        result_vec_type = self.get_vec_type(expr.element_type, expr.lanes)

        def generate_run_calls(x: Node) -> str:
            reg_prefix = (
                f"reg_{x.output_type}" if x.output_type == x.input_type else f"reg_{x.input_type}, reg_{x.output_type}"
            )
            return new_line.join(
                f"{x.var_name}{connector}run({reg_prefix}{sep} {', '.join(run_args)});"
                for run_args in x.run_args_list
                if (sep := ", " if len(run_args) > 0 else " ")  # " " because "" is falsy
                if (connector := "->" if x.is_pointer else ".")
            )

        # Per-block-row output regs (each per_row_results[br][0] is a list of slice regs).
        n_out_slices = len(per_row_results[0][0])

        # kernel.run(T** out_per_row) writes ele_per_iter elements per block-row
        # into out_per_row[br].
        run_signature = f"void run({result_vec_type} **out_per_row)"
        memcpy_block = []
        for br in range(n_block_rows):
            for i in range(n_out_slices):
                memcpy_block.append(
                    f"memcpy(out_per_row[{br}] + {i}, reg_{result_vec_type}.data + "
                    f"{per_row_results[br][0][i]}, sizeof({result_vec_type}));"
                )
        memcpy_lines = new_line.join(memcpy_block)

        # The per-run "register file" is declared as a local inside run() rather than
        # as a struct member. The kernel object is constructed once and mutated across
        # the driver's inner loop (it carries recurrence state in the operator members),
        # so a member register file would be treated as loop-carried: every slot stays
        # live across the loop back-edge and, for large kernels, gets spilled to the
        # stack instead of promoted by SROA/mem2reg. A run()-local gets lifetime markers
        # (run() is always-inlined), so the compiler knows the scratch is dead between
        # iterations and keeps it in registers. The values are pure per-run scratch:
        # persistent state lives in the operator objects, not here.
        reg_decls = new_line.join(
            map(lambda vt: f"Reg<{vt}, {self.reg_counter[vt]}> reg_{vt};", self.vec_types)
        )
        text = f"""
        struct {name} {{
            // harness metadata: lets an external C++ benchmark harness drive this
            // kernel generically (read K::ele_per_iter / K::rows_per_iter / K::vector_type)
            // without a per-kernel calling template.
            static constexpr int ele_per_iter = {self.ele_per_iter};
            static constexpr int rows_per_iter = {self.rows_per_iter};
            using vector_type = {result_vec_type};
            // static array declarations
            {new_line.join(self.prologue)}
            // node declarations
            {new_line.join(map(lambda x: f"{x.class_name} {x.var_name};", self.program))}
            inline __attribute__((always_inline)) {name}({args}) :
              {", ".join(map(lambda x: f"{x.var_name}({x.constructor_args})", self.program))}
              {{}}

            inline __attribute__((always_inline)) {run_signature} {{
                // register declaration (per-run scratch; local so SROA can promote it)
                {reg_decls}
                {new_line.join(map(generate_run_calls, self.program))}
                {memcpy_lines}
            }}

            inline __attribute__((always_inline)) void reset_and_next_row() {{
                {new_line.join([f"{x.var_name}{connector}reset_and_next_row();" for x in self.program if (connector := "->" if x.is_pointer else ".")])}
            }}
        }};

        using {name}_vector_type = {result_vec_type};
        """

        # auto make_{name}({args}) {{
        #     return {name}({", ".join(list(vars.keys()) + list(vars2d.keys()) + ["row_size"])});
        # }}
        # """
        return KernelProgram(
            text,
            self.ele_per_iter,
            set(self.custom_semirings),
            rows_per_iter=self.rows_per_iter,
            computed_signals=list(self.computed_signals),
            custom_op_preludes=list(self.custom_op_preludes),
        )

    def _generate_windowmap(self, wm: WindowMap) -> List[int]:
        assert wm.lanes is not None
        K = wm.lanes
        vec_type = self.get_vec_type(wm.element_type, K)
        self.vec_types.add(vec_type)
        saved_ele = self.ele_per_iter

        # Number of body iterations per outer iteration.
        # Each body iteration consumes and produces 2K elements (one Unzip2 + one Zip2).
        assert (
            saved_ele % (2 * K) == 0
        ), f"WindowMap (lanes={K}): ele_per_iter={saved_ele} is not a multiple of 2*K={2*K}"
        n_body_slices = saved_ele // (2 * K)

        # Generate each input at n_body_slices * 2K rate
        self.ele_per_iter = n_body_slices * 2 * K
        all_input_regs = [self.generate_signal(inp)[0] for inp in wm.inputs]

        # For each input, run n_body_slices Unzip2 calls; collect per-slice even/odd registers.
        for inp_regs, ivars in zip(all_input_regs, wm.input_vars):
            even_regs: List[int] = []
            odd_regs: List[int] = []
            for s in range(n_body_slices):
                even_reg = self._new_register(vec_type)
                odd_reg = self._new_register(vec_type)
                self._declare_node(
                    f"Unzip2<{vec_type}>",
                    "",
                    [[even_reg, odd_reg, inp_regs[2 * s], inp_regs[2 * s + 1]]],
                    vec_type,
                    vec_type,
                )
                even_regs.append(even_reg)
                odd_regs.append(odd_reg)
            self._wm_var_map[ivars[0].name] = even_regs
            self._wm_var_map[ivars[1].name] = odd_regs

        # Generate body at n_body_slices * K lanes
        self.ele_per_iter = n_body_slices * K
        body_regs = self.generate_signal(wm.body)[0]

        # Clean up WMVar mappings
        for ivars in wm.input_vars:
            for wm_var in ivars:
                del self._wm_var_map[wm_var.name]

        # For each body slice, zip the two output substreams back to full rate.
        # Component 0 = f_odd, component 1 = f_even (matches current Zip2 argument order).
        output_regs: List[int] = []
        for s in range(n_body_slices):
            out_lo = self._new_register(vec_type)
            out_hi = self._new_register(vec_type)
            self._declare_node(
                f"Zip2<{vec_type}>",
                "",
                [[out_lo, out_hi, body_regs[s] + 1, body_regs[s]]],
                vec_type,
                vec_type,
            )
            output_regs.extend([out_lo, out_hi])

        self.ele_per_iter = saved_ele
        return output_regs

    def _generate_windowmapvec(self, wmv: WindowMapVec) -> List[int]:
        assert wmv.lanes is not None
        assert wmv.n_interleave == 2, "Only n_interleave=2 (even/odd) is currently supported"
        K = wmv.lanes
        n_interleave = wmv.n_interleave
        n_out = wmv.n
        vec_type = self.get_vec_type(wmv.element_type, K)
        self.vec_types.add(vec_type)
        saved_ele = self.ele_per_iter

        assert (
            saved_ele % (n_interleave * K) == 0
        ), f"WindowMapVec (lanes={K}): ele_per_iter={saved_ele} is not a multiple of n_interleave*K={n_interleave*K}"
        n_body_slices = saved_ele // (n_interleave * K)

        # Generate each input at saved_ele rate (unchanged).
        all_input_regs = [self.generate_signal(inp)[0] for inp in wmv.inputs]

        # For each input, Unzip2 each consecutive pair into even/odd substreams.
        for inp_regs, ivars in zip(all_input_regs, wmv.input_vars):
            even_regs: List[int] = []
            odd_regs: List[int] = []
            for s in range(n_body_slices):
                even_reg = self._new_register(vec_type)
                odd_reg = self._new_register(vec_type)
                self._declare_node(
                    f"Unzip2<{vec_type}>",
                    "",
                    [[even_reg, odd_reg, inp_regs[n_interleave * s], inp_regs[n_interleave * s + 1]]],
                    vec_type,
                    vec_type,
                )
                even_regs.append(even_reg)
                odd_regs.append(odd_reg)
            self._wm_var_map[ivars[0].name] = even_regs
            self._wm_var_map[ivars[1].name] = odd_regs

        # Generate body at n_body_slices * K rate.
        # Body has n_interleave * n_out components; for output i:
        #   even substream = body[bs] + 2*i, odd substream = body[bs] + 2*i+1.
        self.ele_per_iter = n_body_slices * K
        body_regs = self.generate_signal(wmv.body)[0]

        # Clean up WMVar mappings.
        for ivars in wmv.input_vars:
            for wm_var in ivars:
                del self._wm_var_map[wm_var.name]

        # For each body slice, zip the n_out even/odd component pairs back to outer rate.
        # Pre-allocate output register bases: n_slices = 2*n_body_slices, n_out components each.
        n_slices = n_interleave * n_body_slices
        output_regs: List[int] = []
        for s in range(n_body_slices):
            even_base = self._new_register(vec_type, n_out)
            odd_base = self._new_register(vec_type, n_out)
            for i in range(n_out):
                self._declare_node(
                    f"Zip2<{vec_type}>",
                    "",
                    [
                        [
                            even_base + i,
                            odd_base + i,
                            body_regs[s] + n_interleave * i + 1,
                            body_regs[s] + n_interleave * i,
                        ]
                    ],
                    vec_type,
                    vec_type,
                )
            output_regs.extend([even_base, odd_base])

        assert len(output_regs) == n_slices
        self.ele_per_iter = saved_ele
        return output_regs

    def generate_signal(
        self, expr: SignalExpr | SignalExpr2D | KernelExpr | SignalVectorExpr | KernelMatrixExpr
    ) -> Tuple[List[int], Code]:
        # Per-block-row replication: each block_row gets its own emitted operators
        # (its own state), so we key the cache on it. self.current_block_row is set
        # by the caller (generate() and the Repeater/RepeaterVec cases).
        block_row = self.current_block_row
        cache_key = (expr, self.ele_per_iter, block_row)
        if cache_key in self.mapping:
            return self.mapping[cache_key]

        # WindowMap is handled specially: it switches ele_per_iter internally.
        if isinstance(expr, WindowMap):
            regs = self._generate_windowmap(expr)
            result: Tuple[List[int], Code] = (regs, Code("", expr.ty, expr.element_type, expr.lanes))
            self.mapping[cache_key] = result
            return result

        # WindowMapVec is handled specially: it switches ele_per_iter internally.
        if isinstance(expr, WindowMapVec):
            regs = self._generate_windowmapvec(expr)
            wmv_result: Tuple[List[int], Code] = (regs, Code("", expr.ty, expr.element_type, expr.lanes))
            self.mapping[cache_key] = wmv_result
            return wmv_result

        # KernelMatrixExpr has no own registers; just recursively generate each entry.
        if isinstance(expr, KernelMatrixExpr):
            for k in expr.children():
                self.generate_signal(cast(KernelExpr, k))
            placeholder: Tuple[List[int], Code] = ([], Code("", expr.ty, expr.element_type, expr.lanes))
            self.mapping[cache_key] = placeholder
            return placeholder

        assert expr.lanes is not None
        assert self.ele_per_iter % expr.lanes == 0

        vec_type = self.get_vec_type(expr.element_type, expr.lanes)
        self.vec_types.add(vec_type)
        element_type = expr.element_type
        ty = expr.ty

        n_slices = self.ele_per_iter // expr.lanes
        taps: int | None = None
        n_rows: int | None

        if isinstance(expr, SignalExpr):
            match expr:
                case WMVar(name):
                    regs = list(self._wm_var_map[expr.name])
                    assert len(regs) == n_slices, f"WMVar {expr.name!r}: expected {n_slices} slices, got {len(regs)}"
                    # The registers were pre-allocated by _generate_windowmap's Unzip2 nodes;
                    # no new C++ node needs to be declared here.
                    code = ""
                case Num(value):
                    regs = self._new_registers(vec_type, n_slices)
                    code = self._declare_node(
                        f"Signal1DConstant<{vec_type}>",
                        expr.element_type.val_to_str(value, ty),
                        [[r] for r in regs],
                        vec_type,
                        vec_type,
                    )
                case Impulse(value):
                    regs = self._new_registers(vec_type, n_slices)
                    zero_str = expr.element_type.val_to_str(ty.zero(), ty)
                    # Impulse fires exactly once globally at lane 0 of time 0.
                    # With rows_per_iter > 1, each block-row gets its own state, so
                    # only block_row 0 emits the real Signal1DImpulse; block_row > 0
                    # (logically row 1, 2, … within the first block) must emit the
                    # semiring zero so the impulse does not fire again.
                    if block_row == 0:
                        value_str = expr.element_type.val_to_str(value, ty)
                        code = self._declare_node(
                            f"Signal1DImpulse<{vec_type}>",
                            f"{value_str}, {zero_str}",
                            [[r] for r in regs],
                            vec_type,
                            vec_type,
                        )
                    else:
                        code = self._declare_node(
                            f"Signal1DConstant<{vec_type}>",
                            zero_str,
                            [[r] for r in regs],
                            vec_type,
                            vec_type,
                        )
                case Var(name):
                    regs = self._new_registers(vec_type, n_slices)
                    code = self._declare_node(
                        f"Signal1D<{vec_type}>", f"{name}", [[r] for r in regs], vec_type, vec_type
                    )

                case CustomPointwiseOp(a, b):
                    # User-defined pointwise op: lowered like a built-in binop, but the
                    # C++ struct name and definition are benchmark-owned (no semiring
                    # suffix). The struct definition is emitted into the header once.
                    a_regs = self.generate_signal(a)[0]
                    b_regs = self.generate_signal(b)[0]
                    regs = self._new_registers(vec_type, n_slices)
                    if expr.cpp_prelude not in self.custom_op_preludes:
                        self.custom_op_preludes.append(expr.cpp_prelude)
                    code = self._declare_node(
                        f"{expr.cpp_struct}<{vec_type}>",
                        "",
                        [[regs[i], a_regs[i], b_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )

                case SignalExprBinOp(a, b):
                    a_regs = self.generate_signal(a)[0]
                    b_regs = self.generate_signal(b)[0]
                    regs = self._new_registers(vec_type, n_slices)
                    constructor = type(expr).__name__ + self.get_sem_suffix(ty, element_type)
                    code = self._declare_node(
                        f"{constructor}<{vec_type}>",
                        "",
                        [[regs[i], a_regs[i], b_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )
                case SNeg(a):
                    a_regs = self.generate_signal(a)[0]
                    regs = self._new_registers(vec_type, n_slices)
                    code = self._declare_node(
                        f"SNeg<{vec_type}>",
                        "",
                        [[regs[i], a_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )

                case Convolve(a, f):
                    taps, idxs, _ = a.to_sparse_repr()
                    idx_var = self._declare_array("int", idxs)
                    a_regs = self.generate_signal(a)[0]
                    f_regs = self.generate_signal(f)[0]
                    regs = self._new_registers(vec_type, n_slices)
                    constructor = "SConvolve" + self.get_sem_suffix(ty, element_type)
                    code = self._declare_node(
                        f"{constructor}<{vec_type}, {taps}, {idx_var}>",
                        "",
                        [[regs[i], a_regs[i], f_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )
                case SelectMax(values, labels):
                    # Argmax selection: emit one SelectMax<sem> node per slice.
                    # Each candidate value occupies one register; the labels are
                    # a compile-time template array, the candidate source regs a
                    # runtime array (same idiom as Tuple/SignalVector).
                    n_cand = len(values)
                    val_regs = [self.generate_signal(v)[0] for v in values]
                    labels_var = self._declare_array("int", list(labels))
                    regs = self._new_registers(vec_type, n_slices)
                    sem_suffix = self.get_sem_suffix(ty, element_type)
                    run_args_list = []
                    for s in range(n_slices):
                        src_var = self._declare_array("int", [val_regs[j][s] for j in range(n_cand)])
                        run_args_list.append([regs[s], src_var])
                    code = self._declare_node(
                        f"SelectMax{sem_suffix}<{vec_type}, {n_cand}, {labels_var}>",
                        "",
                        run_args_list,
                        vec_type,
                        vec_type,
                    )

                case Recurse(a, g):
                    time_delay, a = a.time_delay(expr.lanes)
                    assert time_delay == a.lanes
                    taps, idxs, _ = a.to_sparse_repr()
                    idx_var = self._declare_array("int", idxs)
                    a_regs = self.generate_signal(a)[0]
                    g_regs = self.generate_signal(g)[0]
                    regs = self._new_registers(vec_type, n_slices)
                    constructor = "SRecurse" + self.get_sem_suffix(ty, element_type)
                    code = self._declare_node(
                        f"{constructor}<{vec_type}, {taps}, {idx_var}>",
                        "",
                        [[regs[i], a_regs[i], g_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )

                case ConvertLanes(a):
                    regs, code = self.enforce_lanes(expr, n_slices)

                case ChangeSemiring(a):
                    # Pure relabeling: the runtime values are identical; the
                    # surrounding operators dispatch on the new ty themselves.
                    regs = self.generate_signal(a)[0]
                    code = ""

                case Ith(g, ith):
                    regs = self._new_registers(vec_type, n_slices)
                    g_regs = self.generate_signal(g)[0]
                    n_rows = self._signal2d_n_rows(g)
                    code = self._declare_node(
                        f"IthRow<{vec_type}, {n_rows}, {ith}>",
                        "",
                        [[regs[i], g_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )

                case VectorAt(g, i):
                    regs = self._new_registers(vec_type, n_slices)
                    g_regs = self.generate_signal(g)[0]
                    code = self._declare_node(
                        f"VectorAt<{vec_type}, {g.n}, {i}>",
                        "",
                        [[regs[i], g_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )

                case _:
                    raise AssertionError(f"CodeGen: unhandled SignalExpr {type(expr)}")

        elif isinstance(expr, KernelExpr):
            coeffs: List[float] | List[SignalExpr]
            if isinstance(expr, TIKernel):
                taps, indices, coeffs = expr.to_sparse_repr()
                index_var = self._declare_array("int", indices)
                value_var = self._declare_array(
                    element_type.to_str(), [f"{expr.element_type.val_to_str(v, expr.ty)}" for v in coeffs]
                )
                regs = self._new_registers(vec_type, n_slices, taps)
                code = self._declare_node(
                    f"TimeInvariantKernel<{taps}, {vec_type}, {index_var}, {value_var}>",
                    "",
                    [[r] for r in regs],
                    vec_type,
                    vec_type,
                )
            elif isinstance(expr, TVKernel):
                taps, indices, coeffs = expr.to_sparse_repr()
                regs = self._new_registers(vec_type, n_slices, taps)
                index_var = self._declare_array("int", indices)
                run_args_list = []
                for i in range(n_slices):
                    signal_regs = [self.generate_signal(coeff)[0][i] for coeff in coeffs]
                    signal_regs_var = self._declare_array("int", signal_regs)
                    run_args_list.append([regs[i], signal_regs_var])
                code = self._declare_node(
                    f"TimeVaryingKernel<{taps}, {vec_type}, {index_var}>", "", run_args_list, vec_type, vec_type
                )
            else:
                assert isinstance(expr, KConvertLanes)
                taps = self.generate_signal(expr.a)[1].taps
                assert taps is not None, "KConvertLanes should have taps information from its child kernel"
                regs, code = self.enforce_lanes(expr, n_slices, taps)

        elif isinstance(expr, SignalExpr2D):
            match expr:
                case Repeater(g, n_rows, prev_rows_var):
                    assert n_rows is not None
                    block_row = self.current_block_row
                    block_size = self._block_size_template()

                    # One RepeaterContext is shared across all block-rows of this
                    # Repeater AST node — created on the first block-row's pass.
                    rep_id = id(expr)
                    if rep_id not in self.shared_repeater_contexts:
                        zero_str = expr.element_type.val_to_str(ty.zero(), ty)
                        self.shared_repeater_contexts[rep_id] = self._declare_repeater_context(
                            vec_type, zero_str, n_rows
                        )
                    repeater_context = self.shared_repeater_contexts[rep_id]

                    # Generate the body for this block-row.
                    self.var2d_context[prev_rows_var.name] = (repeater_context, n_rows - 1)
                    g_regs = self.generate_signal(g)[0]
                    del self.var2d_context[prev_rows_var.name]

                    # Track the body's per-slice inner registers so later block-rows'
                    # Signal2D reads of "row R+block_row" can copy them via Mov.
                    self.rvar_inners.setdefault(prev_rows_var.name, {})[block_row] = list(g_regs)

                    regs = self._new_registers(vec_type, n_slices, n_rows)
                    rep_template = f"Repeater<{vec_type}, {n_rows}, {block_size}, {block_row}>"
                    code = self._declare_node(
                        rep_template,
                        "",
                        [[regs[i], g_regs[i], repeater_context] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )

                    # Register-reuse Mov overrides for in-block reads.
                    # base[0] = current block-row's body output; base[i] for i>0 with
                    # block_row-i >= 0 = body output of block-row (block_row - i).
                    self._emit_repeater_movs(
                        vec_type,
                        n_slices,
                        n_rows,
                        block_row,
                        regs,
                        g_regs,
                        self.rvar_inners[prev_rows_var.name],
                    )

                case RVar2D(name):
                    block_row = self.current_block_row
                    block_size = self._block_size_template()
                    repeater_context, n_rows = self.var2d_context[name]
                    regs = self._new_registers(vec_type, n_slices, n_rows)
                    s2d_template = f"Signal2D<{vec_type}, {n_rows}, {block_size}, {block_row}>"
                    code = self._declare_node(
                        s2d_template,
                        "",
                        [[r, repeater_context] for r in regs],
                        vec_type,
                        vec_type,
                    )

                    # Mov overrides: Signal2D reads offset i+1 from block_row's
                    # current row → row R+block_row-(i+1). If block_row-(i+1) >= 0,
                    # that row was just produced inside this block.
                    if name in self.rvar_inners:
                        inners_by_br = self.rvar_inners[name]
                        for s in range(n_slices):
                            for i in range(n_rows):
                                src_br = block_row - (i + 1)
                                if src_br >= 0 and src_br in inners_by_br:
                                    self._declare_node(
                                        f"Mov<{vec_type}>",
                                        "",
                                        [[regs[s] + i, inners_by_br[src_br][s]]],
                                        vec_type,
                                        vec_type,
                                    )

                case Var2D(name, n_rows):
                    assert n_rows is not None
                    block_row = self.current_block_row
                    block_size = self._block_size_template()
                    regs = self._new_registers(vec_type, n_slices, n_rows)
                    zero_str = expr.element_type.val_to_str(ty.zero(), ty)
                    is2d_template = f"InputSignal2D<{vec_type}, {n_rows}, {block_size}, {block_row}>"
                    code = self._declare_node(
                        is2d_template,
                        f"{name}, {name}_stride, {zero_str}",
                        [[r] for r in regs],
                        vec_type,
                        vec_type,
                    )

                case ComputedSignal2D(name, cpp_struct, prelude, setup_code, make_args, n_rows):
                    assert n_rows is not None
                    block_row = self.current_block_row
                    block_size = self._block_size_template()
                    regs = self._new_registers(vec_type, n_slices, n_rows)
                    resolved_type = f"{cpp_struct}<{vec_type}, {n_rows}, {block_size}, {block_row}>"
                    # A signal that reads external buffers is constructed in driver scope
                    # (from setup_code) and passed into the kernel constructor via `param`;
                    # the member is initialized by copy from that parameter. A signal with
                    # no constructor args (make_args == "") carries no external state, so we
                    # default-construct the member directly and omit the constructor
                    # parameter — this also lets a signal used at several lane widths appear
                    # as several members without forcing the caller to name/pass each type.
                    param = self._new_var()
                    self.computed_signals.append(
                        ComputedSignalDecl(param, resolved_type, prelude, setup_code, make_args)
                    )
                    member_ctor_args = param if make_args.strip() else ""
                    code = self._declare_node(
                        resolved_type,
                        member_ctor_args,
                        [[r] for r in regs],
                        vec_type,
                        vec_type,
                    )

                case Signal2DAt(rv_inner, comp_idx):
                    assert isinstance(
                        rv_inner, RepeaterVec
                    ), f"Signal2DAt.a must be a RepeaterVec, got {type(rv_inner)}"
                    rvec_key = (rv_inner, self.current_block_row)
                    if rvec_key not in self.rvec_comp_regs:
                        self._generate_repeatervec(rv_inner)
                    regs = self.rvec_comp_regs[rvec_key][comp_idx]
                    code = ""

                case _:
                    raise AssertionError(f"CodeGen: unhandled SignalExpr2D {type(expr)}")
        elif isinstance(expr, SignalVectorExpr):
            match expr:
                case SignalVector(data):
                    n_vec = len(data)
                    s_regs = [self.generate_signal(s)[0] for s in data]
                    regs = self._new_registers(vec_type, n_slices, n_vec)
                    run_args_list = []
                    for s in range(n_slices):
                        src_var = self._declare_array("int", [s_regs[j][s] for j in range(n_vec)])
                        run_args_list.append([regs[s], src_var])
                    code = self._declare_node(
                        f"Tuple<{vec_type}, {n_vec}>",
                        "",
                        run_args_list,
                        vec_type,
                        vec_type,
                    )

                case ConvolveV(a_mat, b_in):
                    n_vec = a_mat.n
                    sem_suffix = self.get_sem_suffix(ty, element_type)
                    # Generate all kernel entries via the KernelMatrixExpr handler.
                    assert isinstance(a_mat, KernelMatrix)
                    self.generate_signal(a_mat)
                    # Generate input signal vector; component j is at g_regs[s]+j for slice s.
                    g_regs = self.generate_signal(b_in)[0]

                    # For each output row i, accumulate SConvolve(A[i][j], g[j]) over j.
                    out_regs_per_component: List[List[int]] = []
                    for i in range(n_vec):
                        partial_terms: List[List[int]] = []
                        for j in range(n_vec):
                            k = a_mat.data[i][j]
                            k_taps, k_idxs, _ = k.to_sparse_repr()
                            if k_taps == 0:
                                continue
                            k_regs = self.mapping[(k, self.ele_per_iter, self.current_block_row)][0]
                            idx_var = self._declare_array("int", k_idxs)
                            conv_regs = self._new_registers(vec_type, n_slices)
                            self._declare_node(
                                f"SConvolve{sem_suffix}<{vec_type}, {k_taps}, {idx_var}>",
                                "",
                                [[conv_regs[s], k_regs[s], g_regs[s] + j] for s in range(n_slices)],
                                vec_type,
                                vec_type,
                            )
                            partial_terms.append(conv_regs)
                        out_regs_per_component.append(
                            self._sum_registers(vec_type, n_slices, ty, element_type, partial_terms)
                        )

                    regs = self._new_registers(vec_type, n_slices, n_vec)
                    run_args_list = []
                    for s in range(n_slices):
                        src_var = self._declare_array("int", [out_regs_per_component[i][s] for i in range(n_vec)])
                        run_args_list.append([regs[s], src_var])
                    code = self._declare_node(
                        f"Tuple<{vec_type}, {n_vec}>",
                        "",
                        run_args_list,
                        vec_type,
                        vec_type,
                    )

                case RecurseV(a_mat, g_in):
                    n_vec = a_mat.n
                    assert expr.lanes is not None
                    sem_suffix = self.get_sem_suffix(ty, element_type)
                    assert isinstance(a_mat, KernelMatrix)

                    # Generate input signal vector; component j is at g_regs[s]+j for slice s.
                    g_regs = self.generate_signal(g_in)[0]

                    # Strip one vector-width delay from every kernel and generate.
                    stripped_kernels: List[List[KernelExpr]] = []
                    for i in range(n_vec):
                        stripped_row: List[KernelExpr] = []
                        for j in range(n_vec):
                            k = a_mat.data[i][j]
                            k_delay, sk = k.time_delay(expr.lanes)
                            assert k_delay == expr.lanes, (
                                f"RecurseV kernel A[{i}][{j}] has time delay {k_delay}, "
                                f"expected {expr.lanes} (one full vector width)"
                            )
                            self.generate_signal(sk)
                            stripped_row.append(sk)
                        stripped_kernels.append(stripped_row)

                    # Collect nonzero (i, j) pairs in row-major order.
                    pairs: List[Tuple[int, int, KernelExpr, int, List[int]]] = []
                    for i in range(n_vec):
                        for j in range(n_vec):
                            sk = stripped_kernels[i][j]
                            k_taps, k_idxs, _ = sk.to_sparse_repr()
                            if k_taps > 0:
                                pairs.append((i, j, sk, k_taps, list(k_idxs)))

                    n_pairs = len(pairs)
                    n_pairs = max(n_pairs, 1)  # C++ arrays must have size >= 1

                    # Compute flat tap-index list and per-pair offsets.
                    pair_offsets: List[int] = []
                    flat_idxs_list: List[int] = []
                    off = 0
                    for _, _, _, k_taps, k_idxs in pairs:
                        pair_offsets.append(off)
                        flat_idxs_list.extend(k_idxs)
                        off += k_taps
                    n_flat = off
                    n_flat = max(n_flat, 1)

                    # Per-component history buffer sizes (minimum 1).
                    comp_buf_sizes: List[int] = [1] * n_vec
                    for _, j, _, k_taps, k_idxs in pairs:
                        max_idx = k_idxs[-1]
                        buf_size = (max_idx + expr.lanes - 1) // expr.lanes + 1
                        comp_buf_sizes[j] = max(comp_buf_sizes[j], buf_size)

                    # Prefix-sum to get per-component buffer offsets.
                    comp_buf_offsets_list: List[int] = []
                    total_buf = 0
                    for bs in comp_buf_sizes:
                        comp_buf_offsets_list.append(total_buf)
                        total_buf += bs

                    # Declare static arrays for template parameters.
                    pad = [0]
                    v_pair_i = self._declare_array("int", [p[0] for p in pairs] if n_pairs > 0 else pad)
                    v_pair_j = self._declare_array("int", [p[1] for p in pairs] if n_pairs > 0 else pad)
                    v_pair_taps = self._declare_array("int", [p[3] for p in pairs] if n_pairs > 0 else pad)
                    v_pair_offsets = self._declare_array("int", pair_offsets if n_pairs > 0 else pad)
                    v_flat_idxs = self._declare_array("int", flat_idxs_list if n_flat > 0 else pad)
                    v_comp_buf_sizes = self._declare_array("int", comp_buf_sizes)
                    v_comp_buf_offsets = self._declare_array("int", comp_buf_offsets_list)

                    # Template uses actual counts for for_<> iteration; array sizes in prologue
                    # are padded to >= 1 so C++ array declarations are valid.
                    template_type = (
                        f"SRecurseV{sem_suffix}<{vec_type}, {n_vec}, "
                        f"{n_pairs}, {v_pair_i}, {v_pair_j}, {v_pair_taps}, {v_pair_offsets}, "
                        f"{n_flat}, {v_flat_idxs}, "
                        f"{v_comp_buf_sizes}, {v_comp_buf_offsets}, {total_buf}>"
                    )

                    regs = self._new_registers(vec_type, n_slices, n_vec)
                    run_args_list = []
                    for s in range(n_slices):
                        kernel_bases = [
                            self.mapping[(p[2], self.ele_per_iter, self.current_block_row)][0][s] for p in pairs
                        ]
                        v_kb = self._declare_array("int", kernel_bases if n_pairs > 0 else pad)
                        run_args_list.append([regs[s], v_kb, g_regs[s]])

                    code = self._declare_node(template_type, "", run_args_list, vec_type, vec_type)

                case ConvertLanesV(child_vec):
                    target_lanes = expr.lanes
                    assert target_lanes is not None
                    n_vec = expr.n
                    child_lanes = child_vec.lanes
                    assert child_lanes is not None
                    child_vec_type = self.get_vec_type(child_vec.element_type, child_lanes)
                    a_regs = self.generate_signal(child_vec)[0]
                    regs = self._new_registers(vec_type, n_slices, n_vec)
                    assert target_lanes != child_lanes
                    if target_lanes > child_lanes:
                        factor = target_lanes // child_lanes
                        assert target_lanes % child_lanes == 0
                        code = self._declare_node(
                            f"ConvertN2OneV<{child_vec_type}, {vec_type}, {n_vec}>",
                            "",
                            [[regs[i], a_regs[i * factor]] for i in range(n_slices)],
                            child_vec_type,
                            vec_type,
                        )
                    else:
                        factor = child_lanes // target_lanes
                        assert child_lanes % target_lanes == 0
                        code = ""
                        for off in range(factor):
                            code = self._declare_node(
                                f"ConvertOne2NV<{child_vec_type}, {vec_type}, {n_vec}, {off}>",
                                "",
                                [[regs[i], a_regs[i // factor]] for i in range(n_slices) if i % factor == off],
                                child_vec_type,
                                vec_type,
                            )
                case _:
                    assert False, f"Unsupported SignalVectorExpr: {type(expr)}"
        else:
            raise AssertionError(f"CodeGen: unhandled expr type {type(expr)}")

        assert cache_key not in self.mapping
        self.mapping[cache_key] = (regs, Code(code, ty, element_type, expr.lanes, taps))
        return self.mapping[cache_key]

    def _generate_repeatervec(self, rv: RepeaterVec) -> None:
        """Generate n Repeater C++ nodes (one per component) for a RepeaterVec
        at the active `current_block_row`.

        Stores per-component output registers in
        `self.rvec_comp_regs[(rv, block_row)]`. The shared per-component
        `RepeaterContext` is created on the first block-row's pass and reused.
        """
        assert rv.lanes is not None
        assert rv.n_rows is not None
        n = rv.n
        n_rows = rv.n_rows
        block_row = self.current_block_row
        block_size = self._block_size_template()
        vec_type = self.get_vec_type(rv.element_type, rv.lanes)
        n_slices = self.ele_per_iter // rv.lanes
        self.vec_types.add(vec_type)

        # One RepeaterContext per component, shared across all block-rows.
        rv_id = id(rv)
        if rv_id not in self.shared_repeatervec_contexts:
            zero_str = rv.element_type.val_to_str(rv.ty.zero(), rv.ty)
            self.shared_repeatervec_contexts[rv_id] = [
                self._declare_repeater_context(vec_type, zero_str, n_rows) for _ in range(n)
            ]
        contexts = self.shared_repeatervec_contexts[rv_id]

        # Register all prev_vars so body signals can read them
        for i, pv in enumerate(rv.prev_vars):
            self.var2d_context[pv.name] = (contexts[i], n_rows - 1)

        # Generate the body (SignalVectorExpr) — Tuple node writes n consecutive regs per slice
        body_regs = self.generate_signal(rv.a)[0]

        # Clean up prev_var context
        for pv in rv.prev_vars:
            del self.var2d_context[pv.name]

        # Track per-component, per-block-row inner registers so later block-rows'
        # Signal2D reads can copy them via Mov.
        for i, pv in enumerate(rv.prev_vars):
            comp_inner_per_slice = [body_regs[s] + i for s in range(n_slices)]
            self.rvar_inners.setdefault(pv.name, {})[block_row] = comp_inner_per_slice

        # Declare one Repeater<> per component, reading body_regs[s]+i for slice s
        comp_regs_list: List[List[int]] = []
        for i in range(n):
            comp_regs = self._new_registers(vec_type, n_slices, n_rows)
            rep_template = f"Repeater<{vec_type}, {n_rows}, {block_size}, {block_row}>"
            comp_inner_regs = [body_regs[s] + i for s in range(n_slices)]
            self._declare_node(
                rep_template,
                "",
                [[comp_regs[s], comp_inner_regs[s], contexts[i]] for s in range(n_slices)],
                vec_type,
                vec_type,
            )

            self._emit_repeater_movs(
                vec_type,
                n_slices,
                n_rows,
                block_row,
                comp_regs,
                comp_inner_regs,
                self.rvar_inners[rv.prev_vars[i].name],
            )
            comp_regs_list.append(comp_regs)

        self.rvec_comp_regs[(rv, block_row)] = comp_regs_list

    def _sum_registers(
        self,
        vec_type: str,
        n_slices: int,
        ty: Type,
        element_type: ElementType,
        addends: List[List[int]],
    ) -> List[int]:
        """Sum a list of per-slice register lists using SAdd nodes. Returns result registers.
        If addends is empty, returns a Signal1DConstant(zero) register."""
        sem_suffix = self.get_sem_suffix(ty, element_type)
        accumulated: List[int] | None = None
        for addend in addends:
            if accumulated is None:
                accumulated = addend
            else:
                new_regs = self._new_registers(vec_type, n_slices)
                self._declare_node(
                    f"SAdd{sem_suffix}<{vec_type}>",
                    "",
                    [[new_regs[s], accumulated[s], addend[s]] for s in range(n_slices)],
                    vec_type,
                    vec_type,
                )
                accumulated = new_regs
        if accumulated is None:
            accumulated = self._new_registers(vec_type, n_slices)
            self._declare_node(
                f"Signal1DConstant<{vec_type}>",
                element_type.val_to_str(ty.zero(), ty),
                [[r] for r in accumulated],
                vec_type,
                vec_type,
            )
        return accumulated

    def enforce_lanes(
        self, node: ConvertLanes | KConvertLanes, n_slices: int, kernel_taps: int | None = None
    ) -> Tuple[List[int], str]:
        prefix = "K" if isinstance(node, KConvertLanes) else ""
        taps_suffix = f", {kernel_taps}" if kernel_taps is not None else ""
        reg_size = kernel_taps if kernel_taps is not None else 1

        a = node.a
        lanes = node.lanes
        child_lanes = a.lanes
        assert lanes is not None and child_lanes is not None
        vec_type = self.get_vec_type(node.element_type, lanes)
        child_vec_type = self.get_vec_type(a.element_type, child_lanes)
        a_regs = self.generate_signal(a)[0]
        regs = self._new_registers(vec_type, n_slices, reg_size)

        assert lanes != child_lanes, "ConvertLanes should change the number of lanes"
        if lanes > child_lanes:
            factor = lanes // child_lanes
            assert lanes % child_lanes == 0
            code = self._declare_node(
                f"{prefix}ConvertN2One<{child_vec_type}, {vec_type}{taps_suffix}>",
                "",
                [[regs[i], a_regs[i * factor]] for i in range(n_slices)],
                child_vec_type,
                vec_type,
            )
        else:
            factor = child_lanes // lanes
            assert child_lanes % lanes == 0
            code = ""
            for off in range(factor):
                code = self._declare_node(
                    f"{prefix}ConvertOne2N<{child_vec_type}, {vec_type}{taps_suffix}, {off}>",
                    "",
                    [[regs[i], a_regs[i // factor]] for i in range(n_slices) if i % factor == off],
                    child_vec_type,
                    vec_type,
                )
        return regs, code

    def _block_size_template(self) -> int:
        """Template parameter for `block_size`. Always equals `rows_per_iter`
        (which is >= 1)."""
        return self.rows_per_iter

    def _emit_repeater_movs(
        self,
        vec_type: str,
        n_slices: int,
        n_rows: int,
        block_row: int,
        regs: List[int],
        g_regs: List[int],
        inners_by_br: Dict[int, List[int]],
    ) -> None:
        """Emit Mov nodes that override Repeater::run's cache reads with the
        in-block body outputs. base[0] is always the current block-row's body
        output; base[i] for i in 1..n_rows-1 is block-row (block_row-i)'s body
        output if that row is in this block (block_row-i >= 0); otherwise the
        cache read from Repeater::run stays in place."""
        for s in range(n_slices):
            self._declare_node(f"Mov<{vec_type}>", "", [[regs[s] + 0, g_regs[s]]], vec_type, vec_type)
            for i in range(1, n_rows):
                src_br = block_row - i
                if src_br >= 0 and src_br in inners_by_br:
                    self._declare_node(
                        f"Mov<{vec_type}>",
                        "",
                        [[regs[s] + i, inners_by_br[src_br][s]]],
                        vec_type,
                        vec_type,
                    )

    def _contains_repeater(self, expr: RecLang, visited: Optional[Set[int]] = None) -> bool:
        """True if the expression tree contains a Repeater or RepeaterVec node.
        Used to pick the auto-detect default of `rows_per_iter` (2 for 2D
        recurrences, 1 otherwise)."""
        if visited is None:
            visited = set()
        if id(expr) in visited:
            return False
        visited.add(id(expr))
        if isinstance(expr, (Repeater, RepeaterVec)):
            return True
        for child in expr.children():
            if self._contains_repeater(child, visited):
                return True
        return False

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

    def get_vec_type(self, element_type: ElementType, lanes: int) -> str:
        """
        Get the C++ type of the given element type and lanes.
        Use maximum number of lanes if not provided.
        """
        match element_type:
            case ElementType.Float:
                return f"float_vec{lanes}"
            case ElementType.Double:
                return f"double_vec{lanes}"
            case ElementType.I32:
                return f"int32_vec{lanes}"
            case ElementType.I64:
                return f"int64_vec{lanes}"
            case ElementType.I8:
                return f"int8_vec{lanes}"
            case ElementType.I16:
                return f"int16_vec{lanes}"

    def _new_register(self, vec_type: str, n: int = 1) -> int:
        curr = self.reg_counter.get(vec_type, 0)
        self.reg_counter[vec_type] = curr + n
        return curr

    def _new_registers(self, vec_type: str, n_slices: int, n: int = 1) -> List[int]:
        return [self._new_register(vec_type, n) for _ in range(n_slices)]

    def _new_var(self) -> str:
        k = self.var_counter
        self.var_counter += 1
        return f"v{k}"

    def _declare_array(self, ty: str, value: list) -> str:
        var = self._new_var()
        value_args = ", ".join(map(str, value))
        self.prologue.append(f"constexpr static {ty} {var}[{len(value)}] = {{{value_args}}};")
        return var

    def _declare_repeater_context(self, vec_type: str, zero_str: str, n_rows: int) -> str:
        # RepeaterContext is held by value in the kernel struct; it has no run() method
        # and is passed by reference into Signal2D/Repeater run() calls. Its
        # reset_and_next_row() is invoked alongside other nodes via the value field.
        return self._declare_node(
            f"RepeaterContext<{vec_type}, {n_rows}>",
            f"row_size, {zero_str}",
            [],
            vec_type,
            vec_type,
            False,
        )

    def _declare_node(
        self,
        class_name: str,
        constructor_args: str,
        run_args_list: list,
        input_type: str,
        output_type: str,
        is_pointer: bool = False,
    ) -> str:
        var = self._new_var()
        str_run_args_list = [[str(a) for a in run_args] for run_args in run_args_list]
        node = Node(class_name, var, constructor_args, str_run_args_list, input_type, output_type, is_pointer)
        self.program.append(node)
        return var

    def _to_cpp_list(self, lst: list) -> str:
        return ", ".join(map(str, lst))

    def get_sem_suffix(self, ty: Type, element_type: ElementType | None = None) -> str:
        if isinstance(ty, CustomSemiring):
            self.custom_semirings.add(ty)
            return ty.name
        # int8/int16 tropical ops use saturating addition to prevent overflow
        if element_type in (ElementType.I8, ElementType.I16):
            if ty is Type.Arith:
                return ""
            elif ty is Type.TropMax:
                return "TropMaxSat"
            elif ty is Type.TropMin:
                return "TropMinSat"
        if ty is Type.Arith:
            return ""
        elif ty is Type.TropMax:
            return "TropMax"
        elif ty is Type.TropMin:
            return "TropMin"
        assert False, f"Unknown built-in type: {ty}"

    def _signal2d_n_rows(self, signal2d: SignalExpr2D) -> int:
        # TODO: var2d_context
        if isinstance(signal2d, RVar2D):
            if signal2d.name not in self.var2d_context:
                raise ValueError(f"RVar2D '{signal2d.name}' used outside of Repeater context")
            return self.var2d_context[signal2d.name][1]

        if isinstance(signal2d, (Repeater, Var2D, ComputedSignal2D)):
            assert signal2d.n_rows is not None, (
                f"{type(signal2d).__name__} has n_rows=None at codegen time. "
                "Did you forget to include InferNRows in your schedule before AnnotateLanes?"
            )
            return signal2d.n_rows

        if isinstance(signal2d, Signal2DAt):
            rv = signal2d.a
            assert isinstance(rv, RepeaterVec)
            assert rv.n_rows is not None, (
                "RepeaterVec has n_rows=None at codegen time. "
                "Did you forget to include InferNRows in your schedule before AnnotateLanes?"
            )
            return rv.n_rows

        raise ValueError(f"Ith expects RVar2D, Repeater, Var2D, or Signal2DAt as signal2d, got {type(signal2d)}")
