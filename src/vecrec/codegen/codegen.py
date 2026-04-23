from typing import List, Optional, Dict, Set, Tuple, cast
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D, SignalVectorExpr, KernelMatrixExpr
from vecrec.expr.signal import RVar2D, Var2D
from vecrec.expr.signal_ops import Repeater, Ith
from vecrec.expr.kernel import TIKernel, TVKernel
from vecrec.expr.kernel_matrix import KernelMatrix
from vecrec.expr.signal_vec import SignalVector, ConvolveV, RecurseV, ConvertLanesV, VectorAt
from vecrec.transform import Transform
from vecrec.util import ElementType
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


class KernelProgram:
    text: str
    element_per_iter: int

    def __init__(self, text: str, element_per_iter: int):
        self.text = text
        self.element_per_iter = element_per_iter


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
    # mapping from a node to its register(s) and code information
    mapping: Dict[RecLang, Tuple[List[int], Code]]
    # All the vector types used in the generated code. Useful for determining which registers to declare.
    vec_types: Set[str]
    # Map RVar2D names to their context variables and n_rows
    var2d_context: Dict[str, Tuple[str, int]]
    # Map vector types to their current register count, used for generating new registers
    reg_counter: Dict[str, int]
    var_counter: int

    # The list of AST declaration
    # (classname, variable_name, constructor_args)
    program: List[Node]
    # The lsit of constant declaration
    prologue: List[str]

    def __init__(self, ele_per_iter: int = 16):
        self.ele_per_iter = ele_per_iter
        self.reg_counter = {}
        self.var_counter = 0
        self.vec_types = set()
        self.mapping = {}
        self.program = []
        self.prologue = []
        self.var2d_context = {}

    def clear(self):
        self.reg_counter = {}
        self.var_counter = 0
        self.vec_types = set()
        self.mapping = {}
        self.program = []
        self.prologue = []
        self.var2d_context = {}

    def generate(self, expr: SignalExpr | SignalExpr2D, name: str) -> KernelProgram:

        self.clear()
        result = self.generate_signal(expr)
        vars, vars2d = self.collect_input_variables(expr)
        args_1d = [f"const {elt_type.to_str()} *{var}" for var, elt_type in sorted(vars.items())]
        args_2d = [f"const {elt_type.to_str()} *{var}, int {var}_stride" for var, elt_type in sorted(vars2d.items())]
        all_args = args_1d + args_2d + ["int row_size"]
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

        text = f"""
        struct {name} {{
            // static array declarations
            {new_line.join(self.prologue)}
            // node declarations
            {new_line.join(map(lambda x: f"{x.class_name} {x.var_name};", self.program))}
            // register declaration
            {new_line.join(map(lambda vt: f"Reg<{vt}, {self.reg_counter[vt]}> reg_{vt};", self.vec_types))}
            inline __attribute__((always_inline)) {name}({args}) : 
              {", ".join(map(lambda x: f"{x.var_name}({x.constructor_args})", self.program))}
              {"".join(map(lambda x: f", reg_{x}()", self.vec_types))}
              {{}}
            
            inline __attribute__((always_inline)) void run({result_vec_type} *out) {{
                {new_line.join(map(generate_run_calls, self.program))}
                {new_line.join(map(lambda i: f"memcpy(out + {i}, reg_{result_vec_type}.data + {result[0][i]}, sizeof({result_vec_type}));", range(len(result[0]))))}
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
        return KernelProgram(text, self.ele_per_iter)

    def generate_signal(
        self, expr: SignalExpr | SignalExpr2D | KernelExpr | SignalVectorExpr | KernelMatrixExpr
    ) -> Tuple[List[int], Code]:
        if expr in self.mapping:
            return self.mapping[expr]

        # KernelMatrixExpr has no own registers; just recursively generate each entry.
        if isinstance(expr, KernelMatrixExpr):
            for k in expr.children():
                self.generate_signal(cast(KernelExpr, k))
            placeholder: Tuple[List[int], Code] = ([], Code("", expr.ty, expr.element_type, expr.lanes))
            self.mapping[expr] = placeholder
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
                    value_str = expr.element_type.val_to_str(value, ty)
                    zero_str = expr.element_type.val_to_str(ty.zero(), ty)
                    code = self._declare_node(
                        f"Signal1DImpulse<{vec_type}>",
                        f"{value_str}, {zero_str}",
                        [[r] for r in regs],
                        vec_type,
                        vec_type,
                    )
                case Var(name):
                    regs = self._new_registers(vec_type, n_slices)
                    code = self._declare_node(
                        f"Signal1D<{vec_type}>", f"{name}", [[r] for r in regs], vec_type, vec_type
                    )

                case SignalExprBinOp(a, b):
                    a_regs = self.generate_signal(a)[0]
                    b_regs = self.generate_signal(b)[0]
                    regs = self._new_registers(vec_type, n_slices)
                    constructor = type(expr).__name__ + self.get_sem_suffix(ty)
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
                    constructor = "SConvolve" + self.get_sem_suffix(ty)
                    code = self._declare_node(
                        f"{constructor}<{vec_type}, {taps}, {idx_var}>",
                        "",
                        [[regs[i], a_regs[i], f_regs[i]] for i in range(n_slices)],
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
                    constructor = "SRecurse" + self.get_sem_suffix(ty)
                    code = self._declare_node(
                        f"{constructor}<{vec_type}, {taps}, {idx_var}>",
                        "",
                        [[regs[i], a_regs[i], g_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )

                case ConvertLanes(a):
                    regs, code = self.enforce_lanes(expr, n_slices)

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
                    regs = self._new_registers(vec_type, n_slices)
                    repeater_context = self._declare_repeater_context(vec_type, n_rows)

                    self.var2d_context[prev_rows_var.name] = (repeater_context, n_rows - 1)
                    g_regs = self.generate_signal(g)[0]
                    del self.var2d_context[prev_rows_var.name]

                    code = self._declare_node(
                        f"Repeater<{vec_type}, {n_rows}>",
                        repeater_context,
                        [[regs[i], g_regs[i]] for i in range(n_slices)],
                        vec_type,
                        vec_type,
                    )

                case RVar2D(name):
                    regs = self._new_registers(vec_type, n_slices)

                    repeater_context, n_rows = self.var2d_context[name]
                    code = self._declare_node(
                        f"Signal2D<{vec_type}, {n_rows}>",
                        repeater_context,
                        [[r] for r in regs],
                        vec_type,
                        vec_type,
                    )

                case Var2D(name, n_rows):
                    assert n_rows is not None
                    regs = self._new_registers(vec_type, n_slices, n_rows)
                    zero_str = expr.element_type.val_to_str(ty.zero(), ty)
                    code = self._declare_node(
                        f"InputSignal2D<{vec_type}, {n_rows}>",
                        f"{name}, {name}_stride, {zero_str}",
                        [[r] for r in regs],
                        vec_type,
                        vec_type,
                    )

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
                    sem_suffix = self.get_sem_suffix(ty)
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
                            k_regs = self.mapping[k][0]
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
                    sem_suffix = self.get_sem_suffix(ty)
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
                    for (_, _, _, k_taps, k_idxs) in pairs:
                        pair_offsets.append(off)
                        flat_idxs_list.extend(k_idxs)
                        off += k_taps
                    n_flat = off
                    n_flat = max(n_flat, 1)

                    # Per-component history buffer sizes (minimum 1).
                    comp_buf_sizes: List[int] = [1] * n_vec
                    for (_, j, _, k_taps, k_idxs) in pairs:
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
                        kernel_bases = [self.mapping[p[2]][0][s] for p in pairs]
                        v_kb = self._declare_array("int", kernel_bases if n_pairs > 0 else pad)
                        run_args_list.append([regs[s], v_kb, g_regs[s]])

                    code = self._declare_node(
                        template_type, "", run_args_list, vec_type, vec_type
                    )
                
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

        assert expr not in self.mapping
        self.mapping[expr] = (regs, Code(code, ty, element_type, expr.lanes, taps))
        return self.mapping[expr]

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
        sem_suffix = self.get_sem_suffix(ty)
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
            case ElementType.I32:
                return f"int32_vec{lanes}"
            case ElementType.I64:
                return f"int64_vec{lanes}"

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

    def _declare_repeater_context(self, vec_type: str, n_rows: int) -> str:
        # The input and output type of RepeaterContext are bogus, they are here to generate a parameter for `.run()` calls.
        return self._declare_node(
            f"RepeaterContext<{vec_type}, {n_rows}>*",
            f"new RepeaterContext<{vec_type}, {n_rows}>(row_size)",
            [[]],
            vec_type,
            vec_type,
            True,
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

    def get_sem_suffix(self, ty: Type) -> str:
        match ty:
            case Type.Arith:
                return ""
            case Type.TropMax:
                return "TropMax"
            case Type.TropMin:
                return "TropMin"

    def _signal2d_n_rows(self, signal2d: SignalExpr2D) -> int:
        # TODO: var2d_context
        if isinstance(signal2d, RVar2D):
            if signal2d.name not in self.var2d_context:
                raise ValueError(f"RVar2D '{signal2d.name}' used outside of Repeater context")
            return self.var2d_context[signal2d.name][1]

        if isinstance(signal2d, (Repeater, Var2D)):
            assert signal2d.n_rows is not None, (
                f"{type(signal2d).__name__} has n_rows=None at codegen time. "
                "Did you forget to include InferNRows in your schedule before AnnotateLanes?"
            )
            return signal2d.n_rows

        raise ValueError(f"Ith expects RVar2D, Repeater, or Var2D as signal2d, got {type(signal2d)}")
