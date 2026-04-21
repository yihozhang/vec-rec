from typing import List, Optional, Sequence, Dict, Set, Tuple, Union, cast
from unittest import case
from unittest import case
from vecrec.expr import *
from vecrec.expr.base import SignalExpr2D, SignalVectorExpr
from vecrec.expr.signal import RVar2D, Var2D
from vecrec.expr.signal_ops import Repeater, Ith
from vecrec.expr.kernel import TIKernel, TVKernel
from vecrec.expr.kernel_matrix import KernelMatrix
from vecrec.expr.signal_vec import SignalVector, ConvolveV, RecurseV, VectorAt
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

    def __init__(self, class_name: str, var_name: str, constructor_args: str, run_args_list: List[List[str]], input_type: str, output_type: str):
        self.class_name = class_name
        self.var_name = var_name
        self.constructor_args = constructor_args
        self.run_args_list = run_args_list
        self.input_type = input_type
        self.output_type = output_type


class CodeGen:
    """ele_per_iter determines the number of elements to be processed within each element."""

    ele_per_iter: int
    # mapping from a node to its register(s) and code information
    mapping: Dict[RecLang, Tuple[List[int], Code]]
    # All the vector types used in the generated code. Useful for determining which registers to declare.
    vec_types: Set[str]
    # Map RVar2D names to their context variables and n_rows
    var2d_context: Dict[str, Tuple[str, int]]
    reg_counter: int
    var_counter: int

    # The list of AST declaration
    # (classname, variable_name, constructor_args)
    program: List[Node]
    # The lsit of constant declaration
    prologue: List[str]

    def __init__(self, ele_per_iter: int = 16):
        self.ele_per_iter = ele_per_iter
        self.reg_counter = 0
        self.var_counter = 0
        self.vec_types = set()
        self.mapping = {}
        self.program = []
        self.prologue = []
        self.var2d_context = {}

    def clear(self):
        self.reg_counter = 0
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
        total_regs = self.reg_counter
        new_line = "\n            "
        assert expr.lanes is not None
        result_vec_type = self.get_vec_type(expr.element_type, expr.lanes)

        def generate_run_calls(x: Node) -> str:
            reg_prefix = f"reg_{x.output_type}" if x.output_type == x.input_type else f"reg_{x.input_type}, reg_{x.output_type}"
            return new_line.join(
                f"{x.var_name}.run({reg_prefix}, {', '.join(run_args)});"
                for run_args in x.run_args_list
            )
        text = f"""
        struct {name} {{
            // static array declarations
            {new_line.join(self.prologue)}
            // node declarations
            {new_line.join(map(lambda x: f"{x.class_name} {x.var_name};", self.program))}
            // register declaration
            {new_line.join(map(lambda vt: f"Reg<{vt}, {total_regs}> reg_{vt};", self.vec_types))}
            {name}({args}) : 
              {", ".join(map(lambda x: f"{x.var_name}({x.constructor_args})", self.program))}
              {{}}
            
            void run({result_vec_type} *out) {{
                {new_line.join(map(generate_run_calls, self.program))}
                {new_line.join(map(lambda i: f"memcpy(out + {i}, reg_{result_vec_type}.data + {result[0][i]}, sizeof({result_vec_type}));", range(len(result[0]))))}
            }}

            void reset_and_next_row() {{
            {new_line.join(map(lambda x: f"{x.var_name}.reset_and_next_row();", self.program))}
            }}
        }};

        using {name}_vector_type = {result_vec_type};

        auto make_{name}({args}) {{
            return {name}({", ".join(list(vars.keys()) + list(vars2d.keys()) + ["row_size"])});
        }}
        """
        return KernelProgram(text, self.ele_per_iter)

    def generate_signal(self, expr: SignalExpr | SignalExpr2D | KernelExpr) -> Tuple[List[int], Code]:
        if expr in self.mapping:
            return self.mapping[expr]

        assert expr.lanes is not None
        assert self.ele_per_iter % expr.lanes == 0

        vec_type = self.get_vec_type(expr.element_type, expr.lanes)
        self.vec_types.add(vec_type)
        element_type = expr.element_type
        ty = expr.ty

        n_slices = self.ele_per_iter // expr.lanes
        taps: int | None = None

        if isinstance(expr, SignalExpr):
            match expr:
                case Num(value):
                    regs = self._new_registers(n_slices)
                    code = self._declare_node(
                        f"Signal1DConstant<{vec_type}>",
                        expr.element_type.val_to_str(value, ty),
                        [[r] for r in regs], vec_type, vec_type,
                    )
                case Impulse(value):
                    regs = self._new_registers(n_slices)
                    value_str = expr.element_type.val_to_str(value, ty)
                    zero_str = expr.element_type.val_to_str(ty.zero(), ty)
                    code = self._declare_node(
                        f"Signal1DImpulse<{vec_type}>", f"{value_str}, {zero_str}", [[r] for r in regs], vec_type, vec_type
                    )
                case Var(name):
                    regs = self._new_registers(n_slices)
                    code = self._declare_node(f"Signal1D<{vec_type}>", f"{name}", [[r] for r in regs], vec_type, vec_type)

                case SignalExprBinOp(a, b):
                    a_regs = self.generate_signal(a)[0]
                    b_regs = self.generate_signal(b)[0]
                    regs = self._new_registers(n_slices)
                    constructor = type(expr).__name__ + self.get_sem_suffix(ty)
                    code = self._declare_node(
                        f"{constructor}<{vec_type}>", "",
                        [[regs[i], a_regs[i], b_regs[i]] for i in range(n_slices)],
                        vec_type, vec_type,
                    )
                case SNeg(a):
                    a_regs = self.generate_signal(a)[0]
                    regs = self._new_registers(n_slices)
                    code = self._declare_node(
                        f"SNeg<{vec_type}>", "",
                        [[regs[i], a_regs[i]] for i in range(n_slices)],
                        vec_type, vec_type,
                    )

                case Convolve(a, f):
                    taps, idxs, _ = a.to_sparse_repr()
                    idx_var = self._declare_array("int", idxs)
                    a_regs = self.generate_signal(a)[0]
                    f_regs = self.generate_signal(f)[0]
                    regs = self._new_registers(n_slices)
                    constructor = "SConvolve" + self.get_sem_suffix(ty)
                    code = self._declare_node(
                        f"{constructor}<{vec_type}, {taps}, {idx_var}>", "",
                        [[regs[i], a_regs[i], f_regs[i]] for i in range(n_slices)],
                        vec_type, vec_type,
                    )
                case Recurse(a, g):
                    time_delay, a = a.time_delay(expr.lanes)
                    assert time_delay == a.lanes
                    taps, idxs, _ = a.to_sparse_repr()
                    idx_var = self._declare_array("int", idxs)
                    a_regs = self.generate_signal(a)[0]
                    g_regs = self.generate_signal(g)[0]
                    regs = self._new_registers(n_slices)
                    constructor = "SRecurse" + self.get_sem_suffix(ty)
                    code = self._declare_node(
                        f"{constructor}<{vec_type}, {taps}, {idx_var}>", "",
                        [[regs[i], a_regs[i], g_regs[i]] for i in range(n_slices)],
                        vec_type, vec_type,
                    )

                case Ith(signal2d, ith_row):
                    raise NotImplementedError("Ith not yet implemented in CodeGen")

                case ConvertLanes(a):
                    regs, code = self.enforce_lanes(expr, n_slices)

                case _:
                    raise NotImplementedError(f"CodeGen: unhandled SignalExpr {type(expr)}")
    
        elif isinstance(expr, KernelExpr):
            coeffs: List[float] | List[SignalExpr]
            if isinstance(expr, TIKernel):
                taps, indices, coeffs = expr.to_sparse_repr()
                index_var = self._declare_array("int", indices)
                value_var = self._declare_array(
                    element_type.to_str(), [f"{expr.element_type.val_to_str(v, expr.ty)}" for v in coeffs]
                )
                regs = self._new_registers(n_slices, taps)
                code = self._declare_node(
                    f"TimeInvariantKernel<{taps}, {vec_type}, {index_var}, {value_var}>",
                    "", [[r] for r in regs], vec_type, vec_type,
                )
            elif isinstance(expr, TVKernel):
                taps, indices, coeffs = expr.to_sparse_repr()
                regs = self._new_registers(n_slices, taps)
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

        if isinstance(expr, SignalExpr2D):
            match expr:
                case Repeater(a, n_rows, prev_rows_var):
                    raise NotImplementedError("Repeater not yet implemented in CodeGen")
                
                case RVar2D(name):
                    raise NotImplementedError("RVar2D not yet implemented in CodeGen")
                case Var2D(name, n_rows):
                    assert n_rows is not None
                    regs = self._new_registers(n_slices, n_rows)
                    code = self._declare_node(
                        f"InputSignal2D<{vec_type}, {n_rows}>", f"{name}, {name}_stride, {ty.zero()}", [[r] for r in regs], vec_type, vec_type
                    )

                case _:
                    raise NotImplementedError(f"CodeGen: unhandled SignalExpr2D {type(expr)}")
        else:
            raise NotImplementedError(f"CodeGen: unhandled expr type {type(expr)}")

        assert expr not in self.mapping
        self.mapping[expr] = (regs, Code(code, ty, element_type, expr.lanes, taps))
        return self.mapping[expr]

    def enforce_lanes(self, node: ConvertLanes | KConvertLanes, n_slices: int, kernel_taps: int | None = None) -> Tuple[List[int], str]:
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
        regs = self._new_registers(n_slices, reg_size)

        assert lanes != child_lanes, "ConvertLanes should change the number of lanes"
        if lanes > child_lanes:
            factor = lanes // child_lanes
            assert lanes % child_lanes == 0
            code = self._declare_node(
                f"{prefix}ConvertN2One<{child_vec_type}, {vec_type}{taps_suffix}>", "",
                [[regs[i], a_regs[i * factor]] for i in range(n_slices)],
                child_vec_type, vec_type,
            )
        else:
            factor = child_lanes // lanes
            assert child_lanes % lanes == 0
            code = ""
            for off in range(factor):
                code = self._declare_node(
                    f"{prefix}ConvertOne2N<{child_vec_type}, {vec_type}{taps_suffix}, {off}>", "",
                    [[regs[i], a_regs[i // factor]] for i in range(n_slices) if i % factor == off],
                    child_vec_type, vec_type,
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

    def _new_register(self, n: int = 1) -> int:
        curr = self.reg_counter
        self.reg_counter += n
        return curr

    def _new_registers(self, n_slices: int, n: int = 1) -> List[int]:
        return [self._new_register(n) for _ in range(n_slices)]

    def _new_var(self) -> str:
        k = self.var_counter
        self.var_counter += 1
        return f"v{k}"

    def _declare_array(self, ty: str, value: list) -> str:
        var = self._new_var()
        value_args = ", ".join(map(str, value))
        self.prologue.append(f"constexpr static {ty} {var}[{len(value)}] = {{{value_args}}};")
        return var

    def _declare_node(self, class_name: str, constructor_args: str, run_args_list: list, input_type: str, output_type: str) -> str:
        var = self._new_var()
        str_run_args_list = [[str(a) for a in run_args] for run_args in run_args_list]
        node = Node(class_name, var, constructor_args, str_run_args_list, input_type, output_type)
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


class CodeGen2:
    # Hardware-dependent maximum
    counter: int
    prologue: list[str]
    # Maps RVar2D names to their context variable names (str) and n_rows (int)
    var2d_context: Dict[str, Union[str, int]]
    # Maps expressions (by structural equality) to Code object for caching identical AST structures
    expr_cache: Optional[Dict]
    # Maps multi-use signal exprs to (ptr_var_name, primary_Code); populated on first visit
    cse_ptr_vars: Dict
    # Reference counts per non-trivial signal expression (structural equality)
    ref_counts: Dict

    def __init__(self, cache_cse: bool = True):
        self.counter = 0
        self.prologue = []
        self.var2d_context = {}
        self.expr_cache = {} if cache_cse else None
        self.cse_ptr_vars = {}
        self.ref_counts = {}

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
            case SNeg(a):
                vec_type = self.get_vec_type(expr.element_type, expr.lanes)
                code_a = self.generate_signal(a)
                assert code_a.element_type == expr.element_type
                assert code_a.ty == expr.ty
                code = Code(
                    f"make_SNeg<{vec_type}>({code_a.text})",
                    expr.ty,
                    expr.element_type,
                    expr.lanes,
                )

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
                    f"make_input_signal2d<{vec_type}, {expr.n_rows}>({expr.name}, {expr.name}_stride, {expr.ty.zero()})",
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
                    f"RepeaterContext<{vec_type}, {n_rows}>* {context_var} = RepeaterContext<{vec_type}, {n_rows}>(row_size);"
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
                assert False, f"CodeGen: unhandled case for expr {expr} of type {type(expr)}"

        # Cache and return at the end
        if not self._is_trivial_expr(expr):
            var = self.get_new_var()
            self.prologue.append(f"auto {var} = {code.text};")
            code = Code(var, code.ty, code.element_type, code.lanes, code.taps)

            assert self.expr_cache is not None
            secondary_code = Code(
                f"make_Secondary<{self.get_vec_type(code.element_type, code.lanes)}>({var})",
                code.ty,
                code.element_type,
                code.lanes,
                code.taps,
            )
            self.expr_cache[expr] = secondary_code

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

            assert self.expr_cache is not None
            secondary_code = Code(
                f"make_Secondary<{self.get_vec_type(code.element_type, code.lanes)}>({var})",
                code.ty,
                code.element_type,
                code.lanes,
                code.taps,
            )
            self.expr_cache[expr] = secondary_code
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
            case SignalVector(_data):
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
                    [self.generate_kernel(stripped[i][j]) for j in range(n)] for i in range(n)
                ]

                # Generate C++ code for each input signal
                signal_codes = [self.generate_signal(g.data[i]) for i in range(n)]

                # Build factory call for n=2
                k00, k01, k10, k11 = (
                    kernel_codes[0][0].text,
                    kernel_codes[0][1].text,
                    kernel_codes[1][0].text,
                    kernel_codes[1][1].text,
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
