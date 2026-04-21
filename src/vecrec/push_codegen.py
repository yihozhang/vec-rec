"""Push-based (register-scheduled) code generator for VecRec.

Each node is a C++ struct with a run(Reg_t& reg) method.  The driver owns
all stateful nodes as fields.  Its run() body is a flat sequence of
``n0.run(reg); n1.run(reg); …`` where ``reg`` is a stack-allocated register
file of type ``Reg<vec_type, N>``.  LLVM sees the entire computation in one
function and can promote the register file to physical vector registers.
CSE is free: shared sub-expressions share a single register slot.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from vecrec.codegen import (
    Code,
    KernelExecutable,
    _generate_benchmark_code,
    _generate_kernel_runner_code,
    _numpy_dtype_for_element_type,
)
from vecrec.expr import Convolve, Impulse, Num, Recurse, Type, Var
from vecrec.expr.base import SignalExpr, SignalExpr2D, SignalVectorExpr
from vecrec.expr.kernel import TIKernel, TVKernel
from vecrec.expr.kernel_ops import KConvertLanes
from vecrec.expr.kernel_matrix import KernelMatrix
from vecrec.expr.signal import RVar2D, Var2D
from vecrec.expr.signal_ops import ConvertLanes, Ith, Repeater, SNeg
from vecrec.expr.signal_ops import SAdd  # noqa: F401 – kept for isinstance checks
from vecrec.expr.signal_ops import PointwiseMul, PointwiseDiv, SSub
from vecrec.expr.signal_ops import SignalExprBinOp
from vecrec.expr.signal_vec import ConvolveV, RecurseV, SignalVector, VectorAt
from vecrec.util import ElementType

__all__ = [
    "PushCodeGen",
    "generate_push_kernel_executable",
    "generate_and_run_push_benchmark",
    "push_instantiate_kernels",
]

PUSH_TEMPLATE = files("vecrec.templates") / "push.h"


# ---------------------------------------------------------------------------
# PushCodeGen
# ---------------------------------------------------------------------------


class PushCodeGen:
    """Generates push-based C++ driver structs for signal expressions."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.reg_counter = 0
        self.field_counter = 0
        self.var_counter = 0
        self.driver_counter = 0
        self.prologue: List[str] = []
        self.inner_structs: List[str] = []
        # Maps (structural-eq hash) → out_reg for signal CSE
        self.reg_map: Dict = {}
        # Maps (structural-eq hash) → (k_base_reg, taps, max_index, idxs_var)
        # for TIKernel deduplication
        self.kernel_cache: Dict = {}
        # Maps RVar2D.name → (rv_base_reg, n_prev_rows) inside inner drivers
        self.rvar2d_map: Dict[str, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Allocation helpers
    # ------------------------------------------------------------------

    def _alloc_reg(self, n: int = 1) -> int:
        base = self.reg_counter
        self.reg_counter += n
        return base

    def _new_field(self) -> str:
        name = f"n{self.field_counter}"
        self.field_counter += 1
        return name

    def _new_var(self) -> str:
        name = f"v{self.var_counter}"
        self.var_counter += 1
        return name

    def _new_driver_name(self, outer_name: str) -> str:
        name = f"InnerDriver_{outer_name}_{self.driver_counter}"
        self.driver_counter += 1
        return name

    # ------------------------------------------------------------------
    # Type helpers (mirrors CodeGen)
    # ------------------------------------------------------------------

    def get_vec_type(self, element_type: ElementType, lanes: int) -> str:
        match element_type:
            case ElementType.Float:
                return f"float_vec{lanes}"
            case ElementType.I32:
                return f"int32_vec{lanes}"
            case ElementType.I64:
                return f"int64_vec{lanes}"

    def get_sem_suffix(self, ty: Type) -> str:
        match ty:
            case Type.Arith:
                return ""
            case Type.TropMax:
                return "TropMax"
            case Type.TropMin:
                return "TropMin"

    # ------------------------------------------------------------------
    # Topological sort (post-order DFS, structural equality via __hash__)
    # ------------------------------------------------------------------

    def _all_children(self, expr) -> list:
        """Children to traverse for topo sort (kernel nodes excluded)."""
        match expr:
            case Convolve(k, sig):
                if isinstance(k, TVKernel):
                    _, _, signals = k.to_sparse_repr()
                    return [sig] + list(signals)
                return [sig]
            case Recurse(k, sig):
                if isinstance(k, TVKernel):
                    _, _, signals = k.to_sparse_repr()
                    return [sig] + list(signals)
                return [sig]
            case SNeg(a):
                return [a]
            case SignalExprBinOp(a, b):
                return [a, b]
            case ConvertLanes(_):
                return []  # inner handled by nested driver
            case Var() | Num() | Impulse():
                return []
            case Var2D() | RVar2D():
                return []
            case Repeater(_, _, _):
                return []  # inner handled by nested driver
            case Ith(sig2d, _):
                # Only traverse sig2d if it's not a RVar2D (those are leaves)
                if isinstance(sig2d, RVar2D):
                    return []
                return [sig2d]
            case VectorAt(a, _):
                return [a]
            case RecurseV(a, g):
                # Traverse constituent signals from SignalVector
                assert isinstance(g, SignalVector)
                return list(g.data)
            case ConvolveV(a, b):
                assert isinstance(b, SignalVector)
                return list(b.data)
            case _:
                return list(expr.children()) if hasattr(expr, "children") else []

    def _topo_sort(self, expr, visited: set, result: list) -> None:
        if id(expr) in visited:
            return
        # Use id for visited tracking, but structural eq for CSE in reg_map
        visited.add(id(expr))
        for child in self._all_children(expr):
            self._topo_sort(child, visited, result)
        result.append(expr)

    def topo_sort(self, expr) -> list:
        visited: set = set()
        result: list = []
        self._topo_sort(expr, visited, result)
        return result

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------

    def _ensure_ti_kernel(
        self,
        kernel: TIKernel,
        indices: List[int],
        values: List,
        taps: int,
        element_type: ElementType,
        ty: Type,
        lanes: int,
        fields: List[str],
        ctor_inits: List[str],
        run_stmts: List[str],
    ) -> Tuple[int, str, str, int]:
        """Return (k_base_reg, k_regs_var, idxs_var, max_index).

        Emits PushTIKernel field + run_stmt if not already cached.
        """
        cache_key = kernel  # structural equality via __hash__/__eq__
        if cache_key in self.kernel_cache:
            return self.kernel_cache[cache_key]

        k_base = self._alloc_reg(taps)
        elt_type_str = element_type.to_str()
        vec_type = self.get_vec_type(element_type, lanes)

        # k_regs array (consecutive starting at k_base)
        k_regs_var = self._new_var()
        k_regs_vals = ", ".join(str(k_base + i) for i in range(taps))
        self.prologue.append(f"constexpr static int {k_regs_var}[{taps}] = {{{k_regs_vals}}};")

        # values array
        vals_var = self._new_var()
        val_strs = ", ".join(element_type.val_to_str(v, ty) for v in values)
        self.prologue.append(f"constexpr static {elt_type_str} {vals_var}[{taps}] = {{{val_strs}}};")

        # indices array (for use by Convolve/Recurse)
        idxs_var = self._new_var()
        idxs_strs = ", ".join(map(str, indices))
        self.prologue.append(f"constexpr static int {idxs_var}[{taps}] = {{{idxs_strs}}};")

        # Emit PushTIKernel field
        fn = self._new_field()
        fields.append(f"PushTIKernel<{vec_type}, {taps}, {vals_var}, {k_regs_var}> {fn};")
        # PushTIKernel is default-constructible; no ctor_init needed
        run_stmts.append(f"{fn}.run(reg);")

        max_index = indices[-1] if indices else 0
        result = (k_base, k_regs_var, idxs_var, max_index)
        self.kernel_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Emit one node
    # ------------------------------------------------------------------

    def _emit_node(
        self,
        expr,
        all_vars: Dict[str, ElementType],
        all_vars2d: Dict[str, ElementType],
        outer_name: str,
        fields: List[str],
        ctor_inits: List[str],
        run_stmts: List[str],
        reset_stmts: List[str],
    ) -> int:
        """Process one topo-sorted node.  Returns its output register index."""
        # CSE: if already computed, return cached register
        if expr in self.reg_map:
            return self.reg_map[expr]

        assert expr.lanes is not None, f"lanes not set on {expr}"
        lanes = expr.lanes
        vec_type = self.get_vec_type(expr.element_type, lanes)
        elt_type_str = expr.element_type.to_str()

        match expr:
            # ----------------------------------------------------------
            # Leaf signals
            # ----------------------------------------------------------
            case Var(name):
                out_reg = self._alloc_reg(1)
                fn = self._new_field()
                fields.append(f"PushSignal1D<{vec_type}, {out_reg}> {fn};")
                ctor_inits.append(f"{fn}({name})")
                run_stmts.append(f"{fn}.run(reg);")

            case Num(value):
                out_reg = self._alloc_reg(1)
                val_str = expr.element_type.val_to_str(value, expr.ty)
                fn = self._new_field()
                fields.append(f"PushSignal1DConstant<{vec_type}, {out_reg}> {fn};")
                ctor_inits.append(f"{fn}({val_str})")
                run_stmts.append(f"{fn}.run(reg);")

            case Impulse(value):
                out_reg = self._alloc_reg(1)
                val_str = expr.element_type.val_to_str(value, expr.ty)
                zero_str = expr.element_type.val_to_str(expr.ty.zero(), expr.ty)
                fn = self._new_field()
                fields.append(f"PushSignal1DImpulse<{vec_type}, {out_reg}> {fn};")
                ctor_inits.append(f"{fn}({val_str}, {zero_str})")
                run_stmts.append(f"{fn}.run(reg);")

            case Var2D(name):
                assert expr.n_rows is not None, f"Var2D('{name}') has n_rows=None at codegen time."
                n_rows = expr.n_rows
                out_base = self._alloc_reg(n_rows)
                fn = self._new_field()
                zero_str = expr.element_type.val_to_str(expr.ty.zero(), expr.ty)
                fields.append(f"PushInputSignal2D<{vec_type}, {out_base}, {n_rows}> {fn};")
                ctor_inits.append(f"{fn}({name}, {name}_stride, {zero_str})")
                run_stmts.append(f"{fn}.run(reg);")
                reset_stmts.append(f"{fn}.reset_and_next_row();")
                # For Var2D, reg_map stores the *base* register; Ith extracts rows
                self.reg_map[expr] = out_base
                return out_base

            case RVar2D(name):
                # Pre-filled by enclosing PushRepeater; just record base reg
                if name not in self.rvar2d_map:
                    raise ValueError(f"RVar2D '{name}' used outside Repeater context")
                rv_base, _ = self.rvar2d_map[name]
                self.reg_map[expr] = rv_base
                return rv_base

            # ----------------------------------------------------------
            # Convolve
            # ----------------------------------------------------------
            case Convolve(k, sig):
                in_reg = self.reg_map[sig]
                out_reg = self._alloc_reg(1)
                suffix = self.get_sem_suffix(expr.ty)

                if isinstance(k, TIKernel):
                    taps, indices, values = k.to_sparse_repr()
                    if taps == 0:
                        # Zero kernel → constant zero output
                        fn = self._new_field()
                        zero_str = expr.element_type.val_to_str(expr.ty.zero(), expr.ty)
                        fields.append(f"PushSignal1DConstant<{vec_type}, {out_reg}> {fn};")
                        ctor_inits.append(f"{fn}({zero_str})")
                        run_stmts.append(f"{fn}.run(reg);")
                    else:
                        k_base, k_regs_var, idxs_var, max_index = self._ensure_ti_kernel(
                            k,
                            indices,
                            values,
                            taps,
                            expr.element_type,
                            expr.ty,
                            lanes,
                            fields,
                            ctor_inits,
                            run_stmts,
                        )
                        fn = self._new_field()
                        fields.append(
                            f"PushSConvolve{suffix}<{vec_type}, {out_reg}, {in_reg},"
                            f" {taps}, {max_index}, {idxs_var}, {k_regs_var}> {fn};"
                        )
                        run_stmts.append(f"{fn}.run(reg);")
                        reset_stmts.append(f"{fn}.reset_and_next_row();")

                elif isinstance(k, TVKernel):
                    taps, indices, signals = k.to_sparse_repr()
                    k_regs_list = [self.reg_map[s] for s in signals]
                    k_regs_var = self._new_var()
                    self.prologue.append(
                        f"constexpr static int {k_regs_var}[{taps}] = " f"{{{', '.join(map(str, k_regs_list))}}};"
                    )
                    idxs_var = self._new_var()
                    self.prologue.append(
                        f"constexpr static int {idxs_var}[{taps}] = " f"{{{', '.join(map(str, indices))}}};"
                    )
                    max_index = indices[-1] if indices else 0
                    fn = self._new_field()
                    fields.append(
                        f"PushSConvolve{suffix}<{vec_type}, {out_reg}, {in_reg},"
                        f" {taps}, {max_index}, {idxs_var}, {k_regs_var}> {fn};"
                    )
                    run_stmts.append(f"{fn}.run(reg);")
                    reset_stmts.append(f"{fn}.reset_and_next_row();")
                else:
                    raise NotImplementedError(f"Unsupported kernel type {type(k)}")

            # ----------------------------------------------------------
            # Recurse
            # ----------------------------------------------------------
            case Recurse(k, sig):
                # Strip leading time delay (must equal lanes for vectorization)
                time_delay, stripped_k = k.time_delay(lanes)
                assert time_delay == lanes, f"Recurse kernel time_delay={time_delay} ≠ lanes={lanes}"
                in_reg = self.reg_map[sig]
                out_reg = self._alloc_reg(1)
                suffix = self.get_sem_suffix(expr.ty)

                if isinstance(stripped_k, TIKernel):
                    taps, indices, values = stripped_k.to_sparse_repr()
                    if taps == 0:
                        # No feedback: output = input
                        fn = self._new_field()
                        # Simple copy via PushVectorAt
                        fields.append(f"PushVectorAt<{vec_type}, {out_reg}, {in_reg}> {fn};")
                        run_stmts.append(f"{fn}.run(reg);")
                    else:
                        # Don't use kernel_cache for stripped kernels to avoid
                        # clobbering a potentially different un-stripped version
                        k_base = self._alloc_reg(taps)
                        k_regs_var = self._new_var()
                        k_regs_vals = ", ".join(str(k_base + i) for i in range(taps))
                        self.prologue.append(f"constexpr static int {k_regs_var}[{taps}] = " f"{{{k_regs_vals}}};")
                        vals_var = self._new_var()
                        val_strs = ", ".join(expr.element_type.val_to_str(v, expr.ty) for v in values)
                        self.prologue.append(
                            f"constexpr static {elt_type_str} {vals_var}[{taps}] = " f"{{{val_strs}}};"
                        )
                        idxs_var = self._new_var()
                        idxs_strs = ", ".join(map(str, indices))
                        self.prologue.append(f"constexpr static int {idxs_var}[{taps}] = " f"{{{idxs_strs}}};")
                        max_index = indices[-1]

                        # PushTIKernel for the stripped kernel
                        fn_k = self._new_field()
                        fields.append(f"PushTIKernel<{vec_type}, {taps}, {vals_var}, " f"{k_regs_var}> {fn_k};")
                        run_stmts.append(f"{fn_k}.run(reg);")

                        fn = self._new_field()
                        fields.append(
                            f"PushSRecurse{suffix}<{vec_type}, {out_reg}, {in_reg},"
                            f" {taps}, {max_index}, {idxs_var}, {k_regs_var}> {fn};"
                        )
                        run_stmts.append(f"{fn}.run(reg);")
                        reset_stmts.append(f"{fn}.reset_and_next_row();")

                elif isinstance(stripped_k, TVKernel):
                    taps, indices, signals = stripped_k.to_sparse_repr()
                    k_regs_list = [self.reg_map[s] for s in signals]
                    k_regs_var = self._new_var()
                    self.prologue.append(
                        f"constexpr static int {k_regs_var}[{taps}] = " f"{{{', '.join(map(str, k_regs_list))}}};"
                    )
                    idxs_var = self._new_var()
                    self.prologue.append(
                        f"constexpr static int {idxs_var}[{taps}] = " f"{{{', '.join(map(str, indices))}}};"
                    )
                    max_index = indices[-1] if indices else 0
                    fn = self._new_field()
                    fields.append(
                        f"PushSRecurse{suffix}<{vec_type}, {out_reg}, {in_reg},"
                        f" {taps}, {max_index}, {idxs_var}, {k_regs_var}> {fn};"
                    )
                    run_stmts.append(f"{fn}.run(reg);")
                    reset_stmts.append(f"{fn}.reset_and_next_row();")
                else:
                    raise NotImplementedError(f"Unsupported stripped kernel type {type(stripped_k)}")

            # ----------------------------------------------------------
            # Binary / unary ops
            # ----------------------------------------------------------
            case SignalExprBinOp(a, b):
                in_reg_a = self.reg_map[a]
                in_reg_b = self.reg_map[b]
                out_reg = self._alloc_reg(1)
                suffix = self.get_sem_suffix(expr.ty)
                op_name = "Push" + type(expr).__name__ + suffix
                fn = self._new_field()
                fields.append(f"{op_name}<{vec_type}, {out_reg}, {in_reg_a}, {in_reg_b}> {fn};")
                run_stmts.append(f"{fn}.run(reg);")

            case SNeg(a):
                in_reg_a = self.reg_map[a]
                out_reg = self._alloc_reg(1)
                fn = self._new_field()
                fields.append(f"PushSNeg<{vec_type}, {out_reg}, {in_reg_a}> {fn};")
                run_stmts.append(f"{fn}.run(reg);")

            # ----------------------------------------------------------
            # ConvertLanes → nested inner driver
            # ----------------------------------------------------------
            case ConvertLanes(inner):
                inner_lanes = inner.lanes
                assert inner_lanes is not None
                out_reg = self._alloc_reg(1)

                # Generate inner driver for the sub-expression
                inner_driver_name = self._new_driver_name(outer_name)
                inner_cg = PushCodeGen()
                # Propagate rvar2d_map into inner so RVar2D is visible there too
                inner_cg.rvar2d_map = dict(self.rvar2d_map)
                inner_cg.driver_counter = self.driver_counter

                inner_code = inner_cg._generate_driver(inner, inner_driver_name, all_vars, all_vars2d)
                self.driver_counter = inner_cg.driver_counter
                inner_out_reg = inner_cg.reg_map[inner]

                # Prepend any nested inner structs
                self.inner_structs.extend(inner_cg.inner_structs)
                self.inner_structs.append(inner_code)

                vec_in = self.get_vec_type(expr.element_type, inner_lanes)
                vec_out = vec_type  # outer (target) lanes

                fn = self._new_field()
                if expr.lanes > inner_lanes:
                    # N narrow → 1 wide
                    fields.append(
                        f"PushConvertN2One<{vec_in}, {vec_out}, {out_reg},"
                        f" {inner_out_reg}, {inner_driver_name}> {fn};"
                    )
                    ctor_inits.append(f"{fn}({inner_driver_name}(" f"{_inner_ctor_args(all_vars, all_vars2d)}))")
                else:
                    # 1 wide → N narrow
                    fields.append(
                        f"PushConvertOne2N<{vec_in}, {vec_out}, {out_reg},"
                        f" {inner_out_reg}, {inner_driver_name}> {fn};"
                    )
                    ctor_inits.append(f"{fn}({inner_driver_name}(" f"{_inner_ctor_args(all_vars, all_vars2d)}))")

                run_stmts.append(f"{fn}.run(reg);")
                reset_stmts.append(f"{fn}.reset_and_next_row();")

            # ----------------------------------------------------------
            # Repeater → nested inner driver
            # ----------------------------------------------------------
            case Repeater(inner_expr, n_rows, prev_rows_var):
                assert n_rows is not None, f"Repeater has n_rows=None at codegen time."
                out_base = self._alloc_reg(n_rows)

                # Build nested inner driver with RVar2D context
                inner_driver_name = self._new_driver_name(outer_name)
                inner_cg = PushCodeGen()
                inner_cg.driver_counter = self.driver_counter
                # Propagate any outer rvar2d bindings
                inner_cg.rvar2d_map = dict(self.rvar2d_map)
                # Allocate rv_base_reg in the INNER reg space
                n_prev = n_rows - 1
                rv_base_reg = inner_cg._alloc_reg(n_prev)
                inner_cg.rvar2d_map[prev_rows_var.name] = (rv_base_reg, n_prev)

                inner_code = inner_cg._generate_driver(inner_expr, inner_driver_name, all_vars, all_vars2d)
                self.driver_counter = inner_cg.driver_counter
                inner_out_reg = inner_cg.reg_map[inner_expr]

                self.inner_structs.extend(inner_cg.inner_structs)
                self.inner_structs.append(inner_code)

                fn = self._new_field()
                fields.append(
                    f"PushRepeater<{vec_type}, {n_rows}, {out_base},"
                    f" {rv_base_reg}, {inner_out_reg}, {inner_driver_name}> {fn};"
                )
                ctor_inits.append(f"{fn}(row_size, {inner_driver_name}(" f"{_inner_ctor_args(all_vars, all_vars2d)}))")
                run_stmts.append(f"{fn}.run(reg);")
                reset_stmts.append(f"{fn}.reset_and_next_row();")
                self.reg_map[expr] = out_base
                return out_base

            # ----------------------------------------------------------
            # Ith: extract row from 2D signal
            # ----------------------------------------------------------
            case Ith(sig2d, i):
                if isinstance(sig2d, RVar2D):
                    rv_base, _ = self.rvar2d_map[sig2d.name]
                    out_reg = rv_base + i
                    # No node emitted; just return the pre-filled register
                    self.reg_map[expr] = out_reg
                    return out_reg
                else:
                    base_reg = self.reg_map[sig2d]
                    src = base_reg + i
                    if src == base_reg + i:
                        # Often the register is already at the right place;
                        # emit a copy to keep topo order explicit.
                        out_reg = self._alloc_reg(1)
                        fn = self._new_field()
                        fields.append(f"PushVectorAt<{vec_type}, {out_reg}, {src}> {fn};")
                        run_stmts.append(f"{fn}.run(reg);")
                    else:
                        out_reg = src

            # ----------------------------------------------------------
            # VectorAt(RecurseV, i) or VectorAt(ConvolveV, i)
            # ----------------------------------------------------------
            case VectorAt(vec_expr, i):
                if isinstance(vec_expr, RecurseV):
                    f_base = self.reg_map.get(vec_expr)
                    if f_base is None:
                        # First VectorAt for this RecurseV: generate it now
                        f_base = self._emit_recurse_v2(
                            vec_expr,
                            all_vars,
                            all_vars2d,
                            fields,
                            ctor_inits,
                            run_stmts,
                            reset_stmts,
                        )
                    out_reg = f_base + i
                    # No extra field: f_base+i is already the correct register
                    self.reg_map[expr] = out_reg
                    return out_reg
                elif isinstance(vec_expr, ConvolveV):
                    out_reg = self._emit_convolve_v_row(
                        vec_expr,
                        i,
                        all_vars,
                        all_vars2d,
                        fields,
                        ctor_inits,
                        run_stmts,
                        reset_stmts,
                    )
                else:
                    raise NotImplementedError(f"VectorAt of {type(vec_expr)}")

            case _:
                raise NotImplementedError(f"PushCodeGen: unhandled node type {type(expr)}: {expr}")

        self.reg_map[expr] = out_reg
        return out_reg

    # ------------------------------------------------------------------
    # RecurseV2 helper
    # ------------------------------------------------------------------

    def _emit_recurse_v2(
        self,
        expr: RecurseV,
        all_vars: Dict,
        all_vars2d: Dict,
        fields: List[str],
        ctor_inits: List[str],
        run_stmts: List[str],
        reset_stmts: List[str],
    ) -> int:
        """Emit PushSRecurseV2 for a RecurseV node.  Returns f_base register."""
        assert isinstance(expr.a, KernelMatrix)
        assert isinstance(expr.g, SignalVector)
        n = expr.a.n
        assert n == 2, f"PushCodeGen only supports RecurseV2 (n=2), got n={n}"

        lanes = expr.lanes
        assert lanes is not None
        vec_type = self.get_vec_type(expr.element_type, lanes)
        elt_type_str = expr.element_type.to_str()

        g0_reg = self.reg_map[expr.g.data[0]]
        g1_reg = self.reg_map[expr.g.data[1]]
        f_base = self._alloc_reg(2)
        f0_reg, f1_reg = f_base, f_base + 1

        def _process_kernel(k_raw: TIKernel, label: str) -> Tuple[int, int, str]:
            """Strip, allocate, emit TIKernel.  Returns (taps, kb, idxs_var)."""
            if len(k_raw.data) == 0:
                # Zero kernel
                return 0, 0, "nullptr"
            time_delay, stripped = k_raw.time_delay(lanes)
            assert time_delay == lanes, f"RecurseV kernel {label} time_delay={time_delay} ≠ lanes={lanes}"
            assert isinstance(stripped, TIKernel)
            taps, indices, values = stripped.to_sparse_repr()
            if taps == 0:
                return 0, 0, "nullptr"
            kb = self._alloc_reg(taps)
            vals_var = self._new_var()
            val_strs = ", ".join(expr.element_type.val_to_str(v, expr.ty) for v in values)
            self.prologue.append(f"constexpr static {elt_type_str} {vals_var}[{taps}] = {{{val_strs}}};")
            idxs_var = self._new_var()
            self.prologue.append(f"constexpr static int {idxs_var}[{taps}] = " f"{{{', '.join(map(str, indices))}}};")
            fn_k = self._new_field()
            k_regs_var = self._new_var()
            k_regs_vals = ", ".join(str(kb + i) for i in range(taps))
            self.prologue.append(f"constexpr static int {k_regs_var}[{taps}] = {{{k_regs_vals}}};")
            fields.append(f"PushTIKernel<{vec_type}, {taps}, {vals_var}, {k_regs_var}> {fn_k};")
            run_stmts.append(f"{fn_k}.run(reg);")
            max_index = indices[-1]
            return taps, kb, idxs_var, max_index  # type: ignore[return-value]

        r00 = _process_kernel(cast(TIKernel, expr.a.data[0][0]), "K00")
        r01 = _process_kernel(cast(TIKernel, expr.a.data[0][1]), "K01")
        r10 = _process_kernel(cast(TIKernel, expr.a.data[1][0]), "K10")
        r11 = _process_kernel(cast(TIKernel, expr.a.data[1][1]), "K11")

        def _fmt(res) -> Tuple[int, int, int, str]:
            if len(res) == 3:  # zero kernel
                return (0, 0, 0, "nullptr")
            taps, kb, idxs_var, max_index = res
            return taps, max_index, kb, idxs_var

        t00, mx00, kb00, ix00 = _fmt(r00)
        t01, mx01, kb01, ix01 = _fmt(r01)
        t10, mx10, kb10, ix10 = _fmt(r10)
        t11, mx11, kb11, ix11 = _fmt(r11)

        fn = self._new_field()
        fields.append(
            f"PushSRecurseV2<{vec_type},"
            f" {f0_reg}, {f1_reg}, {g0_reg}, {g1_reg},"
            f" {t00}, {mx00}, {kb00}, {ix00},"
            f" {t01}, {mx01}, {kb01}, {ix01},"
            f" {t10}, {mx10}, {kb10}, {ix10},"
            f" {t11}, {mx11}, {kb11}, {ix11}> {fn};"
        )
        run_stmts.append(f"{fn}.run(reg);")
        reset_stmts.append(f"{fn}.reset_and_next_row();")

        self.reg_map[expr] = f_base
        return f_base

    # ------------------------------------------------------------------
    # ConvolveV row helper (FIR vector row i)
    # ------------------------------------------------------------------

    def _emit_convolve_v_row(
        self,
        expr: ConvolveV,
        row: int,
        all_vars: Dict,
        all_vars2d: Dict,
        fields: List[str],
        ctor_inits: List[str],
        run_stmts: List[str],
        reset_stmts: List[str],
    ) -> int:
        """Emit PushSConvolve + PushSAdd for VectorAt(ConvolveV, row).

        Returns out_reg for the final accumulated output.
        """
        assert isinstance(expr.a, KernelMatrix)
        assert isinstance(expr.b, SignalVector)
        lanes = expr.lanes
        assert lanes is not None
        vec_type = self.get_vec_type(expr.element_type, lanes)
        n = expr.a.n

        term_regs: List[int] = []
        for j in range(n):
            k = cast(TIKernel, expr.a.data[row][j])
            sig = expr.b.data[j]
            if len(k.data) == 0:
                continue  # zero kernel: no contribution
            taps, indices, values = k.to_sparse_repr()
            if taps == 0:
                continue
            in_reg = self.reg_map[sig]
            k_base, k_regs_var, idxs_var, max_index = self._ensure_ti_kernel(
                k,
                indices,
                values,
                taps,
                expr.element_type,
                expr.ty,
                lanes,
                fields,
                ctor_inits,
                run_stmts,
            )
            out_r = self._alloc_reg(1)
            fn = self._new_field()
            fields.append(
                f"PushSConvolve<{vec_type}, {out_r}, {in_reg}," f" {taps}, {max_index}, {idxs_var}, {k_regs_var}> {fn};"
            )
            run_stmts.append(f"{fn}.run(reg);")
            reset_stmts.append(f"{fn}.reset_and_next_row();")
            term_regs.append(out_r)

        if not term_regs:
            # All zero: constant zero output
            out_reg = self._alloc_reg(1)
            zero_str = expr.element_type.val_to_str(expr.ty.zero(), expr.ty)
            fn = self._new_field()
            fields.append(f"PushSignal1DConstant<{vec_type}, {out_reg}> {fn};")
            ctor_inits.append(f"{fn}({zero_str})")
            run_stmts.append(f"{fn}.run(reg);")
            return out_reg

        acc_reg = term_regs[0]
        for r in term_regs[1:]:
            new_reg = self._alloc_reg(1)
            fn = self._new_field()
            fields.append(f"PushSAdd<{vec_type}, {new_reg}, {acc_reg}, {r}> {fn};")
            run_stmts.append(f"{fn}.run(reg);")
            acc_reg = new_reg
        return acc_reg

    # ------------------------------------------------------------------
    # Driver struct generation
    # ------------------------------------------------------------------

    def _generate_driver(
        self,
        expr,
        driver_name: str,
        all_vars: Dict[str, ElementType],
        all_vars2d: Dict[str, ElementType],
    ) -> str:
        """Generate a complete driver struct definition string.

        The struct has:
          - fields for each stateful node
          - ``using reg_type = Reg<vec_type, N>;``
          - constructor taking input pointers and row_size
          - ``void run(reg_type& reg)``
          - ``void reset_and_next_row()``
        """
        fields: List[str] = []
        ctor_inits: List[str] = []
        run_stmts: List[str] = []
        reset_stmts: List[str] = []

        topo = self.topo_sort(expr)
        for node in topo:
            self._emit_node(
                node,
                all_vars,
                all_vars2d,
                driver_name,
                fields,
                ctor_inits,
                run_stmts,
                reset_stmts,
            )

        out_reg = self.reg_map[expr]
        n_regs = self.reg_counter
        assert expr.lanes is not None
        vec_type = self.get_vec_type(expr.element_type, expr.lanes)

        # Build constructor argument list
        args_1d = [f"const {elt.to_str()} *{var}" for var, elt in sorted(all_vars.items())]
        args_2d = [f"const {elt.to_str()} *{var}, int {var}_stride" for var, elt in sorted(all_vars2d.items())]
        all_args = args_1d + args_2d + ["int row_size"]
        ctor_args_str = ", ".join(all_args)

        # Build initializer list (skip fields without ctor_inits)
        init_list = ", ".join(ctor_inits) if ctor_inits else ""

        lines: List[str] = []
        lines.append(f"struct {driver_name} {{")
        for f in fields:
            lines.append(f"    {f}")
        lines.append(f"    using reg_type = Reg<{vec_type}, {n_regs}>;")
        if init_list:
            lines.append(f"    {driver_name}({ctor_args_str}) : {init_list} {{}}")
        else:
            lines.append(f"    {driver_name}({ctor_args_str}) {{}}")
        lines.append(f"    inline void run(reg_type& reg) {{")
        for stmt in run_stmts:
            lines.append(f"        {stmt}")
        lines.append("    }")
        lines.append("    void reset_and_next_row() {")
        for stmt in reset_stmts:
            lines.append(f"        {stmt}")
        lines.append("    }")
        lines.append("};")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API: generate(expr, name) → Code
    # ------------------------------------------------------------------

    def generate(self, expr: SignalExpr | SignalExpr2D, name: str) -> Code:
        """Generate a complete push driver for *expr*, compatible with the
        pull-based runner infrastructure (make_{name} / {name}_vector_type).
        """
        self._reset()

        all_vars, all_vars2d = self.collect_input_variables(expr)

        driver_name = f"Driver_{name}"
        driver_code = self._generate_driver(expr, driver_name, all_vars, all_vars2d)

        out_reg = self.reg_map[expr]
        assert expr.lanes is not None
        vec_type = self.get_vec_type(expr.element_type, expr.lanes)
        n_regs = self.reg_counter

        # Build make_{name}() factory that returns a heap-allocated driver.
        # The driver's public run(vec_type*) overload wraps run(reg_type&).
        args_1d = [f"const {elt.to_str()} *{var}" for var, elt in sorted(all_vars.items())]
        args_2d = [f"const {elt.to_str()} *{var}, int {var}_stride" for var, elt in sorted(all_vars2d.items())]
        all_args = args_1d + args_2d + ["int row_size"]
        ctor_args_str = ", ".join(all_args)

        var_pass_1d = [var for var in sorted(all_vars.keys())]
        var_pass_2d = []
        for var in sorted(all_vars2d.keys()):
            var_pass_2d.extend([var, f"{var}_stride"])
        var_pass = ", ".join(var_pass_1d + var_pass_2d + ["row_size"])

        # Add run(vec_type*) public wrapper to the driver struct by extending it
        wrapper_lines: List[str] = []
        wrapper_lines.append(f"struct PubDriver_{name} : {driver_name} {{")
        wrapper_lines.append(f"    using {driver_name}::{driver_name};")
        wrapper_lines.append(f"    inline void run({vec_type}* out) {{")
        wrapper_lines.append(f"        reg_type reg;")
        wrapper_lines.append(f"        {driver_name}::run(reg);")
        wrapper_lines.append(f"        *out = reg[{out_reg}];")
        wrapper_lines.append("    }")
        wrapper_lines.append("};")

        namespace_name = f"push_{name}"

        # make_fn and type_alias live at the top level so callers don't need to
        # know the namespace; everything else (prologue arrays, inner structs,
        # driver structs) goes inside the per-kernel namespace to avoid name
        # collisions (v0, v1, … arrays) when multiple kernels are compiled together.
        make_fn = (
            f"auto make_{name}({ctor_args_str}) {{\n"
            f"    return new {namespace_name}::PubDriver_{name}({var_pass});\n"
            f"}}"
        )
        type_alias = f"using {name}_vector_type = {vec_type};"

        prologue_str = "\n".join(self.prologue)
        inner_str = "\n".join(self.inner_structs)
        namespace_content = "\n".join(
            filter(
                None,
                [
                    prologue_str,
                    inner_str,
                    driver_code,
                    "\n".join(wrapper_lines),
                ],
            )
        )
        full_text = "\n".join(
            filter(
                None,
                [
                    f"namespace {namespace_name} {{",
                    namespace_content,
                    f"}} // namespace {namespace_name}",
                    make_fn,
                    type_alias,
                ],
            )
        )

        return Code(full_text, expr.ty, expr.element_type, expr.lanes)

    # ------------------------------------------------------------------
    # collect_input_variables (mirrors CodeGen)
    # ------------------------------------------------------------------

    def collect_input_variables(
        self,
        expr,
        vars_1d: Optional[Dict[str, ElementType]] = None,
        vars_2d: Optional[Dict[str, ElementType]] = None,
    ) -> Tuple[Dict[str, ElementType], Dict[str, ElementType]]:
        if vars_1d is None:
            vars_1d = {}
        if vars_2d is None:
            vars_2d = {}

        match expr:
            case Var(name):
                vars_1d[name] = expr.element_type
            case Var2D(name, _):
                vars_2d[name] = expr.element_type
            case _:
                children = list(expr.children()) if hasattr(expr, "children") else []
                for child in children:
                    self.collect_input_variables(child, vars_1d, vars_2d)

        return vars_1d, vars_2d

    def clear(self) -> None:
        self._reset()


# ---------------------------------------------------------------------------
# Helper: build constructor argument string for inner drivers
# ---------------------------------------------------------------------------


def _inner_ctor_args(
    all_vars: Dict[str, ElementType],
    all_vars2d: Dict[str, ElementType],
) -> str:
    """Comma-separated list of variable names + strides for inner driver ctors."""
    args_1d = sorted(all_vars.keys())
    args_2d = []
    for var in sorted(all_vars2d.keys()):
        args_2d.extend([var, f"{var}_stride"])
    return ", ".join(args_1d + args_2d + ["row_size"])


# ---------------------------------------------------------------------------
# Push-specific instantiate / benchmark / executable helpers
# ---------------------------------------------------------------------------

_COMMON_TEMPLATE = files("vecrec.templates") / "common.h"


def push_instantiate_kernels(path: str, codes: List[Code]) -> None:
    # Inline common.h (replacing the #include directive) so the generated
    # header is self-contained and doesn't need -I pointing to the templates dir.
    push_text = PUSH_TEMPLATE.read_text()
    common_text = _COMMON_TEMPLATE.read_text()
    text = push_text.replace('#include "common.h"', common_text, 1)
    for code in codes:
        text += "\n" + code.text
    with open(path, "w") as f:
        f.write(text)


def generate_and_run_push_benchmark(
    codegen: PushCodeGen,
    exprs: Sequence[SignalExpr | SignalExpr2D],
    include_correctness_check: bool = True,
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
    if compiler_flags is None:
        compiler_flags = ["-std=c++20", "-O3", "-march=native", "-I", "."]

    if header_path is None:
        fd, header_path = tempfile.mkstemp(suffix=".h", prefix="push_header_", dir="/tmp")
        os.close(fd)

    if benchmark_path is None:
        fd, benchmark_path = tempfile.mkstemp(suffix=".cpp", prefix="push_bench_", dir="/tmp")
        os.close(fd)

    if executable_path is None:
        fd, executable_path = tempfile.mkstemp(prefix="push_bench_", dir="/tmp")
        os.close(fd)
    print(header_path, benchmark_path, executable_path)

    kernel_names = [f"k{i}" for i in range(len(exprs))]
    codes = [codegen.generate(expr, nm) for expr, nm in zip(exprs, kernel_names)]
    push_instantiate_kernels(header_path, codes)

    # Collect variables
    all_vars: Dict[str, ElementType] = {}
    all_vars2d: Dict[str, ElementType] = {}
    for expr in exprs:
        v, v2d = codegen.collect_input_variables(expr)
        all_vars.update(v)
        all_vars2d.update(v2d)

    benchmark_code = _generate_benchmark_code(
        all_vars,
        all_vars2d,
        exprs[0].element_type,
        kernel_names,
        header_path,
        include_correctness_check,
        correctness_tolerance,
        input_size,
        warmup_iterations,
        benchmark_iterations,
        row_size,
    )
    with open(benchmark_path, "w") as f:
        f.write(benchmark_code)

    compile_cmd = [compiler] + compiler_flags + [benchmark_path, "-o", executable_path]
    print(compile_cmd)
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if compile_result.returncode != 0:
        return {
            "return_code": compile_result.returncode,
            "output": "",
            "error": compile_result.stderr,
        }

    run_result = subprocess.run([executable_path], capture_output=True, text=True)
    return {
        "return_code": run_result.returncode,
        "output": run_result.stdout,
        "error": run_result.stderr,
    }


def generate_push_kernel_executable(
    codegen: PushCodeGen,
    expr: SignalExpr | SignalExpr2D,
    header_path: Optional[str] = None,
    runner_path: Optional[str] = None,
    executable_path: Optional[str] = None,
    compiler: str = "clang++",
    compiler_flags: Optional[List[str]] = None,
    row_size: int = 1024,
) -> KernelExecutable:
    if compiler_flags is None:
        compiler_flags = ["-std=c++20", "-O3", "-march=native", "-I", "."]

    if header_path is None:
        fd, header_path = tempfile.mkstemp(suffix=".h", prefix="push_header_", dir="/tmp")
        os.close(fd)

    if runner_path is None:
        fd, runner_path = tempfile.mkstemp(suffix=".cpp", prefix="push_runner_", dir="/tmp")
        os.close(fd)

    if executable_path is None:
        fd, executable_path = tempfile.mkstemp(prefix="push_runner_", dir="/tmp")
        os.close(fd)
    print(runner_path, executable_path)

    kernel_name = "k0"
    code = codegen.generate(expr, kernel_name)
    push_instantiate_kernels(header_path, [code])

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
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if compile_result.returncode != 0:
        raise RuntimeError(f"Push codegen compilation failed:\n{compile_result.stderr}")

    return KernelExecutable(
        executable_path=executable_path,
        variable_order=sorted(all_vars.keys()),
        variable_types=all_vars,
        vars2d_info=all_vars2d,
        output_type=expr.element_type,
        row_size=row_size,
    )
