#!/usr/bin/env python3
"""Extract the generated C++ and assembly for one VecRec kernel schedule.

Given a benchmark name and a schedule selector, this emits three self-contained
files into an output dir:
  <stem>.h          - the kernel header (common.h inlined; fully standalone)
  <stem>_probe.cpp  - a tiny harness that isolates the kernel in ONE noinline
                      function `bench()`, so its assembly is readable, while
                      keeping `kernel` a LOCAL (so SROA/regalloc behave exactly
                      as in the real driver -- see the SROA note below).
  <stem>.s          - the assembly (Intel syntax), produced by the printed
                      clang++ command (this script also runs it).

It prints the per-recurrence SIMD lane widths (so you can confirm a schedule is
scalar vs vectorized), a clang++ command, and an instruction-mix summary of
`bench`.

CRITICAL (SROA note): do NOT isolate a kernel by passing it BY REFERENCE into a
noinline function (`void run_one(k0 &k, ...)`). Reached through a pointer arg, the
kernel's register file can never be SROA-promoted, so the assembly shows spurious
load/store traffic that the real (inlined-into-the-loop) driver does not have.
This harness constructs `kernel` as a local inside `bench()` and mirrors the
driver's row loop (reset_and_next_row per row), so what you read is what runs.

Usage:
  uv run python .claude/skills/extract-kernel-asm/extract_asm.py BENCH [options]

  BENCH                substring of a registered benchmark name (run_all.py)
  --index N            schedule index (from build_schedules()); default 0
  --vectorized         pick the schedule with the WIDEST recurrence lanes
  --scalar             pick the schedule with the NARROWEST recurrence lanes
  --match SUBSTR       pick the first schedule whose pp() contains SUBSTR
  --epi-mult M         CodeGen ele_per_iter_multiple (default 2)
  --rpi K              CodeGen rows_per_iter (default 1)
  --out DIR            output dir (default: scratch/asm)
  --no-asm             only emit .h/.cpp, skip compiling to .s
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Make the benchmarks package importable (Benchmark, the registry).
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

from vecrec.codegen import CodeGen  # noqa: E402
from vecrec.codegen.generate_kernel import instantiate_kernels  # noqa: E402
from vecrec.expr.pretty import pp  # noqa: E402
from run_all import REGISTRY  # noqa: E402


def find_benchmark(name_substr: str):
    matches = [b for b in REGISTRY if name_substr in b.name]
    if not matches:
        sys.exit(f"No benchmark matches {name_substr!r}. Try: " + ", ".join(b.name for b in REGISTRY))
    if len(matches) > 1 and not any(b.name == name_substr for b in matches):
        print(f"warning: {name_substr!r} matches {[b.name for b in matches]}; using {matches[0].name}")
    return next((b for b in matches if b.name == name_substr), matches[0])


def max_recurrence_lanes(code_text: str) -> int:
    """Largest SIMD lane count among the kernel's SRecurse* instantiations."""
    widths = [int(m) for m in re.findall(r"SRecurse\w*<\s*\w+?vec(\d+)", code_text)]
    return max(widths) if widths else 0


def pick_schedule(bench, args) -> int:
    scheds = bench.build_schedules()
    if args.match is not None:
        for i, s in enumerate(scheds):
            if args.match in pp(s):
                return i
        sys.exit(f"No schedule pp() contains {args.match!r}")
    if args.vectorized or args.scalar:
        # Rank by widest recurrence lanes (needs a quick codegen pass per schedule).
        lanes = []
        for s in scheds:
            code = CodeGen(ele_per_iter_multiple=args.epi_mult, rows_per_iter=args.rpi).generate(s, "k0")
            lanes.append(max_recurrence_lanes(code.text))
        return int(max(range(len(lanes)), key=lambda i: lanes[i]) if args.vectorized
                   else min(range(len(lanes)), key=lambda i: lanes[i]))
    return args.index


def build_probe(code, out_type_cpp: str, all_vars, all_vars2d) -> str:
    """A noinline bench() that constructs the kernel as a LOCAL and mirrors the
    driver's row loop. Handles ComputedSignal2D (its setup_code allocates the raw
    buffers) and materialized 1D/2D Var inputs."""
    epi = code.element_per_iter
    rpi = code.rows_per_iter
    row_size = epi * 16              # multiple of epi; >= a real row
    rows = rpi * 64                  # row-blocks processed per rep
    reps = 40                        # enough run() calls for a steady-state body

    # Kernel constructor args, in codegen order: computed signals, 1D vars, 2D vars, row_size.
    ctor = [f"{d.resolved_type}({d.make_args})" for d in code.computed_signals]
    ctor += [f"{v}.data()" for v in sorted(all_vars)]
    ctor += [f"{v}.data(), row_size" for v in sorted(all_vars2d)]
    ctor.append("row_size")

    L = []
    L.append('#include "%s"' % "KERNEL_HEADER")
    L += ["#include <cstdint>", "#include <vector>", "#include <cstdio>", "#include <x86intrin.h>", ""]
    # `N` sizes the raw buffers (a few rows of slack past what the loop processes,
    # so a block_row>0 computed signal can read one row ahead without overrunning).
    L.append(f"static const int row_size = {row_size};")
    L.append(f"static const int N = {rows + 4} * row_size;")
    L.append("")
    L.append("__attribute__((noinline)) long bench() {")
    # ComputedSignal2D setup (allocates e.g. s1/s2 from N, row_size). Emit once each.
    seen: set = set()
    for d in code.computed_signals:
        if d.setup_code not in seen:
            seen.add(d.setup_code)
            for line in d.setup_code.splitlines():
                L.append("    " + line)
    # Materialized var buffers.
    for v, et in sorted(all_vars.items()):
        L.append(f"    std::vector<{et.to_str()}> {v}(N, ({et.to_str()})1);")
    for v, et in sorted(all_vars2d.items()):
        L.append(f"    std::vector<{et.to_str()}> {v}(N, ({et.to_str()})1);")
    L.append(f"    const int N_proc = {rows} * row_size;")
    L.append("    long acc = 0;")
    L.append(f"    for (int rep = 0; rep < {reps}; rep++) {{")
    L.append(f"        auto kernel = k0({', '.join(ctor)});")
    L.append("        using elt_type = typename ElementType<k0_vector_type>::type;")
    L.append("        constexpr int vec_lanes = sizeof(k0_vector_type) / sizeof(elt_type);")
    L.append(f"        k0_vector_type ob[{rpi}][{epi}/vec_lanes];")
    L.append(f"        k0_vector_type* o[{rpi}] = {{ {', '.join('ob[%d]' % r for r in range(rpi))} }};")
    L.append("        for (int rb = 0; rb < N_proc; rb += %d * row_size) {" % rpi)
    L.append(f"            for (int col = 0; col < row_size; col += {epi}) {{")
    L.append("                kernel.run(o);")
    L.append("                acc ^= (long)ob[0][0][0];")
    L.append("            }")
    L.append(f"            for (int r = 0; r < {rpi}; r++) kernel.reset_and_next_row();")
    L.append("        }")
    L.append("    }")
    L.append("    return acc;")
    L.append("}")
    L.append("")
    L.append("int main() { volatile long x = bench(); return (int)x; }")
    return "\n".join(L)


def instr_mix(asm_path: str, fn: str = "bench") -> list[tuple[int, str]]:
    with open(asm_path) as f:
        lines = f.readlines()
    body, inside = [], False
    for ln in lines:
        if re.match(rf"^{fn}:|^_?Z\d*{fn}", ln) or ln.strip().startswith(fn + ":"):
            inside = True
        if inside:
            body.append(ln)
            if re.match(r"^\s*\.cfi_endproc", ln):
                break
    counts: dict[str, int] = {}
    for ln in body:
        m = re.match(r"^\s+([a-z][a-z0-9]+)\b", ln)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return sorted(((c, op) for op, c in counts.items()), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--vectorized", action="store_true")
    ap.add_argument("--scalar", action="store_true")
    ap.add_argument("--match", default=None)
    ap.add_argument("--epi-mult", type=int, default=2)
    ap.add_argument("--rpi", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "asm"))
    ap.add_argument("--no-asm", action="store_true")
    args = ap.parse_args()

    bench = find_benchmark(args.bench)
    idx = pick_schedule(bench, args)
    scheds = bench.build_schedules()
    expr = scheds[idx]

    cg = CodeGen(ele_per_iter_multiple=args.epi_mult, rows_per_iter=args.rpi)
    code = cg.generate(expr, "k0")
    all_vars, all_vars2d = cg.collect_input_variables(expr)

    os.makedirs(args.out, exist_ok=True)
    stem = f"{bench.name}_s{idx}_e{args.epi_mult}_r{args.rpi}"
    hdr = os.path.join(args.out, stem + ".h")
    cpp = os.path.join(args.out, stem + "_probe.cpp")
    asm = os.path.join(args.out, stem + ".s")

    instantiate_kernels(hdr, [code])
    probe = build_probe(code, expr.element_type.to_str(), all_vars, all_vars2d).replace(
        "KERNEL_HEADER", os.path.basename(hdr))
    with open(cpp, "w") as f:
        f.write(probe)

    print(f"benchmark      : {bench.name}")
    print(f"schedule       : [{idx}] of {len(scheds)}")
    print(f"  {pp(expr)}")
    print(f"ele_per_iter   : {code.element_per_iter}   rows_per_iter: {code.rows_per_iter}")
    print(f"recurrence lanes (widest SRecurse): {max_recurrence_lanes(code.text)}")
    print(f"header         : {hdr}")
    print(f"probe          : {cpp}")

    compile_cmd = f"clang++ -std=c++20 -O3 -march=native -S -masm=intel {cpp} -o {asm}"
    print(f"asm command    : {compile_cmd}")
    if args.no_asm:
        return
    rc = os.system(compile_cmd)
    if rc != 0:
        sys.exit(f"compile failed (rc={rc})")
    print(f"assembly       : {asm}")
    mix = instr_mix(asm, "bench")
    total = sum(c for c, _ in mix)
    moves = sum(c for c, op in mix if op.startswith(("mov", "vmov")))
    print(f"\nbench() instruction mix (total={total}, data-moves={moves} = {100*moves//max(total,1)}%):")
    for c, op in mix[:18]:
        print(f"  {c:5d}  {op}")


if __name__ == "__main__":
    main()
