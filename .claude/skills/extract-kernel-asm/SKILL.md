---
name: extract-kernel-asm
description: >
  Extract the generated C++ and x86 assembly for a single VecRec kernel schedule,
  so you can see how a recurrence actually lowers (SIMD vs scalar, spills, shuffles).
  Use when a benchmark is slower than expected, when you want to confirm a transform
  vectorized (or failed to vectorize) a recurrence, when diagnosing register spilling
  / SROA, or whenever someone asks "what does the generated code/assembly look like".
  Works for any benchmark in benchmarks/run_all.py REGISTRY and any candidate schedule.
---

# Extract a kernel's assembly

VecRec lowers a scheduled expression to a C++ struct `k0` with an always-inlined
`run()` that emits `ele_per_iter * rows_per_iter` output cells per call. To see how
a schedule really performs you read the assembly of that `run()` body. This skill
generates a **self-contained** header + an isolated probe and dumps the assembly.

## The one rule that matters: don't defeat SROA

The scratch "register file" (`Reg<vec_type, N>`) is per-`run()` scratch. The compiler
keeps it in registers only when it can prove the storage is dead between loop
iterations. Two ways to ruin that — avoid both:

1. **Never isolate the kernel by reference.** A `noinline void run_one(k0 &k, …)`
   reaches the kernel through a pointer arg, so its register file is *forced* to
   memory and every slot becomes a load/store. The assembly then looks catastrophic
   even when the real driver is fine. Always construct `kernel` as a **local** inside
   the isolated function (the helper does this).
2. Be aware that whether `Reg` is a struct member vs a `run()`-local changes whether
   SROA fires (members of the loop-carried kernel object spill for large kernels).
   The helper mirrors the real driver structure, so what you read is representative.

## Quick start

```bash
# fully vectorized schedule of affine-gap, tuned epi/rpi:
uv run python .claude/skills/extract-kernel-asm/extract_asm.py affine-gap \
    --vectorized --epi-mult 4 --rpi 2
```

This prints the chosen schedule, its `ele_per_iter`/`rows_per_iter`, the widest
recurrence lane count, and a `bench()` instruction-mix summary, and writes to
`scratch/asm/`:

- `<stem>.h`         — kernel header (`common.h` inlined; compiles standalone)
- `<stem>_probe.cpp` — isolates the kernel in one `noinline bench()` (SROA-safe)
- `<stem>.s`         — Intel-syntax assembly

## Selecting the schedule

`build_schedules()` returns the candidate schedules (index 0 is the optimizer's
reference/least-transformed). Pick one with:

- `--index N`     explicit index
- `--vectorized`  widest recurrence lanes (the most-dilated/SIMD form)
- `--scalar`      narrowest (often `vec1`, a serial scalar recurrence)
- `--match STR`   first schedule whose `pp()` contains `STR`
  (e.g. `--match 'rec([-∞, -∞, -2]'` for a specific dilation)

Tune codegen with `--epi-mult M` (ele_per_iter multiple) and `--rpi K`
(rows_per_iter), matching the config the sweep picked for that benchmark.

## Reading the output

Look at the `bench()` instruction mix:

- **`vpmaxsd`/`vpminsd` + `vpaddd`** — vectorized tropical (max/min-plus) work. Good.
- **`cmp` + `cmovg`/`cmovl`** in a long run — a **scalar** recurrence (a `vecN=vec1`
  serial dependency chain). The recurrence wasn't vectorized in this schedule.
- **`valignd`/`valignq`/`vshufi64x2`/`vperm*`** — cross-lane shuffles, usually the
  parallel-prefix (Kogge-Stone) scan of a *dilated* recurrence.
- **High `mov`/`vmovdqa64`/`vmovaps`/`vmovups` fraction (≳50%)** — register spilling;
  the live working set exceeds the physical registers. Check `sizeof(k0)` and the
  `main`/`bench` stack frame (`sub rsp, N`).

To inspect by hand:

```bash
S=scratch/asm/<stem>.s
# bench() is name-mangled in the asm; find its label first, then isolate the body.
LBL=$(grep -oE '^_Z[0-9]*benchv?:' "$S" | head -1 | tr -d ':')
awk -v L="$LBL:" '$0 ~ "^"L{f=1} f{print} /\.cfi_endproc/{if(f)exit}' "$S" \
  | grep -oE '^\s+[a-z][a-z0-9]+' | sort | uniq -c | sort -rn | head -25

# scalar-recurrence chain? (a long run of these == a vecN=vec1 serial recurrence)
awk -v L="$LBL:" '$0 ~ "^"L{f=1} f{print} /\.cfi_endproc/{if(f)exit}' "$S" \
  | grep -cE '^\s+(cmp|cmovg|cmovl)\b'

# stack frame (spill pressure) of the real driver, if you generated one:
awk '/^main:/{f=1} f && /sub\trsp,/{print $3; exit}' <driver>.s
```

For a *representative full driver* (the `now()`-bracketed timed loop) rather than the
isolated probe, generate it via `vecrec.codegen.generate_kernel._generate_benchmark_code`
(see `benchmarks/benchmark.py` → `generate_and_run_benchmark`) and dump `main`; the
hot loop is the inlined `kernel.run` between the two `chrono::now()` calls.

## Compare two schedules

Generate scalar and vectorized forms and diff their mixes — this is how you tell
whether a transform vectorized a recurrence and whether it was worth it:

```bash
uv run python .claude/skills/extract-kernel-asm/extract_asm.py BENCH --scalar     --epi-mult 4 --rpi 2
uv run python .claude/skills/extract-kernel-asm/extract_asm.py BENCH --vectorized --epi-mult 4 --rpi 2
```
