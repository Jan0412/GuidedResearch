# Multi-output grading fix and a prompt delta layer

**Date:** 2026-08-18
**Target branch:** `prompt-deltas`, off `prm-labeling`
**Trees touched:** `GuidedResearch` (prompt layer), `KernelBench` (`scripts/` only)

## Why

Level-6 correctness sits at ~19.6% (gpt-oss-120b round 0, 13,704/69,748) and the ORM's
usable training signal is limited by
how few correct, speed-differentiated kernels exist per problem. Two causes were
measured, and neither is the generator's competence.

**1. Multi-output problems are ungradeable.** 1,726 of 17,437 level-6 problems (9.9%)
have a `forward()` returning a tuple. `eval.py:784` compares `output.shape !=
output_new.shape`, which raises `AttributeError` on a tuple. A correct kernel cannot
pass. The top error on these problems is `'tuple' object has no attribute 'shape'`
(24.5%).

**2. The prompt withholds information the model needs.** 86% of the prompt is stock
KernelBench text that cannot express level-6-specific facts (resolved input shapes) or
harness-specific constraints (the FP32 tolerance).

Measured correctness, `gpt-oss-120b`, level 6 round 0:

| group | correct | slots | rate |
|---|---|---|---|
| single-output, no dropout | 13,443 | 57,188 | **23.5%** |
| single-output + dropout | 195 | 5,656 | 3.4% |
| multi-output, no dropout | 66 | 5,556 | **1.2%** |
| multi-output + dropout | 0 | 1,348 | 0.0% |

Fixing multi-output grading recovers the 5,556-slot cell: ~1,306 correct at the healthy
rate versus 66 today, so **≈ +1,240 correct kernels per existing run (+9%)** — from
compute already spent.

## Scope

**In:** tuple-output grading normalisation; a prompt delta layer with four additive
deltas; a five-arm A/B on level 6 problems 0–500.

**Out, deliberately:**

- **The Dropout / missing `.eval()` bug.** `.eval()` is never called, and there is no
  re-seed between the reference and candidate forward passes (`eval.py:764-782`), so
  dropout masks differ and 94.7% of generated kernels for those problems retain
  dropout. This costs ~20 points on 1,751 problems (10.0%). It is left alone on purpose:
  official KernelBench exercises this path too (16 of 250 problems across levels 2–3),
  so changing it would diverge from the benchmark. Multi-output has no such defence —
  **zero** official problems return a tuple, so that path was never defined upstream.
- **Prompt deltas without measured evidence**: swapping the one-shot example, imposing
  plan structure, and resolving the plan/"no other text" instruction conflict. All three
  require replacing rather than appending text, and none has measured support. A
  variant worth a second round is **appending a second worked example** — a masked
  reduction with real strides and a 2D grid — since appending stays inside the delta
  framework where replacing does not; it is held back because it would compete with
  `pitfalls` for the same failure bucket and muddy attribution. The
  instruction conflict was verified harmless (0% of attempts lack a plan, 0% emit test
  code or trailing prose) because the two-pass sampler's `## Plan` prefill makes each
  instruction govern a disjoint span.
- **Rewriting the eval harness.** 175 GB / 950k files of existing labels depend on it,
  and eval bugs are silent rather than loud. Audit instead if confidence is needed.

## Design boundary

The KernelBench clone (pinned `423217d`) has an existing invariant worth making
explicit and preserving:

- **`src/kernelbench/` is never modified.** Upstream, byte-for-byte.
- **`scripts/` and `pyproject.toml` are ours.** All three existing local modifications
  live there, including `scripts/eval_from_generations.py`, the file that calls
  `eval_kernel_against_ref`.

The tuple fix therefore lives in the `scripts/` driver layer. The prompt layer lives
entirely in `GuidedResearch/kernel_gen/`. No cross-tree imports — the two trees have
separate venvs, so the normaliser is self-contained.

## Part A — tuple-output normalisation

`eval_kernel_against_ref(original_model_src: str, custom_model_src: str, ...)` takes
both models as **source strings**, and our driver supplies both. So normalisation
happens before KernelBench sees anything.

No AST rewriting and no regex surgery: we **append** a wrapper.

To `original_model_src`:

```python
_KB_INNER = Model
class Model(_KB_INNER):
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out
```

To `custom_model_src`, identical with `ModelNew`.

Properties this shape buys:

- **Append-only**, so there is no way to mangle the reference.
- **Self-disabling.** The `isinstance` check makes it a no-op when `forward` already
  returns a tensor, so it applies unconditionally to every problem and never needs to
  detect which ones are multi-output — removing a whole class of detection bugs.
- **Symmetric.** Identical logic on both sides; it cannot favour the candidate.
- Relies on last-definition-wins, the same mechanism the converter's footer already uses
  for its duplicate `get_init_inputs()`.

**Semantics, fixed in writing:** correctness is judged on the **primary (first) output**.
Secondary outputs — attention weights, intermediate states — are not compared. This is a
benchmark convention for a case upstream never defined, applied identically to both sides.

**Edge cases:** 1-element tuples (`return (x,)`, 8 problems) collapse to the element;
lists are treated as tuples; `forward` signatures taking kwargs pass through.

Behind a driver flag, default on, with the old behaviour reachable for comparison.

## Part B — the prompt delta layer

`build_base_prompt()` gains one parameter; the underlying call is otherwise unchanged:

```python
def build_base_prompt(problem, backend="triton", option="one_shot",
                      include_hardware=False, gpu_name=None,
                      deltas: frozenset[str] = frozenset()) -> str:
```

All three deltas are **additive or a flag**, so no vendoring of `prompts.toml` and no
string surgery are needed, and byte-identity is structural rather than argued.

### `contract`

An appended block, derived statically from `checker.lint.shapes.shapes_from_source`
(already a dependency of `critics.py`) plus the last `get_init_inputs()`. Coverage on
level 6: shapes resolve for **99.7%** of problems, and the final `get_init_inputs()` is
a literal in **100%**. Nothing is executed.

```
## Input contract
ModelNew is constructed as ModelNew(48) and called with 3 positional inputs,
all on CUDA and contiguous:
  arg 1: float32, shape (48, 48, 48, 48)
  arg 2: float32, shape (48, 48, 48, 48)
  arg 3: float32, shape (48, 48, 48, 48)
This supersedes any shape stated in comments or docstrings above.
```

Evidence: **100%** of level-6 files define `get_init_inputs()` twice (the converter
footer silently overrides the original); `get_inputs()` canonically returns
`[48,48,48,48]` while original docstrings describe 2D/3D tensors; the 3.1% of problems
whose `forward()` unpacks a conflicting rank score **12.3% versus 19.9%**. The closing
line is what resolves those contradictions.

### `precision`

A static appended block stating: correctness is checked against a true-FP32 reference at
tolerance **1e-4**; Triton's `tl.dot` defaults to **TF32** (~1e-3 error) and will fail;
`input_precision="ieee"` is required on float32 inputs; cuBLAS is already near-optimal
for FP32 GEMM, so fusing around `torch.matmul` beats reimplementing it.

Evidence: tolerance is 1e-4 (`eval.py:95`); torch runs `allow_tf32=False`,
`float32_matmul_precision=highest`; Triton 3.6 documents tf32 as the fp32 default.
`tl.dot` appears in **2.4%** of correct kernels but **54.1%** of failures in the
1e-4..1e-2 band — a 22x enrichment. `tl.dot` kernels are 26.2% (gpt-oss) / 29.7%
(DeepSeek) of all attempts at **1.9% / 2.7%** correct.

Honest bound: 82% of `tl.dot` kernels fail at compile/runtime before precision is
reached, and the 104 kernels that already set `ieee` were 0% correct (84.6% of them died
at compile/runtime). TF32 is necessary but not sufficient. This is why it is an A/B arm
rather than an assumption. It targets speed specifically: replacing matmul yields p90
speedup **2.219** versus **1.333** when kept.

### `hardware`

No new text — passes `include_hardware=True, gpu_name="H100"` to the existing call,
injecting KernelBench's own H100 specs and best-practice bullets. Currently off, so the
model is never told what GPU it targets.

### `pitfalls`

A static appended block of six rules for Triton 3.6, each tied to a measured failure
class rather than to general good practice:

1. Define `ModelNew` as a complete `nn.Module` and finish the class — **881 failures**
   (37.4% of the CUDA/compile bucket) are `module 'temp_module' has no attribute
   'ModelNew'`.
2. Mask every `tl.load` / `tl.store` unless the extent divides the block size exactly —
   **558 failures** (23.7% of that bucket) are illegal memory accesses.
3. Keep the grid lambda and the kernel signature consistent: a block size read as
   `meta["BLOCK_SIZE"]` must be declared `BLOCK_SIZE: tl.constexpr` and passed by
   keyword — **785 failures** across `'Keyword argument BLOCK_SIZE was specified but
   unrecognised'` (452), `dynamic_func() got multiple values` (193) and
   `missing 1 required positional argument` (140).
4. Never call a `@triton.jit` function from Python; launch it with `kernel[grid](...)` —
   **362 failures**.
5. Import everything used (`torch.nn as nn`, `typing.Optional`) — ~90 `NameError`s.
6. Target Triton 3.6 only; do not invent builtins — e.g. `tl.constant` does not exist
   (80), `'dtype' object is not callable` (131).

**Honest bound.** These classes total roughly 4-6% of failures, so the plausible gain is
**1-3 points** — close to the 2.7pt detection floor. Two facts cap what any rule list can
do: 10.8% of runtime errors are stored **truncated** and their message is unrecoverable,
and the error tail is extremely long (14,594 distinct shapes; the top 25 cover only
37.5%). Most failures are idiosyncratic logic errors, not repeated API misuse.

An earlier draft of this delta was sized off a regex that matched the source lines Triton
embeds in its error text rather than the messages, which overstated the constexpr and
builtin classes by roughly 5x. The counts above come from extracting the message after
the caret line.

### Invariants

- **`deltas=frozenset()` reproduces today's prompt byte-for-byte**, pinned by a test.
  This is what licenses reusing existing runs as arm 0.
- **Deterministic ordering** — iterate a fixed tuple, never a set. Set iteration order
  varies per process; this exact mistake made the notebook's §3b jitter irreproducible.
- **Graceful degradation** — if `shapes_from_source` returns `[]` (0.3%), the contract
  block is omitted rather than half-rendered.
- Surfaced as `--prompt-deltas contract,precision` — a comma string, per the `cli.py`
  rule that argparse dests are YAML keys and `nargs` is forbidden. Because
  `artifacts.write_config` persists `dict(vars(args))`, every run's
  `generation_config.yaml` records which deltas produced it.

## The A/B rollout

**Level 6, problem ids 0–500, 4 samples, `gpt-oss-120b`.** The id range spans 0–500
inclusive but contains exactly 500 problems — one id in that range has no file, which is
expected and needs no special handling. The slice is 500 **distinct** sources (no
byte-identical duplicates, unlike the corpus as a whole), 51 multi-output (10.2%), 40
with dropout (8.0%).

| arm | deltas | generate? | kernels |
|---|---|---|---|
| 0 | *(stock)* | **no — reuse existing round 0** | 2,000 (re-grade only) |
| 1 | `contract` | yes | 2,000 |
| 2 | `precision` | yes | 2,000 |
| 3 | `hardware` | yes | 2,000 |
| 4 | `pitfalls` | yes | 2,000 |

**Arm 0 is free.** The existing kb6 runs already cover all 500 problems at exactly 4
samples, round 0 — gpt-oss 410/2,000 (20.5%), DeepSeek 403/2,000 (20.2%). Round 0 of a
lintloop run is an unrefined generation with no feedback in existence, which is exactly
the stock arm. **Precondition:** the byte-identity test must pass; if it fails, arm 0
must be regenerated. Arm 0 is re-graded under the new driver so it is internally
consistent with the delta arms.

- Generation: 4 arms × 2,000 completions, `--rounds 1`.
- Eval: 10,000 kernels × ~6.4 GPU-s ≈ **18 GPU-hours**, of which arm 0's re-grade is ~3.6.
- Generate all 500 contiguously; **exclude the 40 dropout problems at analysis time** —
  they cannot respond to any delta, and excluding them at generation would need an
  unwieldy 460-id list.
- Power: 460 usable problems × 4 = 1,840 slots/arm at p≈0.22 gives SE of a between-arm
  difference ≈1.36pt, so 2σ ≈ **2.7pt absolute** (~+12% relative). Adequate for
  `contract` and `precision`; likely **not** adequate for `hardware`, and marginal for
  `pitfalls` whose expected effect is 1-3pt. A null result on arms 3 or 4 must be
  recorded as inconclusive, not as "does not help".
- **Multiple comparisons.** Four delta arms are each tested against arm 0, so at α=0.05
  per test the family-wise false-positive rate is ~18.5%. Treat a single arm clearing
  2σ as a candidate, not a conclusion; confirm it on the free DeepSeek stock baseline
  before adopting it into the production prompt.
- Same problem set across all arms, so comparison is **paired per problem**. Pairing is
  at problem level only: a changed prompt prefix changes sampling, so slots are not
  paired.
- DeepSeek-V4-Flash is held as a replication arm if a delta lands; its stock baseline is
  also free.

**Metrics.** Primary: correctness rate. Secondary, and the one that matters for the
goal of more fast kernels: count of correct kernels with speedup > 1.1, and the `tl.dot`
correctness rate specifically for arm 2.

**Framing caveat.** Prompt changes are aimed at correctness, not directly at speed.
Within a problem, code structure barely predicts which kernel is faster — 51.9%
(kernel launches), 53.1% (torch ops retained), 55.3% (`@triton.jit` count), 51.4%
(file length), against 50% chance — and median speedup is ≈1.00 across every strategy
cell measured. Speed is expected to follow from unblocking correct matmul kernels, not
from better-written kernels in general.

## Testing

- **Byte-identity:** `build_base_prompt(p, deltas=frozenset())` equals
  `get_prompt_for_backend(...)` across levels 1/2/6. Licenses arm-0 reuse.
- **Golden file per delta** on three fixtures: problem 19 (3 inputs, multi-output),
  10000 (docstring/rank conflict), 0 (empty init args).
- **Degradation:** a problem where `shapes_from_source` returns `[]` omits the contract
  block without crashing.
- **Determinism:** two calls with the same delta set produce identical strings.
- **Tuple wrapper:** tensor passthrough, 2-tuple, 1-tuple, list, kwargs-taking `forward`.
- **Symmetry:** `Model` and `ModelNew` wrapping share one code path.

Tests run on the cluster (`./sync-up && ./remote 'python -m pytest ...'`).

## Risks

| risk | mitigation |
|---|---|
| Byte-identity test fails, invalidating arm-0 reuse | Detected before any GPU spend; fall back to generating arm 0 (2,000 completions) |
| Tuple wrapper changes grading semantics | Documented as "primary output only"; symmetric; flag-gated with old behaviour reachable |
| `hardware` and `pitfalls` arms underpowered at 1,840 slots | Stated up front; a null result is recorded as inconclusive, not negative |
| Four arms vs one control inflates family-wise false positives to ~18.5% | A winner is a candidate; confirm on the free DeepSeek stock baseline before adopting |
| 8% of the slice is dropout-dead | Excluded at analysis, not generation |
| Slurm QoS caps `gres/gpu=16` across all jobs | Four arms run in waves; do not size wall clock off arm count |
| Cluster home at 176.2/200 GiB | Check headroom before launching; past the cap nothing writes and running jobs die |

## Branch and commit plan

No commits without explicit approval.

1. **Commit the 16 modified files on `prm-labeling`** as the as-run provenance of every
   kernel graded so far — they exist only on unbacked-up scratch. Includes the
   `critics.py` submission-gate ablation; commit as-is rather than reverting, because it
   is part of what actually ran.
2. **Branch `prompt-deltas` off that commit**, so both branches carry the provenance.
3. KernelBench-side change (`scripts/` only) is a separate repo, hence its own commit.

## Open decisions

- Whether to commit the two untracked notebooks (`reranker_final_eval_fokus.ipynb`,
  `.executed.ipynb`). Recommendation: commit the source, leave the executed output out.
- Whether step 1 lands on `prm-labeling` or directly on `prompt-deltas`.
  Recommendation: `prm-labeling`, so provenance is not branch-specific.

## Related standing findings (not in scope)

Recorded so they are not lost, not proposed here:

- The three generator runs in evaluation should be folded into ORM training. Going 1→2
  generators multiplied big speed pairs by **4.37x**; `list_size: 24` has headroom (4 of
  7,604 lists are full); 22.8% of lists have exactly one positive and contribute zero
  speed signal.
- ORM checkpoints should be selected on held-out levels 1/2, not kb6 val — arm ordering
  inverts between in-domain and OOD.
- The lint loop costs −25.7%/−31.7% correctness as a final answer but gives +8.7%/+7.3%
  as a candidate pool. Since training consumes rounds [0,1,2], keep it as a pool; do not
  invest in repair.
- `reranker_final_eval_fokus.ipynb` §3b draws per-column jitter from
  `set(POLICIES.values())`, whose iteration order varies per process, so plots are not
  reproducible. One-line fix (`sorted(...)`) reported, not applied.
