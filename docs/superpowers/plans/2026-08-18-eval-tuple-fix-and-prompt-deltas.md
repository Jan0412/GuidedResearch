# Tuple-Output Grading Fix and Prompt Delta Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make multi-output level-6 problems gradeable, and add a flag-gated prompt delta layer so four prompt variants can be A/B tested against a free stock baseline.

**Architecture:** Two independent pieces in two different repos. Part A appends a
subclass wrapper to both model source strings in the KernelBench *driver* layer, so
`src/kernelbench/` is never touched. Part B adds an additive-only delta registry to
`kernel_gen`, where `deltas=frozenset()` reproduces today's prompt byte-for-byte.

**Tech Stack:** Python 3.12, pytest, argparse, pydra (KernelBench configs), Triton 3.6,
Slurm + ssh for cluster execution.

**Design spec:** `docs/superpowers/specs/2026-08-18-eval-tuple-fix-and-prompt-deltas-design.md`

## Global Constraints

- **`KernelBench/src/kernelbench/` is NEVER modified.** All Part A changes live in
  `KernelBench/scripts/`. This is the user's explicit constraint.
- **`deltas=frozenset()` must reproduce the current prompt byte-for-byte.** This is what
  licenses reusing existing runs as the A/B control arm. Task 6 pins it with a test.
- **Delta ordering is deterministic** — iterate a module-level tuple, never a set.
  Set iteration order varies per process.
- **Graceful degradation** — if `shapes_from_source` returns `[]` (0.3% of problems), the
  contract block is omitted entirely, never half-rendered. Never raise.
- **CLI multi-value flags are comma-separated strings, never `nargs`.** An argparse dest
  is a public YAML key (see `kernel_gen/core/cli.py` module docstring); PyYAML writes a
  list as block lines starting with `-`, which the flat scanner in `checker/runs.py` drops.
- **Terse comments.** No long comment blocks or docstrings, in Python or bash. Match the
  density of surrounding code.
- **Commit messages: one sentence, no `Co-Authored-By`, no Claude/Anthropic trailers.**
  This overrides any harness default.
- **Do not commit until the human partner has explicitly granted commit authority for
  this branch.** If it has not been granted, stage changes and report instead.
- **Where tests run:** pure-string tests (Tasks 1, 2, 3, 4, 5, 8) run locally with
  `python3 -m pytest`. Tasks 6 and 7 import `kernelbench` and must run on the cluster via
  `./sync-up && ./remote 'python -m pytest ...'` from the `hpi2` directory.
- **Never run two cluster test jobs concurrently.** `sync-up` targets a fixed remote path
  per tree, so concurrent implementers would overwrite each other.

## File Structure

**KernelBench tree** (`hpi2/KernelBench`) — Part A:

- Create `scripts/kb_normalize.py` — pure string transforms; no torch import.
- Create `scripts/test_kb_normalize.py` — runs locally, no torch needed.
- Modify `scripts/eval_from_generations.py` — apply the transform at both
  `eval_kernel_against_ref` call sites (currently lines ~221 and ~317), behind a flag.

**GuidedResearch tree** (`hpi2/GuidedResearch`) — Part B:

- Create `kernel_gen/core/prompt_deltas.py` — registry, parser, block builders.
- Create `kernel_gen/tests/unit/test_prompt_deltas.py` — block/registry tests.
- Modify `kernel_gen/core/prompts.py` — `build_base_prompt` gains `deltas`.
- Modify `kernel_gen/core/cli.py` — `--prompt-deltas`.
- Modify `kernel_gen/arms/lintloop.py` — thread the flag through (call site line ~226).
- Modify `kernel_gen/tests/unit/test_prompts.py` — add the byte-identity test.
- Create `kernel_gen/arm_stats.py` — A/B analysis (a package module, so tests import it via the conftest repo-root bootstrap; `scripts/` is not a package).
- Create `kernel_gen/tests/unit/test_compare_arms.py` — analysis tests.

---

### Task 1: Tuple-output normaliser (pure function)

**Files:**
- Create: `KernelBench/scripts/kb_normalize.py`
- Test: `KernelBench/scripts/test_kb_normalize.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `wrap_primary_output(src: str, class_name: str) -> str` and
  `normalize_pair(ref_src: str, kernel_src: str, enabled: bool = True) -> tuple[str, str]`.
  Task 2 calls `normalize_pair`.

Work in the `hpi2/KernelBench` tree for this task.

- [ ] **Step 1: Write the failing test**

Create `KernelBench/scripts/test_kb_normalize.py`:

```python
"""kb_normalize: collapse a multi-output forward to its primary output.

Deliberately torch-free -- the wrapper only subclasses whatever the source defines,
so a plain class exercises the same code path and the test runs locally.
"""

from kb_normalize import normalize_pair, wrap_primary_output


def _load(src: str, name: str):
    ns: dict = {}
    exec(compile(src, "<test>", "exec"), ns)
    return ns[name]


def test_tensor_return_passes_through_unchanged():
    src = "class Model:\n    def forward(self, x):\n        return x * 2\n"
    model = _load(wrap_primary_output(src, "Model"), "Model")()
    assert model.forward(3) == 6


def test_two_tuple_collapses_to_first_element():
    src = "class Model:\n    def forward(self, x):\n        return (x, x * 10)\n"
    model = _load(wrap_primary_output(src, "Model"), "Model")()
    assert model.forward(3) == 3


def test_one_element_tuple_collapses_to_the_element():
    src = "class Model:\n    def forward(self, x):\n        return (x,)\n"
    model = _load(wrap_primary_output(src, "Model"), "Model")()
    assert model.forward(7) == 7


def test_list_return_collapses_like_a_tuple():
    src = "class Model:\n    def forward(self, x):\n        return [x, x + 1]\n"
    model = _load(wrap_primary_output(src, "Model"), "Model")()
    assert model.forward(4) == 4


def test_empty_tuple_is_returned_untouched():
    src = "class Model:\n    def forward(self, x):\n        return ()\n"
    model = _load(wrap_primary_output(src, "Model"), "Model")()
    assert model.forward(1) == ()


def test_keyword_arguments_reach_the_inner_forward():
    src = "class Model:\n    def forward(self, x, scale=1):\n        return (x * scale, x)\n"
    model = _load(wrap_primary_output(src, "Model"), "Model")()
    assert model.forward(3, scale=5) == 15


def test_wrapping_twice_is_harmless():
    src = "class Model:\n    def forward(self, x):\n        return (x, x)\n"
    once = wrap_primary_output(src, "Model")
    model = _load(wrap_primary_output(once, "Model"), "Model")()
    assert model.forward(9) == 9


def test_normalize_pair_wraps_model_and_modelnew():
    ref = "class Model:\n    def forward(self, x):\n        return (x, 0)\n"
    new = "class ModelNew:\n    def forward(self, x):\n        return (x, 1)\n"
    ref_out, new_out = normalize_pair(ref, new)
    assert _load(ref_out, "Model")().forward(2) == 2
    assert _load(new_out, "ModelNew")().forward(2) == 2


def test_normalize_pair_disabled_returns_sources_unchanged():
    ref, new = "a = 1\n", "b = 2\n"
    assert normalize_pair(ref, new, enabled=False) == (ref, new)
```

- [ ] **Step 2: Run test to verify it fails**

Run from `hpi2/KernelBench`:
```bash
python3 -m pytest scripts/test_kb_normalize.py -v
```
Expected: collection error, `ModuleNotFoundError: No module named 'kb_normalize'`.

- [ ] **Step 3: Write minimal implementation**

Create `KernelBench/scripts/kb_normalize.py`:

```python
"""Grade multi-output problems on their primary output.

KernelBench's correctness check does `output.shape != output_new.shape`, which raises
on a tuple -- so every level-6 problem whose forward returns a tuple is unpassable
(1,726 problems, ~1% correct). Zero official KernelBench problems return a tuple, so
this normalises a case upstream never defined rather than changing its behaviour.

Appending a subclass rather than rewriting the return keeps the transform incapable of
mangling the source, and the isinstance check makes it a no-op for single-output
problems -- so it is applied unconditionally and never has to detect anything.
"""

from __future__ import annotations

_WRAPPER = '''

_KB_INNER_{name} = {name}


class {name}(_KB_INNER_{name}):
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        return out
'''


def wrap_primary_output(src: str, class_name: str) -> str:
    return src.rstrip("\n") + "\n" + _WRAPPER.format(name=class_name)


def normalize_pair(
    ref_src: str, kernel_src: str, enabled: bool = True
) -> tuple[str, str]:
    if not enabled:
        return ref_src, kernel_src
    return (
        wrap_primary_output(ref_src, "Model"),
        wrap_primary_output(kernel_src, "ModelNew"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest scripts/test_kb_normalize.py -v
```
Expected: 9 passed, no warnings.

- [ ] **Step 5: Commit** (only if commit authority granted — see Global Constraints)

```bash
git add scripts/kb_normalize.py scripts/test_kb_normalize.py
git commit -m "Grade multi-output problems on their primary output by appending a collapsing subclass to both model sources"
```

---

### Task 2: Wire the normaliser into the eval driver

**Files:**
- Modify: `KernelBench/scripts/eval_from_generations.py`
- Test: `KernelBench/scripts/test_kb_normalize.py` (extend)

**Interfaces:**
- Consumes: `normalize_pair` from Task 1.
- Produces: a `normalize_multi_output` config option, default `True`.

Work in the `hpi2/KernelBench` tree.

- [ ] **Step 1: Read the two call sites**

```bash
grep -n -B25 "eval_kernel_against_ref(" scripts/eval_from_generations.py
```

There are two calls (around lines 221 and 317). Note the enclosing function signature of
each and whether a `configs` object is in scope. Both pass `original_model_src=ref_arch_src`
and `custom_model_src=kernel_src`.

- [ ] **Step 2: Write the failing test**

Append to `KernelBench/scripts/test_kb_normalize.py`:

```python
def test_driver_exposes_normalisation_and_defaults_to_on():
    import eval_from_generations as efg

    assert hasattr(efg, "NORMALIZE_MULTI_OUTPUT_DEFAULT")
    assert efg.NORMALIZE_MULTI_OUTPUT_DEFAULT is True
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python3 -m pytest scripts/test_kb_normalize.py::test_driver_exposes_normalisation_and_defaults_to_on -v
```
Expected: FAIL with `AttributeError` on `NORMALIZE_MULTI_OUTPUT_DEFAULT`.

If the import itself fails because the driver needs heavy dependencies (`pydra`, `torch`,
`modal`) that are absent locally, report this as DONE_WITH_CONCERNS and move the test to
the cluster run instead of stubbing the import.

- [ ] **Step 4: Write minimal implementation**

In `scripts/eval_from_generations.py`, add near the other module-level constants:

```python
from kb_normalize import normalize_pair

NORMALIZE_MULTI_OUTPUT_DEFAULT = True
```

Then immediately before **each** of the two `eval_kernel_against_ref(` calls, insert:

```python
ref_arch_src, kernel_src = normalize_pair(
    ref_arch_src, kernel_src, normalize_multi_output
)
```

Thread `normalize_multi_output` into scope at each site:
- Where a pydra `configs` object is available, read
  `getattr(configs, "normalize_multi_output", NORMALIZE_MULTI_OUTPUT_DEFAULT)` and also
  add `normalize_multi_output: bool = True` to the pydra Config class alongside
  `measure_performance`.
- Where the call is inside a worker function without `configs`, add a
  `normalize_multi_output: bool = NORMALIZE_MULTI_OUTPUT_DEFAULT` keyword parameter to
  that function and pass the config value from its caller.

Do not change any other behaviour of the driver.

- [ ] **Step 5: Run test to verify it passes**

```bash
python3 -m pytest scripts/test_kb_normalize.py -v
```
Expected: all pass.

- [ ] **Step 6: Verify src/ is untouched**

```bash
git status --porcelain src/
```
Expected: **empty output**. Any change under `src/` violates a Global Constraint.

- [ ] **Step 7: Commit** (only if commit authority granted)

```bash
git add scripts/eval_from_generations.py scripts/test_kb_normalize.py
git commit -m "Normalise multi-output models at both eval call sites behind a default-on flag"
```

---

### Task 3: Delta registry, parser, and applier

**Files:**
- Create: `GuidedResearch/kernel_gen/core/prompt_deltas.py`
- Test: `GuidedResearch/kernel_gen/tests/unit/test_prompt_deltas.py`

**Interfaces:**
- Consumes: `kernel_gen.core.model.Problem` (fields: `level`, `problem_id`, `name`,
  `ref_arch_src`).
- Produces: `DELTA_ORDER: tuple[str, ...]`, `parse_deltas(spec: str) -> frozenset[str]`,
  `apply_deltas(prompt: str, problem: Problem, deltas: frozenset[str]) -> str`,
  and `BLOCKS: dict[str, Callable[[Problem], str | None]]`. Tasks 4-6 add to `BLOCKS`
  and call `apply_deltas`.

Work in the `hpi2/GuidedResearch` tree. All of this task runs locally.

- [ ] **Step 1: Write the failing test**

Create `GuidedResearch/kernel_gen/tests/unit/test_prompt_deltas.py`:

```python
"""``kernel_gen.core.prompt_deltas``: additive, individually-switchable prompt blocks.

The invariant the A/B rests on is that an empty delta set changes nothing -- that is
what lets an existing run stand in as the control arm.
"""

from __future__ import annotations

import pytest

from kernel_gen.core.model import Problem
from kernel_gen.core.prompt_deltas import (
    DELTA_ORDER,
    apply_deltas,
    parse_deltas,
)

REF = (
    "import torch\n"
    "import torch.nn as nn\n\n\n"
    "class Model(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return x * 2\n\n\n"
    "def get_inputs():\n"
    "    return [torch.rand([48, 48, 48, 48])]\n\n\n"
    "def get_init_inputs():\n"
    "    return [48]\n"
)


def _problem() -> Problem:
    return Problem(level=6, problem_id=0, name="0_Demo.py", ref_arch_src=REF)


def test_empty_delta_set_leaves_the_prompt_untouched():
    prompt = "BASE PROMPT\n"
    assert apply_deltas(prompt, _problem(), frozenset()) == prompt


def test_parse_deltas_accepts_a_comma_string():
    assert parse_deltas("precision,hardware") == frozenset({"precision", "hardware"})


def test_parse_deltas_of_empty_string_is_empty():
    assert parse_deltas("") == frozenset()
    assert parse_deltas(None) == frozenset()


def test_parse_deltas_tolerates_whitespace_and_duplicates():
    assert parse_deltas(" precision , precision ,hardware") == frozenset(
        {"precision", "hardware"}
    )


def test_parse_deltas_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown prompt delta"):
        parse_deltas("contract,nonsense")


def test_hardware_is_not_a_text_block():
    # `hardware` is a flag on the KernelBench call, not appended text.
    assert "hardware" in DELTA_ORDER


def test_delta_order_is_a_tuple_not_a_set():
    # Set iteration order varies per process; the rendered prompt must not.
    assert isinstance(DELTA_ORDER, tuple)


def test_apply_deltas_is_deterministic_across_calls():
    p = _problem()
    both = frozenset({"precision", "pitfalls"})
    assert apply_deltas("BASE\n", p, both) == apply_deltas("BASE\n", p, both)
```

- [ ] **Step 2: Run test to verify it fails**

Run from `hpi2/GuidedResearch`:
```bash
python3 -m pytest kernel_gen/tests/unit/test_prompt_deltas.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'kernel_gen.core.prompt_deltas'`.

- [ ] **Step 3: Write minimal implementation**

Create `GuidedResearch/kernel_gen/core/prompt_deltas.py` exactly as below. Blocks
returning `None` is the correct minimal implementation for this task's tests — a `None`
block is skipped by `apply_deltas`, which is also the production behaviour when a block
cannot be rendered. Tasks 4 and 5 replace those three function bodies.

```python
"""Additive prompt deltas: named blocks appended to KernelBench's prompt.

Every delta is additive or a flag, so an empty set reproduces the stock prompt
byte-for-byte -- the property that lets an existing run serve as the A/B control.
`hardware` carries no text; it flips include_hardware on the KernelBench call.
"""

from __future__ import annotations

from typing import Callable

from .model import Problem

#: Fixed application order. NEVER iterate a set here -- per-process set ordering
#: would make the rendered prompt irreproducible.
DELTA_ORDER: tuple[str, ...] = ("contract", "precision", "pitfalls", "hardware")

#: Deltas that flip a KernelBench argument instead of appending text.
FLAG_DELTAS: frozenset[str] = frozenset({"hardware"})


def _contract_block(problem: Problem) -> str | None:
    return None


def _precision_block(problem: Problem) -> str | None:
    return None


def _pitfalls_block(problem: Problem) -> str | None:
    return None


BLOCKS: dict[str, Callable[[Problem], str | None]] = {
    "contract": _contract_block,
    "precision": _precision_block,
    "pitfalls": _pitfalls_block,
}


def parse_deltas(spec: str | None) -> frozenset[str]:
    if not spec:
        return frozenset()
    names = {part.strip() for part in spec.split(",") if part.strip()}
    unknown = sorted(names - set(DELTA_ORDER))
    if unknown:
        raise ValueError(
            f"unknown prompt delta {unknown}; known: {list(DELTA_ORDER)}"
        )
    return frozenset(names)


def apply_deltas(prompt: str, problem: Problem, deltas: frozenset[str]) -> str:
    if not deltas:
        return prompt
    out = prompt
    for name in DELTA_ORDER:
        if name not in deltas or name in FLAG_DELTAS:
            continue
        block = BLOCKS[name](problem)
        if block:
            out = out.rstrip("\n") + "\n\n" + block.strip("\n") + "\n"
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest kernel_gen/tests/unit/test_prompt_deltas.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit** (only if commit authority granted)

```bash
git add kernel_gen/core/prompt_deltas.py kernel_gen/tests/unit/test_prompt_deltas.py
git commit -m "Add a prompt delta registry whose empty set leaves the prompt untouched"
```

---

### Task 4: The `contract` block

**Files:**
- Modify: `GuidedResearch/kernel_gen/core/prompt_deltas.py`
- Test: `GuidedResearch/kernel_gen/tests/unit/test_prompt_deltas.py` (extend)

**Interfaces:**
- Consumes: `checker.lint.shapes.shapes_from_source(source: str) -> list`, which returns
  a list of `((dim, ...), dtype_str)` pairs, e.g.
  `[((48, 48, 48, 48), 'float32'), ((48, 48, 48, 48), 'float32')]`.
- Produces: a populated `_contract_block`.

Runs locally — `shapes_from_source` is AST-based and imports no torch.

- [ ] **Step 1: Write the failing test**

Append to `kernel_gen/tests/unit/test_prompt_deltas.py`:

```python
def test_contract_block_states_constructor_args_and_every_input():
    ref = REF.replace("return [torch.rand([48, 48, 48, 48])]",
                      "return [torch.rand([48, 48, 48, 48]), torch.rand([48, 48, 48, 48])]")
    p = Problem(level=6, problem_id=19, name="19_Demo.py", ref_arch_src=ref)
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert "## Input contract" in out
    assert "ModelNew(48)" in out
    assert "2 positional inputs" in out
    assert out.count("float32, shape (48, 48, 48, 48)") == 2
    assert "supersedes any shape stated in comments or docstrings" in out


def test_contract_block_renders_a_no_argument_constructor():
    ref = REF.replace("    return [48]", "    return []")
    p = Problem(level=6, problem_id=0, name="0_Demo.py", ref_arch_src=ref)
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert "ModelNew()" in out


def test_contract_block_uses_the_last_get_init_inputs_definition():
    # Every level-6 file defines get_init_inputs twice; the converter footer wins.
    ref = REF + "\n\ndef get_init_inputs():\n    return [99]\n"
    p = Problem(level=6, problem_id=1, name="1_Demo.py", ref_arch_src=ref)
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert "ModelNew(99)" in out
    assert "ModelNew(48)" not in out


def test_contract_block_is_omitted_when_shapes_cannot_be_resolved():
    p = Problem(level=6, problem_id=2, name="2_Demo.py", ref_arch_src="x = 1\n")
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert out == "BASE\n"


def test_contract_block_never_raises_on_unparseable_source():
    p = Problem(level=6, problem_id=3, name="3_Demo.py", ref_arch_src="def (:\n")
    assert apply_deltas("BASE\n", p, frozenset({"contract"})) == "BASE\n"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest kernel_gen/tests/unit/test_prompt_deltas.py -k contract -v
```
Expected: the first three fail (`"## Input contract" in out` is False — the stub returns
`None`); the last two already pass.

- [ ] **Step 3: Write minimal implementation**

In `prompt_deltas.py`, add the imports and replace `_contract_block`:

```python
import ast


def _init_args(source: str) -> list | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    last = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
            last = node
    if last is None:
        return None
    for node in ast.walk(last):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
            return list(value) if isinstance(value, (list, tuple)) else [value]
    return None


def _contract_block(problem: Problem) -> str | None:
    from checker.lint.shapes import shapes_from_source

    try:
        shapes = shapes_from_source(problem.ref_arch_src)
    except Exception:  # noqa: BLE001 - a missing block is better than a failed run
        return None
    if not shapes:
        return None
    args = _init_args(problem.ref_arch_src)
    if args is None:
        return None

    ctor = f"ModelNew({', '.join(repr(a) for a in args)})"
    lines = [
        "## Input contract",
        f"ModelNew is constructed as {ctor} and called with {len(shapes)} positional "
        "inputs, all on CUDA and contiguous:",
    ]
    for i, (shape, dtype) in enumerate(shapes, 1):
        dims = ", ".join(str(d) for d in shape)
        lines.append(f"  arg {i}: {dtype}, shape ({dims})")
    lines.append(
        "This supersedes any shape stated in comments or docstrings above."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest kernel_gen/tests/unit/test_prompt_deltas.py -k contract -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit** (only if commit authority granted)

```bash
git add kernel_gen/core/prompt_deltas.py kernel_gen/tests/unit/test_prompt_deltas.py
git commit -m "Render an input contract block from the reference's shapes and final get_init_inputs"
```

---

### Task 5: The `precision` and `pitfalls` blocks

**Files:**
- Modify: `GuidedResearch/kernel_gen/core/prompt_deltas.py`
- Test: `GuidedResearch/kernel_gen/tests/unit/test_prompt_deltas.py` (extend)

**Interfaces:**
- Consumes: nothing new — both are static text ignoring `problem`.
- Produces: populated `_precision_block` and `_pitfalls_block`.

Runs locally.

- [ ] **Step 1: Write the failing test**

Append to `kernel_gen/tests/unit/test_prompt_deltas.py`:

```python
def test_precision_block_names_the_tolerance_and_the_tf32_default():
    out = apply_deltas("BASE\n", _problem(), frozenset({"precision"}))
    assert "## Numerical precision" in out
    assert "1e-4" in out
    assert 'input_precision="ieee"' in out
    assert "TF32" in out


def test_pitfalls_block_carries_every_rule():
    out = apply_deltas("BASE\n", _problem(), frozenset({"pitfalls"}))
    assert "## Triton pitfalls" in out
    for needle in (
        "ModelNew",
        "mask",
        "tl.constexpr",
        "@triton.jit",
        "import",
        "Triton 3.6",
    ):
        assert needle in out


def test_static_blocks_ignore_the_problem():
    a = apply_deltas("BASE\n", _problem(), frozenset({"precision", "pitfalls"}))
    other = Problem(level=1, problem_id=7, name="7_X.py", ref_arch_src="z = 0\n")
    assert apply_deltas("BASE\n", other, frozenset({"precision", "pitfalls"})) == a


def test_blocks_are_ordered_by_delta_order_not_by_set_iteration():
    out = apply_deltas("BASE\n", _problem(), frozenset({"pitfalls", "precision"}))
    assert out.index("## Numerical precision") < out.index("## Triton pitfalls")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest kernel_gen/tests/unit/test_prompt_deltas.py -k "precision or pitfalls or static_blocks or delta_order" -v
```
Expected: FAIL — `"## Numerical precision" in out` is False (stubs return `None`).

- [ ] **Step 3: Write minimal implementation**

Replace the two stubs in `prompt_deltas.py`:

```python
_PRECISION = """\
## Numerical precision
Correctness is checked against the PyTorch reference running in true FP32, with a
tolerance of 1e-4. Triton's tl.dot defaults to TF32 on this GPU, which carries roughly
1e-3 of error and therefore FAILS this check. If you call tl.dot on float32 inputs you
must pass input_precision="ieee". Note also that cuBLAS is already near-optimal for FP32
matmul here: fusing work around torch.matmul usually beats reimplementing the GEMM.
"""

_PITFALLS = """\
## Triton pitfalls
- Define ModelNew as a complete nn.Module and finish the class.
- Mask every tl.load and tl.store unless the extent is an exact multiple of the block
  size; an unmasked tail is an illegal memory access.
- Keep the launch grid and the kernel signature consistent: a block size read as
  meta["BLOCK_SIZE"] must be declared BLOCK_SIZE: tl.constexpr and passed by keyword.
- Never call a @triton.jit function from Python; launch it as kernel[grid](...).
- Import everything you use, including torch.nn as nn and typing.Optional.
- Target Triton 3.6 and do not invent builtins; tl.constant does not exist.
"""


def _precision_block(problem: Problem) -> str | None:
    return _PRECISION


def _pitfalls_block(problem: Problem) -> str | None:
    return _PITFALLS
```

- [ ] **Step 4: Run the whole file to verify it passes**

```bash
python3 -m pytest kernel_gen/tests/unit/test_prompt_deltas.py -v
```
Expected: all pass, including
`test_apply_deltas_orders_blocks_by_delta_order_not_set_order` from Task 3.

- [ ] **Step 5: Commit** (only if commit authority granted)

```bash
git add kernel_gen/core/prompt_deltas.py kernel_gen/tests/unit/test_prompt_deltas.py
git commit -m "Add the precision and pitfalls prompt blocks"
```

---

### Task 6: Integrate deltas into `build_base_prompt`

**Files:**
- Modify: `GuidedResearch/kernel_gen/core/prompts.py`
- Test: `GuidedResearch/kernel_gen/tests/unit/test_prompts.py` (extend)

**Interfaces:**
- Consumes: `apply_deltas`, `FLAG_DELTAS` from Task 3.
- Produces: `build_base_prompt(..., deltas: frozenset[str] = frozenset())`. Task 7 passes
  the parsed flag here.

**This task's tests import `kernelbench` and must run on the cluster.** From the `hpi2`
directory: `./sync-up && ./remote 'python -m pytest <path> -v'`. Do not run a cluster
test job while another is running.

- [ ] **Step 1: Write the failing test**

Append to `kernel_gen/tests/unit/test_prompts.py`:

```python
def test_empty_deltas_reproduce_the_stock_prompt_byte_for_byte():
    # The A/B reuses an existing run as its control arm; that is only legitimate while
    # this holds.
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    ref = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"
    problem = Problem(level=1, problem_id=1, name="1_X.py", ref_arch_src=ref)
    stock = get_prompt_for_backend(ref_arch_src=ref, backend="triton", option="one_shot")
    assert build_base_prompt(problem, deltas=frozenset()) == stock


def test_a_text_delta_appends_to_the_stock_prompt_without_altering_it():
    ref = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"
    problem = Problem(level=1, problem_id=1, name="1_X.py", ref_arch_src=ref)
    stock = build_base_prompt(problem, deltas=frozenset())
    withd = build_base_prompt(problem, deltas=frozenset({"precision"}))
    assert withd.startswith(stock.rstrip("\n"))
    assert "## Numerical precision" in withd


def test_hardware_delta_turns_on_the_hardware_section():
    ref = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x\n"
    problem = Problem(level=1, problem_id=1, name="1_X.py", ref_arch_src=ref)
    out = build_base_prompt(problem, deltas=frozenset({"hardware"}), gpu_name="H100")
    assert "H100" in out
```

- [ ] **Step 2: Run test to verify it fails**

From `hpi2`:
```bash
./sync-up && ./remote 'python -m pytest kernel_gen/tests/unit/test_prompts.py -k "deltas or hardware_delta" -v'
```
Expected: FAIL — `build_base_prompt() got an unexpected keyword argument 'deltas'`.

- [ ] **Step 3: Write minimal implementation**

In `kernel_gen/core/prompts.py`, change `build_base_prompt`:

```python
def build_base_prompt(
    problem: Problem,
    backend: str = "triton",
    option: str = "one_shot",
    include_hardware: bool = False,
    gpu_name: str | None = None,
    deltas: frozenset[str] = frozenset(),
) -> str:
    """KernelBench's own prompt constructor, plus any enabled additive deltas."""
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    from .prompt_deltas import apply_deltas

    hardware = include_hardware or "hardware" in deltas
    prompt = get_prompt_for_backend(
        ref_arch_src=problem.ref_arch_src,
        backend=backend,
        option=option,
        include_hardware=hardware,
        gpu_name=gpu_name if hardware else None,
    )
    return apply_deltas(prompt, problem, deltas)
```

Note the `gpu_name` guard now keys off `hardware`, not `include_hardware`, so the
`hardware` delta supplies the GPU name it needs.

- [ ] **Step 4: Run test to verify it passes**

```bash
./sync-up && ./remote 'python -m pytest kernel_gen/tests/unit/test_prompts.py -v'
```
Expected: all pass, including the pre-existing tests in that file.

- [ ] **Step 5: Commit** (only if commit authority granted)

```bash
git add kernel_gen/core/prompts.py kernel_gen/tests/unit/test_prompts.py
git commit -m "Thread additive prompt deltas through build_base_prompt, keeping the empty set byte-identical"
```

---

### Task 7: `--prompt-deltas` CLI flag and lintloop wiring

**Files:**
- Modify: `GuidedResearch/kernel_gen/core/cli.py`
- Modify: `GuidedResearch/kernel_gen/arms/lintloop.py` (call site ~line 226)
- Test: `GuidedResearch/kernel_gen/tests/unit/test_cli.py` (extend)

**Interfaces:**
- Consumes: `parse_deltas` from Task 3, `build_base_prompt(..., deltas=...)` from Task 6.
- Produces: an argparse dest `prompt_deltas` (a comma string), persisted into
  `generation_config.yaml` by `artifacts.write_config`.

- [ ] **Step 1: Write the failing test**

Append to `kernel_gen/tests/unit/test_cli.py`, matching that file's existing style for
building a parser:

```python
def test_prompt_deltas_defaults_to_empty_string():
    import argparse

    from kernel_gen.core import cli

    parser = argparse.ArgumentParser()
    cli.add_prompt_args(parser)
    args = parser.parse_args([])
    assert args.prompt_deltas == ""


def test_prompt_deltas_is_a_single_comma_string_not_a_list():
    # An argparse dest is a public YAML key; nargs would serialise as a block list
    # whose lines the flat config scanner drops.
    import argparse

    from kernel_gen.core import cli

    parser = argparse.ArgumentParser()
    cli.add_prompt_args(parser)
    args = parser.parse_args(["--prompt-deltas", "contract,precision"])
    assert args.prompt_deltas == "contract,precision"
    assert isinstance(args.prompt_deltas, str)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest kernel_gen/tests/unit/test_cli.py -k prompt_deltas -v
```
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'prompt_deltas'`.

- [ ] **Step 3: Write minimal implementation**

In `kernel_gen/core/cli.py`, inside `add_prompt_args`, add:

```python
    parser.add_argument(
        "--prompt-deltas",
        default="",
        help="Comma-separated additive prompt deltas: contract, precision, pitfalls, "
             "hardware. Empty (default) reproduces the stock KernelBench prompt "
             "byte-for-byte, which is what makes an existing run a valid control arm.",
    )
```

In `kernel_gen/arms/lintloop.py`, add the import beside the existing prompt imports:

```python
from kernel_gen.core.prompt_deltas import parse_deltas
```

Then at the `build_base_prompt` call (around line 226), parse once outside the closure
and pass it in:

```python
    deltas = parse_deltas(args.prompt_deltas)

    def base_prompt(problem: Problem) -> str:
        if problem.problem_id not in prompt_cache:
            prompt_cache[problem.problem_id] = build_base_prompt(
                problem,
                backend=args.backend,
                option=args.option,
                include_hardware=args.include_hardware,
                gpu_name=args.gpu_name,
                deltas=deltas,
            )
        return prompt_cache[problem.problem_id]
```

Also add `"prompt_deltas"` to the config key list around line 215 so it is recorded in
`generation_config.yaml`. Read that list first and match its existing formatting.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest kernel_gen/tests/unit/test_cli.py -v
```
Expected: all pass.

- [ ] **Step 5: Verify the flag reaches a rendered prompt**

From `hpi2`:
```bash
./sync-up && ./remote 'python -m kernel_gen.arms.lintloop --model x --level 1 --problems 0 --dry-run --prompt-deltas precision | tail -20'
```
Expected: the rendered prompt ends with the `## Numerical precision` block.

- [ ] **Step 6: Commit** (only if commit authority granted)

```bash
git add kernel_gen/core/cli.py kernel_gen/arms/lintloop.py kernel_gen/tests/unit/test_cli.py
git commit -m "Add --prompt-deltas and record it in the generation config"
```

---

### Task 8: Arm comparison script

**Files:**
- Create: `GuidedResearch/kernel_gen/arm_stats.py`
- Test: `GuidedResearch/kernel_gen/tests/unit/test_compare_arms.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `compare(control: dict, arm: dict, exclude: set[int]) -> dict` with keys
  `n_control`, `n_arm`, `p_control`, `p_arm`, `diff`, `se`, `sigma`.

Runs locally — pure arithmetic over already-parsed dicts.

The A/B excludes the 40 dropout problems at analysis time and reports a difference
against a 2σ threshold of ~2.7pt. With four arms against one control the family-wise
false-positive rate is ~18.5%, so a winner is a candidate, not a conclusion.

- [ ] **Step 1: Write the failing test**

Create `kernel_gen/tests/unit/test_compare_arms.py`:

```python
"""compare_arms: correctness difference between one A/B arm and the control."""

from __future__ import annotations

import math

from kernel_gen.arm_stats import compare

# {problem_id: [bool, ...]} -- one entry per sample slot
CONTROL = {1: [True, False, False, False], 2: [False] * 4, 3: [True, True, False, False]}
ARM = {1: [True, True, False, False], 2: [False] * 4, 3: [True, True, True, False]}


def test_counts_every_slot_when_nothing_is_excluded():
    r = compare(CONTROL, ARM, exclude=set())
    assert r["n_control"] == 12
    assert r["n_arm"] == 12


def test_reports_the_correctness_rate_of_each_side():
    r = compare(CONTROL, ARM, exclude=set())
    assert r["p_control"] == 3 / 12
    assert r["p_arm"] == 5 / 12


def test_excluded_problems_are_dropped_from_both_sides():
    r = compare(CONTROL, ARM, exclude={2})
    assert r["n_control"] == 8
    assert r["p_control"] == 3 / 8


def test_difference_is_arm_minus_control():
    r = compare(CONTROL, ARM, exclude=set())
    assert math.isclose(r["diff"], 5 / 12 - 3 / 12)


def test_sigma_is_the_difference_over_its_standard_error():
    r = compare(CONTROL, ARM, exclude=set())
    assert math.isclose(r["sigma"], r["diff"] / r["se"])


def test_zero_variance_does_not_divide_by_zero():
    allwrong = {1: [False, False]}
    r = compare(allwrong, allwrong, exclude=set())
    assert r["diff"] == 0.0
    assert r["sigma"] == 0.0


def test_only_problems_present_in_both_arms_are_compared():
    r = compare({1: [True, False]}, {1: [True, True], 9: [True, True]}, exclude=set())
    assert r["n_control"] == 2
    assert r["n_arm"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest kernel_gen/tests/unit/test_compare_arms.py -v
```
Expected: `ModuleNotFoundError: No module named 'kernel_gen.arm_stats'`.

If `scripts` is not an importable package, add an empty `scripts/__init__.py` as part of
this step.

- [ ] **Step 3: Write minimal implementation**

Create `GuidedResearch/kernel_gen/arm_stats.py`:

```python
"""Compare one A/B arm's correctness against the control arm.

Paired at problem level only: a changed prompt prefix changes sampling, so slots do
not correspond one-to-one across arms.
"""

from __future__ import annotations

import math


def compare(control: dict, arm: dict, exclude: set) -> dict:
    shared = (set(control) & set(arm)) - set(exclude)
    c = [ok for pid in shared for ok in control[pid]]
    a = [ok for pid in shared for ok in arm[pid]]
    n_c, n_a = len(c), len(a)
    p_c = sum(c) / n_c if n_c else 0.0
    p_a = sum(a) / n_a if n_a else 0.0
    var = (p_c * (1 - p_c) / n_c if n_c else 0.0) + (
        p_a * (1 - p_a) / n_a if n_a else 0.0
    )
    se = math.sqrt(var)
    diff = p_a - p_c
    return {
        "n_control": n_c,
        "n_arm": n_a,
        "p_control": p_c,
        "p_arm": p_a,
        "diff": diff,
        "se": se,
        "sigma": diff / se if se else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest kernel_gen/tests/unit/test_compare_arms.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit** (only if commit authority granted)

```bash
git add kernel_gen/arm_stats.py kernel_gen/tests/unit/test_compare_arms.py
git commit -m "Add an arm comparison helper reporting the correctness difference and its sigma"
```

---

## Out of Scope

Recorded so no task attempts them:

- The Dropout / missing `.eval()` bug. Left as KernelBench does it, by decision.
- Replacing the one-shot example, plan structure, and the instruction cleanup.
- Regenerating the level-6 corpus or its baselines.
- Running the A/B itself — that is GPU execution, not code, and follows this plan.
