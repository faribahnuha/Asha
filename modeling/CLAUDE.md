# Coding Standard for `modeling/`

This directory-scoped file applies only inside `modeling/` — not the rest of the datathon
repo. The root `CLAUDE.md` still applies everywhere, including here; this file adds stricter
rules specific to the actual scoring/ranking code, because that code's output is what tells an
ambulance driver where to go. Exploratory notebooks, data cleaning in `pre-processing/`, and docs
at the repo root are not held to this standard — it would make normal data exploration
impossible, and that's not what it's for.

## Where this standard comes from

The rules below are adapted from the "Power of 10" rules for safety-critical C (Gerard Holzmann,
JPL) — a defensible, externally sourced standard, not one invented for this project. Several of
those rules are literally about C (pointers, the preprocessor, `goto`). This file translates each
one to its closest Python equivalent rather than pretending Python has pointers to restrict. Where
a rule doesn't translate, that's stated explicitly instead of silently dropped — consistent with
this project's own rule against treating "doesn't apply" the same as "wasn't checked."

## The three-bullet file header

Every file in `modeling/` opens with a comment block with exactly three bullets:

1. **How this relates to the last file** — what it consumes from upstream, what it hands to
   downstream, or "nothing — this is the first stage" / "nothing — this is a cross-cutting
   tool, not a pipeline stage."
2. **What this file does** — one or two sentences, not a restatement of every function.
3. **Known limitations / assumptions.** This is the project's answer to the "idk, what's good
   practice" question this file was created to settle. Reasoning: the single biggest risk in
   this codebase isn't a crash, it's a confident-looking wrong answer in a tool that routes
   ambulances (see root `CLAUDE.md`: "patient safety first," "flag uncertainty," "never present
   an unchecked assumption as fact"). A header that only says what a file does and hides what it
   assumes is exactly the failure mode this project has spent the whole design process trying to
   stop. Glancing at this one line should tell a reader what NOT to trust about this file without
   reading the code.

## The rules, and what they mean in this codebase

| Original rule | What it means here |
|---|---|
| Simple control flow; no `goto`/recursion | No recursive functions anywhere in `modeling/`. Every function returns without calling itself, directly or indirectly. |
| Loops must have a fixed, statically provable upper bound | Prefer vectorized numpy/pandas operations — their iteration count is the array length, which is itself checked against `MAX_CANDIDATES_PER_REQUEST` before the operation runs, so it's bounded by construction. Where a plain Python `for` loop is unavoidable, it must iterate over something whose length is fixed at compile time (e.g. a 4-entry scenario dict) or asserted bounded before the loop starts — never a `while True`. |
| No dynamic memory allocation after initialization | Python allocates dynamically by nature, so this becomes: don't grow a list or DataFrame inside a per-request loop. Load reference tables (scenario configs, service lists) once as module-level constants; build each request's working table in one `merge`/`assign` pass, not by appending rows one at a time. |
| Functions ≤ ~60 lines | Enforced in every file here. If a function is approaching that, it's a sign it's doing two things — split it. |
| ≥ 2 assertions per function, on average, boolean and side-effect-free, each with an explicit recovery path | Every function in `modeling/` asserts its own parameters on entry and at least one property of its own result before returning. An assertion failure must be something the caller can act on (a clear message, a typed error) — never a bare `assert True` added just to hit a count. |
| Smallest possible variable scope | Module-level constants only for genuinely fixed configuration (scenario tables, column names, bounds). Everything else declared inside the function that uses it. |
| Check every non-void return value; validate every parameter | Every function here asserts its inputs before doing anything with them, and every call site that uses another function's return value does something with it (assigns it, checks it, or asserts on it) rather than discarding it. |
| Restrict the preprocessor to header inclusion and simple macros | No direct equivalent in Python. Closest translation: no `exec`/`eval`, no metaclasses, no monkey-patching, no dynamically-generated attribute names. If a future file needs any of these, it needs a documented reason in its own header, the same way the original rule requires a comment on every conditional-compilation directive. |
| ≤ 1 level of pointer dereference; no function pointers | No direct equivalent. Closest translation: avoid chaining more than one level of attribute/dict access without naming an intermediate variable (`row["a"]["b"]["c"]` in one line is exactly the thing this rule is really about), and avoid storing bare functions in data structures to call indirectly later — call named functions directly, so control flow is traceable by reading top to bottom. |
| Compile with every warning on, pass static analysis with zero warnings, daily | Python's equivalent of "the compiler": `ruff check` at its strictest settings, and `mypy --strict` (every function here has full type hints for this reason). Both must run clean before a commit touches `modeling/`, not just "when there's time." This hasn't been wired into CI yet — see Open Items below. |

## File layout

```
modeling/
  CLAUDE.md                     <- this file
  candidate_pool/
    candidate_pool.py           <- stage 1: build + gate + tier one patient's candidate pool
  scoring/
    scoring.py                  <- stage 2: entropy weights, severity boost, TOPSIS, given a tier
  recommend/
    recommend.py                <- stage 3: orchestrates 1 and 2 into the final top-N result
  self_checks/
    run_all_checks.py           <- not a pipeline stage; runs every module's own checker in one command
```

Folder names aren't numbered (`01_`, `02_`, ...) because a leading digit makes a directory an
invalid Python package name and breaks `import`. Pipeline order is instead stated as the first
line of every file's header — read that line, not the folder name, to know where a file sits.

## The end-of-file checker

Every pipeline file ends with a `self_check() -> bool` function and an `if __name__ == "__main__":`
block that runs it and exits 1 on failure. Each `self_check()`:

- Builds its own small, fixed, in-memory test data — no file I/O, so a wrong file path can never
  make a check silently pass by finding nothing to check.
- Asserts specific, known-correct outputs, not just "it ran without crashing."
- Prints which individual checks failed if any did, so a failure is diagnosable from the output
  alone.

`modeling/self_checks/run_all_checks.py` calls all three and exits non-zero if any failed — run
this one command (`python3 modeling/self_checks/run_all_checks.py`) rather than each file
individually.

## Open items (not yet done, don't claim otherwise)

- `ruff`/`mypy --strict` are not wired into CI yet — self-checks currently only prove the logic
  is correct on synthetic data, not that the code meets the static-analysis bar this file sets.
- No test yet covers the full national dataset end to end (a smoke test against
  `dataset/blood_points_v2.csv` was run manually while building this pipeline and passed, but
  it isn't automated or re-run on change).
- `detour_factor` and `avg_speed_kmph` (used in `candidate_pool.compute_drive_minutes`) are
  required parameters precisely because straight-line distance is this project's permanent
  distance model — not OSM/live routing, by decision — so there is no "measured" value to default
  to; whoever calls this code chooses the constants and is responsible for them.
