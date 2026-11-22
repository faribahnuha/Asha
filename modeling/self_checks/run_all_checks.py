"""
==============================================================================
MODULE: run_all_checks.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: this is NOT a pipeline stage — it does
  not sit between candidate_pool.py, scoring.py, and recommend.py the way
  they sit relative to each other. It is a single entry point that calls
  each of those three files' own self_check() function and reports one
  combined pass/fail, for a CI step or a pre-commit hook to run in one
  command instead of three.
- WHAT THIS FILE DOES: imports the three pipeline modules, calls each
  module's self_check(), and prints a summary. Exits 0 only if every module
  passed.
- KNOWN LIMITATIONS / ASSUMPTIONS: this only runs the fixed, in-memory
  checks each module carries at its own end — it does not load or validate
  the real dataset CSVs. See the integration note at the bottom of
  modeling/CLAUDE.md for how to smoke-test against real data.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODELING_ROOT = Path(__file__).resolve().parent.parent
for _sibling_dir in ("candidate_pool", "scoring", "recommend", "capability", "success"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import candidate_pool  # noqa: E402  (see sys.path setup above)
import scoring  # noqa: E402  (see sys.path setup above)
import recommend  # noqa: E402  (see sys.path setup above)
import capability_model  # noqa: E402  (see sys.path setup above)
import success_model  # noqa: E402  (see sys.path setup above)
import blood_access  # noqa: E402  (see sys.path setup above)
import recommend_v2  # noqa: E402  (see sys.path setup above)

# Fixed, known list of (module_name, module) pairs to check. Not discovered
# dynamically at runtime — adding a new modeling/ module means adding one
# line here, on purpose, so this file can never silently check zero modules.
MODULES_TO_CHECK: tuple[tuple[str, object], ...] = (
    ("candidate_pool", candidate_pool),
    ("scoring", scoring),
    ("recommend", recommend),
    ("capability_model", capability_model),
    ("success_model", success_model),
    ("blood_access", blood_access),
    ("recommend_v2", recommend_v2),
)


def run_all_checks() -> bool:
    """Call self_check() on every module in MODULES_TO_CHECK and print a
    pass/fail line for each. Returns True only if every module passed.
    """
    assert len(MODULES_TO_CHECK) >= 1, "MODULES_TO_CHECK must never be empty"

    all_passed = True
    for module_name, module in MODULES_TO_CHECK:
        assert hasattr(module, "self_check"), f"{module_name} has no self_check() function"
        module_passed = module.self_check()
        print(f"  {module_name}.self_check(): {'PASSED' if module_passed else 'FAILED'}")
        all_passed = all_passed and module_passed

    assert isinstance(all_passed, bool), "all_passed must be a plain boolean"
    return all_passed


if __name__ == "__main__":
    print("Running all modeling/ self-checks:")
    passed = run_all_checks()
    print("ALL CHECKS PASSED" if passed else "ONE OR MORE CHECKS FAILED")
    sys.exit(0 if passed else 1)
