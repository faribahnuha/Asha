"""
==============================================================================
MODULE: validate_against_reference_implementation.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: tests modeling/scoring/scoring.py's
  entropy_weights() and topsis_scores() against an independent, third-party
  implementation (the `pymcdm` library) rather than only against the
  hand-written cases in scoring.py's own self_check(). Those self-checks can
  only catch bugs I thought to write a test for; this catches a bug in the
  formula itself, the kind self-checking your own code can't.
- WHAT THIS FILE DOES: builds a fixed decision matrix, computes weights and
  scores with both our implementation and pymcdm's, and asserts they match
  once equivalent conventions are used (pymcdm's `types` is 1/-1 for
  profit/cost, not a boolean; its TOPSIS defaults to min-max normalization,
  not the vector normalization the original 1981 Hwang & Yoon paper — and
  our code — uses). Also documents a real, confirmed difference: pymcdm's
  entropy_weights() zeroes an ENTIRE column's entropy if any single
  candidate has a 0 there, which is cruder than the standard formula ours
  uses (treating only that one term as 0, per the limit p*log(p) -> 0).
- KNOWN LIMITATIONS / ASSUMPTIONS: requires the `pymcdm` package
  (pip install pymcdm). This is a one-off verification, not part of the
  modeling/ pipeline — nothing here runs at request time.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_MODELING_ROOT = Path(__file__).resolve().parent.parent
_SCORING_PATH = str(_MODELING_ROOT / "scoring")
if _SCORING_PATH not in sys.path:
    sys.path.append(_SCORING_PATH)

import scoring  # noqa: E402

try:
    from pymcdm.methods import TOPSIS
    from pymcdm import weights as mcdm_weights
    from pymcdm.normalizations import vector_normalization
except ImportError:
    print("pymcdm is not installed. Run: pip install pymcdm")
    sys.exit(1)

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

# A fixed 5-candidate x 3-criteria matrix, no zeros, so the comparison isn't
# confounded by pymcdm's cruder zero-handling (documented above). 1.0 is the
# smallest needed_services value to keep this test isolated to the formula
# itself, not that known edge case.
NO_ZEROS_MATRIX: np.ndarray = np.array([
    [12.0, 2.0, 1.0],
    [45.0, 3.0, 3.0],
    [8.0, 1.0, 1.0],
    [30.0, 2.0, 2.0],
    [60.0, 3.0, 2.0],
])
IS_COST = np.array([True, False, False])  # drive_min is cost; blood_level, needed_services are benefit
FLOATING_POINT_TOLERANCE: float = 1e-9


def validate_entropy_weights(matrix: np.ndarray, is_cost: np.ndarray) -> bool:
    """Compare our entropy_weights() against pymcdm's on a matrix with no
    zero values. Returns True if they match within floating-point tolerance.

    PARAMETERS:
        matrix: decision matrix, candidates x criteria, no zero values.
        is_cost: unused by entropy weighting itself (entropy doesn't care
            about cost/benefit direction) — accepted for a consistent
            function signature across this file's validators.
    """
    assert matrix.ndim == 2, "matrix must be 2-D"
    assert not np.any(matrix == 0), "this check requires a zero-free matrix; see this file's header"

    our_weights = scoring.entropy_weights(matrix)
    reference_weights = mcdm_weights.entropy_weights(matrix)

    max_diff = float(np.max(np.abs(our_weights - reference_weights)))
    matched = max_diff < FLOATING_POINT_TOLERANCE
    print(f"entropy_weights: our={np.round(our_weights, 6)} pymcdm={np.round(reference_weights, 6)} "
          f"max_diff={max_diff:.2e} -> {'MATCH' if matched else 'MISMATCH'}")
    return matched


def validate_topsis_scores(matrix: np.ndarray, is_cost: np.ndarray) -> bool:
    """Compare our topsis_scores() against pymcdm's TOPSIS, forced to use
    vector normalization (the original Hwang & Yoon 1981 convention, and
    what our implementation uses) instead of pymcdm's min-max default.

    PARAMETERS:
        matrix: decision matrix, candidates x criteria.
        is_cost: boolean array, True where lower is better.
    """
    assert matrix.ndim == 2, "matrix must be 2-D"
    pymcdm_types = np.where(is_cost, -1, 1)  # pymcdm convention: 1=profit, -1=cost

    shared_weights = scoring.entropy_weights(matrix)  # use the SAME weights for both, isolating normalization
    our_scores = scoring.topsis_scores(matrix, shared_weights, is_cost)

    reference_method = TOPSIS(normalization_function=vector_normalization)
    reference_scores = reference_method(matrix, shared_weights, pymcdm_types)

    max_diff = float(np.max(np.abs(our_scores - reference_scores)))
    matched = max_diff < FLOATING_POINT_TOLERANCE
    print(f"topsis_scores:   our={np.round(our_scores, 6)} pymcdm={np.round(reference_scores, 6)} "
          f"max_diff={max_diff:.2e} -> {'MATCH' if matched else 'MISMATCH'}")
    return matched


def document_known_zero_handling_difference() -> None:
    """Not a pass/fail check — prints the confirmed, understood difference
    between our entropy formula and pymcdm's when a criterion has a zero
    value for some candidate, so this discrepancy is recorded rather than
    silently rediscovered later.
    """
    matrix_with_zero = NO_ZEROS_MATRIX.copy()
    matrix_with_zero[2, 2] = 0.0  # one candidate has needed_services = 0

    our_weights = scoring.entropy_weights(matrix_with_zero)
    pymcdm_weights_with_zero = mcdm_weights.entropy_weights(matrix_with_zero)
    print("\n--- Known, understood difference: a criterion with one zero value ---")
    print(f"our entropy weights:    {np.round(our_weights, 4)}")
    print(f"pymcdm entropy weights: {np.round(pymcdm_weights_with_zero, 4)}")
    print("pymcdm zeroes an ENTIRE column's entropy if ANY candidate has a 0 there "
          "(see pymcdm.weights.entropy_weights source: `if np.any(col == 0): entropies[i] = 0`). "
          "Ours treats only that one p*log(p) term as 0 (the standard convention, "
          "since lim p->0 of p*log(p) = 0), not the whole column. This is why they "
          "diverge whenever a criterion has a partial zero — confirmed by this file, "
          "not assumed.")


def main() -> int:
    """Entry point. Runs both checks, prints results, exits 1 if either fails."""
    print("=== Validating modeling/scoring/scoring.py against pymcdm (independent reference) ===\n")
    entropy_matched = validate_entropy_weights(NO_ZEROS_MATRIX, IS_COST)
    topsis_matched = validate_topsis_scores(NO_ZEROS_MATRIX, IS_COST)
    document_known_zero_handling_difference()

    all_matched = entropy_matched and topsis_matched
    print(f"\n{'ALL CHECKS MATCHED THE REFERENCE IMPLEMENTATION' if all_matched else 'MISMATCH FOUND — investigate before trusting scoring.py'}")
    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(main())
