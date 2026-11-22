"""
==============================================================================
MODULE: scoring.py                                           (stage 2 of 3)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: takes the output of
  modeling/candidate_pool/candidate_pool.py — one tier (confirmed OR
  unconfirmed) of a patient's reachable candidates, already filtered to the
  time window — and ranks it. It does not know or care how its input tier
  was built; it only trusts the criteria columns it is told to use.
- WHAT THIS FILE DOES: implements the entropy-weighted TOPSIS scoring from
  model-design.md (draft)'s "Code" section, generalized to accept a
  DIFFERENT criteria list for the confirmed tier (drive_min, blood_level,
  needed_services) versus the unconfirmed fallback tier (drive_min,
  blood_level only) — see model-design.md (draft)'s "Unknown vs.
  confirmed-absent" section for why those two tiers use different criteria
  instead of one tier scoring a guessed needed_services value.
- KNOWN LIMITATIONS / ASSUMPTIONS: entropy weighting measures how much a
  criterion VARIES across this one patient's candidates, not how much it
  matters clinically — the severity boost is the only clinical override, and
  it is a stated judgment call (x2), not a measured quantity. See
  model-design.md (draft)'s "Weaknesses and mitigations" table.
==============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

# The three criteria used when ranking the CONFIRMED tier.
# Source: model-design.md (draft), "Criteria" table.
CONFIRMED_TIER_CRITERIA: tuple[str, ...] = ("drive_min", "blood_level", "needed_services")
CONFIRMED_TIER_IS_COST: np.ndarray = np.array([True, False, False])  # True: lower is better

# The two criteria used when ranking the UNCONFIRMED (fallback) tier.
# needed_services is deliberately absent — see model-design.md (draft).
UNCONFIRMED_TIER_CRITERIA: tuple[str, ...] = ("drive_min", "blood_level")
UNCONFIRMED_TIER_IS_COST: np.ndarray = np.array([True, False])

# Which criterion each severity level boosts, and by how much.
# A stated choice (model-design.md (draft) says: test at x1.5 and x3 too).
SEVERITY_BOOST_CRITERION: dict[str, str] = {"high": "drive_min", "lower": "needed_services"}
SEVERITY_BOOST_FACTOR: float = 2.0

# Must match candidate_pool.py's MAX_CANDIDATES_PER_REQUEST: a tier being
# scored is a subset of one patient's reachable pool, so it can never
# legitimately be larger than the pool itself. Same reasoning as there —
# this is a bug tripwire, not a cap on how many facilities get scored.
MAX_CANDIDATES_PER_TIER: int = 3500

# TOPSIS adds this to the denominator so a tier where every candidate is
# equally close to both the ideal and the worst facility does not divide by
# zero. Small enough to never visibly change a real score.
TOPSIS_EPSILON: float = 1e-12


def entropy_weights(criteria_matrix: np.ndarray) -> np.ndarray:
    """Weight each criterion (column) by how much it varies across the
    candidates (rows): a criterion that is identical for everyone carries no
    information and gets (close to) zero weight.

    PARAMETERS:
        criteria_matrix: shape (num_candidates, num_criteria), all values
            must be finite and non-negative (TOPSIS cost/benefit convention).
    """
    assert criteria_matrix.ndim == 2, "criteria_matrix must be 2-D (candidates x criteria)"
    assert criteria_matrix.shape[0] >= 1, "at least one candidate is required"
    assert np.isfinite(criteria_matrix).all(), "criteria_matrix contains NaN or infinite values"

    num_candidates = criteria_matrix.shape[0]
    column_sums = criteria_matrix.sum(axis=0)
    proportions = np.divide(
        criteria_matrix, column_sums, out=np.zeros_like(criteria_matrix, dtype=float), where=column_sums > 0
    )
    plogp = np.where(proportions > 0, proportions * np.log(np.where(proportions > 0, proportions, 1.0)), 0.0)

    if num_candidates == 1:
        # A single candidate has no spread to measure; every criterion is
        # equally uninformative, so weights fall back to a plain average.
        return np.full(criteria_matrix.shape[1], 1.0 / criteria_matrix.shape[1])

    entropy = -plogp.sum(axis=0) / np.log(num_candidates)
    entropy[column_sums == 0] = 1.0  # criterion absent for every candidate: carries no information
    spread = 1.0 - entropy

    if spread.sum() == 0:
        return np.full(criteria_matrix.shape[1], 1.0 / criteria_matrix.shape[1])
    weights = spread / spread.sum()
    assert np.isclose(weights.sum(), 1.0), "entropy weights must sum to 1"
    return weights


def apply_severity_boost(entropy_based_weights: np.ndarray, urgency: str, criteria: tuple[str, ...]) -> np.ndarray:
    """Boost the weight of the criterion that matters most for this
    severity, then rescale so the weights still sum to 1. If that criterion
    is not one of this tier's criteria (the unconfirmed tier has no
    needed_services), no boost is applied — plain entropy weights pass
    through unchanged; this is documented, not a silent no-op bug.

    PARAMETERS:
        entropy_based_weights: output of entropy_weights(), same length as
            `criteria`.
        urgency: must be a key of SEVERITY_BOOST_CRITERION ("high" or "lower").
        criteria: the criteria list in use for this tier.
    """
    assert urgency in SEVERITY_BOOST_CRITERION, f"unknown urgency level: {urgency}"
    assert len(entropy_based_weights) == len(criteria), "weights and criteria length must match"

    boosted_weights = entropy_based_weights.copy()
    boost_criterion = SEVERITY_BOOST_CRITERION[urgency]
    if boost_criterion in criteria:
        boosted_weights[criteria.index(boost_criterion)] *= SEVERITY_BOOST_FACTOR

    rescaled_weights = boosted_weights / boosted_weights.sum()
    assert np.isclose(rescaled_weights.sum(), 1.0), "rescaled weights must sum to 1"
    return rescaled_weights


def topsis_scores(criteria_matrix: np.ndarray, weights: np.ndarray, is_cost: np.ndarray) -> np.ndarray:
    """Score each candidate from 0 to 1: closeness to an ideal facility
    (best value on every criterion) and distance from the worst one.

    PARAMETERS:
        criteria_matrix: shape (num_candidates, num_criteria).
        weights: shape (num_criteria,), must sum to 1.
        is_cost: shape (num_criteria,) booleans; True means lower is better.
    """
    assert criteria_matrix.shape[1] == len(weights) == len(is_cost), "shape mismatch between inputs"
    assert np.isclose(weights.sum(), 1.0), "weights must sum to 1 before scoring"

    column_norms = np.sqrt((criteria_matrix**2).sum(axis=0))
    normalized = np.divide(
        criteria_matrix, column_norms, out=np.zeros_like(criteria_matrix, dtype=float), where=column_norms > 0
    )
    weighted = normalized * weights

    ideal_best = np.where(is_cost, weighted.min(axis=0), weighted.max(axis=0))
    ideal_worst = np.where(is_cost, weighted.max(axis=0), weighted.min(axis=0))
    distance_to_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    distance_to_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    scores = distance_to_worst / (distance_to_best + distance_to_worst + TOPSIS_EPSILON)
    assert bool(((scores >= 0.0) & (scores <= 1.0 + TOPSIS_EPSILON)).all()), "TOPSIS score out of [0, 1]"
    return scores


def rank_tier(tier: pd.DataFrame, urgency: str, criteria: tuple[str, ...], is_cost: np.ndarray) -> pd.DataFrame:
    """Score and rank every candidate in one tier, highest score first.

    PARAMETERS:
        tier: must contain every column named in `criteria`, and at least
            one row.
        urgency: passed to apply_severity_boost().
        criteria: CONFIRMED_TIER_CRITERIA or UNCONFIRMED_TIER_CRITERIA.
        is_cost: the matching *_IS_COST array for `criteria`.
    """
    assert len(tier) >= 1, "rank_tier() requires at least one candidate"
    assert len(tier) <= MAX_CANDIDATES_PER_TIER, "tier exceeds the fixed upper bound"
    missing_columns = [column for column in criteria if column not in tier.columns]
    assert not missing_columns, f"tier is missing required columns: {missing_columns}"

    criteria_matrix = tier[list(criteria)].to_numpy(dtype=float)
    weights = apply_severity_boost(entropy_weights(criteria_matrix), urgency, criteria)
    scores = topsis_scores(criteria_matrix, weights, is_cost)

    ranked = tier.assign(score=scores).sort_values("score", ascending=False)
    assert len(ranked) == len(tier), "ranking must not change the row count"
    return ranked


def self_check() -> bool:
    """End-of-file checker. Verifies entropy_weights(), apply_severity_boost(),
    topsis_scores(), and rank_tier() on small, fixed, in-memory inputs.
    Returns True only if every check passes.
    """
    all_identical_columns = np.array([[5.0, 1.0], [5.0, 1.0], [5.0, 1.0]])
    weights_1 = entropy_weights(all_identical_columns)
    check_1 = np.isclose(weights_1.sum(), 1.0) and np.allclose(weights_1, 0.5)  # no spread anywhere: even split

    varied_columns = np.array([[10.0, 1.0], [20.0, 1.0], [30.0, 1.0]])
    weights_2 = entropy_weights(varied_columns)
    check_2 = weights_2[0] > weights_2[1]  # column 0 varies, column 1 doesn't: more weight on column 0

    boosted = apply_severity_boost(np.array([0.5, 0.5]), "high", ("drive_min", "blood_level"))
    check_3 = boosted[0] > boosted[1]  # "high" boosts drive_min, which is criteria[0] here

    unboosted = apply_severity_boost(np.array([0.5, 0.5]), "lower", ("drive_min", "blood_level"))
    check_4 = np.isclose(unboosted[0], unboosted[1])  # "lower" boosts needed_services: absent, no effect here

    scores = topsis_scores(np.array([[5.0, 3.0], [50.0, 1.0]]), np.array([0.5, 0.5]), np.array([True, False]))
    check_5 = scores[0] > scores[1]  # row 0 has a lower (better) cost value and a higher benefit value

    tier = pd.DataFrame({"drive_min": [10.0, 40.0], "blood_level": [3.0, 1.0]})
    ranked = rank_tier(tier, "high", UNCONFIRMED_TIER_CRITERIA, UNCONFIRMED_TIER_IS_COST)
    check_6 = ranked.iloc[0]["drive_min"] == 10.0  # closer + better blood level should rank first

    all_checks_passed = all([check_1, check_2, check_3, check_4, check_5, check_6])
    if not all_checks_passed:
        print("scoring.self_check(): FAILED", [check_1, check_2, check_3, check_4, check_5, check_6])
    return all_checks_passed


if __name__ == "__main__":
    import sys

    passed = self_check()
    print("scoring.self_check():", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)
