"""
==============================================================================
MODULE: recommend.py                                         (stage 3 of 3)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: this is the orchestrator. It calls
  modeling/candidate_pool/candidate_pool.py to build and gate one patient's
  candidate pool, then calls modeling/scoring/scoring.py to rank whichever
  tier applies, and assembles the final top-N result. Nothing downstream of
  this file exists yet in modeling/ — its output is what a dashboard or API
  layer would display to the responder.
- WHAT THIS FILE DOES: implements the None / One / Several branching from
  model-design.md (draft)'s "How the model works" step 3, and — inside
  "Several" — the confirmed-tier-first, unconfirmed-tier-fallback branching
  from that same file's "Unknown vs. confirmed-absent" section. This is the
  direct replacement for the single `recommend()` function shown in
  model-design.md (draft)'s "Code" section, split across the three modeling/
  files instead of living in one notebook cell. One rule generalizes across
  all three outcomes: the responder should see TOP_N_RESULTS options
  whenever that many facilities exist anywhere, even if fewer than that are
  within the time window — see _pad_to_top_n() below. Padding never hides
  that a result is outside the window; it's always labeled as such.
- KNOWN LIMITATIONS / ASSUMPTIONS: this file does not fetch or clean data —
  it assumes `blood_candidates` and `facility_services` are already loaded,
  already cleaned, and match the schemas candidate_pool.py expects. It also
  assumes straight-line distance (via detour_factor and avg_speed_kmph) is
  the permanent distance model for this project — see candidate_pool.py's
  header — and that the caller supplies those two values explicitly.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# candidate_pool/ and scoring/ are sibling packages under modeling/, not
# installed packages, so the two sibling directories are added to sys.path
# explicitly and predictably here, in exactly one place, rather than via
# any import-time magic.
_MODELING_ROOT = Path(__file__).resolve().parent.parent
for _sibling_dir in ("candidate_pool", "scoring"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import candidate_pool as cp  # noqa: E402  (see sys.path setup above)
import scoring  # noqa: E402  (see sys.path setup above)

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

# How many ranked facilities to return to the responder.
# Source: model-design.md (draft), "How the model works" step 8.
TOP_N_RESULTS: int = 3

# Note text shown for each of the three outcomes. Centralized here so the
# wording is consistent and reviewable in one place, not scattered through
# the branching logic below.
NOTE_ONLY_OPTION: str = "Only option within the time window"
NOTE_UNCONFIRMED_FALLBACK: str = (
    "No reachable facility has confirmed service data; ranked on drive time "
    "and blood level only — capability not confirmed, call ahead"
)
NOTE_OUTSIDE_WINDOW: str = "Outside the time window for this scenario; shown because fewer than {top_n} were in range"
NOTE_NONE_SO_FAR: str = ""


def recommend(
    blood_candidates: pd.DataFrame,
    facility_services: pd.DataFrame,
    patient_lat: float,
    patient_lon: float,
    scenario: str,
    detour_factor: float,
    avg_speed_kmph: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return the top TOP_N_RESULTS facilities for one patient, plus the
    scoring weights used (empty when no ranking was needed, i.e. the None
    or One outcomes).

    PARAMETERS:
        blood_candidates: one row per facility with blood
            (dataset/blood_points_v2.csv filtered to in_candidate_pool=True),
            must have unique_id, lat, lon, blood_level columns.
        facility_services: one row per (unique_id, service) pair
            (dataset/facility_services_v2.csv schema).
        patient_lat, patient_lon: the responder's reported location.
        scenario: must be a key of cp.URGENCY_BY_SCENARIO.
        detour_factor, avg_speed_kmph: measured values, forwarded to
            cp.compute_drive_minutes() — see that function's docstring.
    """
    assert scenario in cp.URGENCY_BY_SCENARIO, f"unknown scenario: {scenario}"
    assert len(blood_candidates) <= cp.MAX_CANDIDATES_PER_REQUEST, "candidate table exceeds the fixed upper bound"
    assert "blood_level" in blood_candidates.columns, "blood_candidates is missing blood_level"

    clean_candidates = cp.drop_bad_coordinates(blood_candidates)
    with_services = cp.attach_needed_services(clean_candidates, facility_services, scenario)
    with_services["drive_min"] = cp.compute_drive_minutes(
        patient_lat, patient_lon, with_services["lat"], with_services["lon"], detour_factor, avg_speed_kmph
    )
    reachable_pool = cp.filter_within_time_window(with_services, scenario)

    if reachable_pool.empty:
        ranked, weights = with_services.iloc[0:0].copy(), {}
    elif len(reachable_pool) == 1:
        ranked, weights = _handle_single_option(reachable_pool), {}
    else:
        ranked, weights = _rank_several(reachable_pool, scenario)

    result = _pad_to_top_n(ranked, with_services)
    assert len(result) <= TOP_N_RESULTS, "recommend() must never return more than TOP_N_RESULTS rows"
    return result, weights


def _pad_to_top_n(ranked: pd.DataFrame, with_services: pd.DataFrame) -> pd.DataFrame:
    """If `ranked` has fewer than TOP_N_RESULTS rows, fill the remaining
    slots with the closest candidates NOT already in `ranked` — regardless
    of whether they fall inside the time window — each clearly labeled as
    outside the window. This is the one rule that applies whether the real
    outcome was None, One, or Several-but-fewer-than-three reachable: the
    responder should see TOP_N_RESULTS options whenever that many exist at
    all, and padding never hides that a result is outside the window.

    PARAMETERS:
        ranked: already-scored/noted results, 0 to TOP_N_RESULTS rows.
            May be empty (the None-reachable case).
        with_services: the full (unwindowed) candidate table for this
            patient, must have a "drive_min" column and the same columns
            as `ranked`.
    """
    assert len(ranked) <= TOP_N_RESULTS, "_pad_to_top_n() received more than TOP_N_RESULTS rows to start with"
    assert "drive_min" in with_services.columns, "with_services is missing drive_min"

    remaining_slots = TOP_N_RESULTS - len(ranked)
    if remaining_slots <= 0:
        return ranked

    already_chosen_ids = set(ranked["unique_id"]) if "unique_id" in ranked.columns else set()
    not_yet_chosen = with_services[~with_services["unique_id"].isin(already_chosen_ids)]
    if not_yet_chosen.empty:
        return ranked  # nothing left anywhere to pad with

    padding = not_yet_chosen.nsmallest(remaining_slots, "drive_min").copy()
    padding["score"] = 0.0
    padding["note"] = NOTE_OUTSIDE_WINDOW.format(top_n=TOP_N_RESULTS)

    combined = pd.concat([ranked, padding], ignore_index=True)
    assert len(combined) <= TOP_N_RESULTS, "padding must never push the result past TOP_N_RESULTS"
    return combined


def _handle_single_option(reachable_pool: pd.DataFrame) -> pd.DataFrame:
    """Outcome: exactly one reachable facility. No ranking needed.

    PARAMETERS:
        reachable_pool: must contain exactly one row.
    """
    assert len(reachable_pool) == 1, "_handle_single_option() requires exactly one row"
    result = reachable_pool.copy()
    result["score"] = 1.0
    result["note"] = NOTE_ONLY_OPTION
    return result


def _rank_one_tier(
    tier: pd.DataFrame, urgency: str, criteria: tuple[str, ...], is_cost, fallback_note: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rank one non-empty tier (confirmed or unconfirmed) on its own
    criteria, labeling every row with `fallback_note` (empty string for the
    confirmed tier). Returns (ranked_tier, weights) — weights are empty when
    the tier has only one row, since nothing was actually ranked.

    PARAMETERS:
        tier: must contain at least one row.
        urgency: passed to scoring.apply_severity_boost().
        criteria, is_cost: this tier's criteria list and cost/benefit flags.
        fallback_note: text to attach to every row in this tier's output.
    """
    assert len(tier) >= 1, "_rank_one_tier() requires at least one row"

    if len(tier) == 1:
        return tier.assign(score=1.0, note=fallback_note), {}

    ranked = scoring.rank_tier(tier, urgency, criteria, is_cost).assign(note=fallback_note)
    weights = scoring.apply_severity_boost(
        scoring.entropy_weights(tier[list(criteria)].to_numpy(dtype=float)), urgency, criteria
    )
    return ranked, dict(zip(criteria, weights.round(3)))


def _rank_several(reachable_pool: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Outcome: several reachable facilities. Rank the confirmed tier first;
    if there are still fewer than TOP_N_RESULTS results, fill the remaining
    slots from the unconfirmed tier (still within the time window, just
    lower priority) BEFORE recommend()'s outer padding step ever reaches
    outside the window — a reachable-but-unconfirmed facility must never be
    mislabeled as unreachable. Confirmed-tier weights are used whenever the
    confirmed tier contributed anything; the unconfirmed tier never scores
    on a guessed value for the criterion it's missing.

    PARAMETERS:
        reachable_pool: must contain at least two rows and the
            cp.CONFIRMED_FLAG_COLUMN boolean column.
        scenario: must be a key of cp.URGENCY_BY_SCENARIO.
    """
    assert len(reachable_pool) >= 2, "_rank_several() requires at least two candidates"
    assert cp.CONFIRMED_FLAG_COLUMN in reachable_pool.columns, "reachable_pool is missing the confirmed flag"

    urgency = cp.URGENCY_BY_SCENARIO[scenario]
    confirmed_tier, unconfirmed_tier = cp.split_confirmed_unconfirmed(reachable_pool)

    ranked_confirmed, weights = (
        _rank_one_tier(confirmed_tier, urgency, scoring.CONFIRMED_TIER_CRITERIA, scoring.CONFIRMED_TIER_IS_COST, NOTE_NONE_SO_FAR)
        if not confirmed_tier.empty
        else (confirmed_tier, {})
    )

    if len(ranked_confirmed) < TOP_N_RESULTS and not unconfirmed_tier.empty:
        ranked_unconfirmed, unconfirmed_weights = _rank_one_tier(
            unconfirmed_tier, urgency, scoring.UNCONFIRMED_TIER_CRITERIA, scoring.UNCONFIRMED_TIER_IS_COST, NOTE_UNCONFIRMED_FALLBACK
        )
        if not weights:  # confirmed tier contributed nothing: fall back to unconfirmed weights
            weights = unconfirmed_weights
    else:
        ranked_unconfirmed = unconfirmed_tier.iloc[0:0]  # empty, same columns

    combined = pd.concat([ranked_confirmed, ranked_unconfirmed], ignore_index=True)
    top_results = combined.head(TOP_N_RESULTS)
    assert len(top_results) <= TOP_N_RESULTS, "must never return more than TOP_N_RESULTS rows"
    return top_results, weights


def self_check() -> bool:
    """End-of-file checker. Runs recommend() through each of its three
    outcomes (None / One / Several) on small, fixed, in-memory data, plus
    the confirmed-vs-unconfirmed branch inside "Several", plus the padding
    rule that applies across all three. Returns True only if every check
    passes.
    """
    # lon spacing of 0.05 deg (~5.5 km at this latitude) keeps all three within
    # the 60-minute pregnancy window at a normal driving speed.
    blood_candidates = pd.DataFrame(
        {
            "unique_id": ["near", "mid", "far"],
            "lat": [19.00, 19.00, 19.00],
            "lon": [73.00, 73.05, 73.10],
            "blood_level": [2, 2, 3],
        }
    )
    facility_services_empty = pd.DataFrame({"unique_id": [], "service": []})
    # Two confirmed candidates ("near", "mid") so the confirmed tier has
    # enough rows to actually be ranked, not just returned as a single option.
    facility_services_with_data = pd.DataFrame({"unique_id": ["near", "mid"], "service": ["icu", "icu"]})

    # Outcome: None reachable. Patient is ~1,500 km from every candidate, so
    # none falls inside the 60-minute window at any normal driving speed.
    # All TOP_N_RESULTS slots must still be filled from padding, each flagged.
    none_result, none_weights = recommend(
        blood_candidates, facility_services_empty, 5.00, 73.00, "pregnancy_bleeding", 1.3, 30.0
    )
    expected_outside_note = NOTE_OUTSIDE_WINDOW.format(top_n=TOP_N_RESULTS)
    check_1 = (
        none_weights == {}
        and len(none_result) == TOP_N_RESULTS
        and (none_result["note"] == expected_outside_note).all()
    )

    # Outcome: exactly One reachable (only "near" is in the candidate table at all)
    # -> but blood_candidates as a whole has 3 rows, so padding should still fill
    # the other two slots from "mid" and "far", outside the window and flagged.
    one_candidate_reachable = blood_candidates[blood_candidates["unique_id"] == "near"]
    padded_with_full_table, one_weights = recommend(
        blood_candidates, facility_services_empty, 19.00, 73.00, "pregnancy_bleeding", 1.3, 0.05
    )
    # At this crawl-speed, only "near" (0 km away) is inside the window; "mid"/"far" are not.
    check_2 = (
        one_weights == {}
        and len(padded_with_full_table) == TOP_N_RESULTS
        and (padded_with_full_table["unique_id"] == "near").sum() == 1
        and padded_with_full_table[padded_with_full_table["unique_id"] == "near"].iloc[0]["note"] == NOTE_ONLY_OPTION
        and (padded_with_full_table[padded_with_full_table["unique_id"] != "near"]["note"] == expected_outside_note).all()
    )
    assert one_candidate_reachable is not None, "sanity check: variable must be used"  # keeps this scenario documented

    # Outcome: Several reachable, nobody confirmed -> unconfirmed fallback tier is used.
    several_unconfirmed_result, several_unconfirmed_weights = recommend(
        blood_candidates, facility_services_empty, 19.00, 73.00, "pregnancy_bleeding", 1.3, 30.0
    )
    check_3 = (
        "needed_services" not in several_unconfirmed_weights
        and len(several_unconfirmed_result) == TOP_N_RESULTS
        and (several_unconfirmed_result["note"] == NOTE_UNCONFIRMED_FALLBACK).all()
    )

    # Outcome: Several reachable, two confirmed candidates exist -> confirmed tier is
    # ranked on all three criteria; the unconfirmed candidate ("far") is reachable too,
    # so no padding is needed and "far" simply ranks (not pads) behind the confirmed two.
    several_confirmed_result, several_confirmed_weights = recommend(
        blood_candidates, facility_services_with_data, 19.00, 73.00, "pregnancy_bleeding", 1.3, 30.0
    )
    check_4 = "needed_services" in several_confirmed_weights and len(several_confirmed_result) == TOP_N_RESULTS

    all_checks_passed = all([check_1, check_2, check_3, check_4])
    if not all_checks_passed:
        print("recommend.self_check(): FAILED", [check_1, check_2, check_3, check_4])
    return all_checks_passed


if __name__ == "__main__":
    passed = self_check()
    print("recommend.self_check():", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)
