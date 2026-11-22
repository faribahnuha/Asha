"""
==============================================================================
MODULE: hand_written_test_cases.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: formalizes model-design.md (draft)'s
  "Testing without ground truth" section 1 ("Hand-written test cases") as
  actual, runnable assertions against the real recommend() pipeline — that
  section listed 7 cases in prose; none of them existed as code before this
  file. Calls modeling/recommend/recommend.py directly, the same way
  modeling/demo/try_it_out.py does.
- WHAT THIS FILE DOES: builds small, fixed, synthetic candidate tables for
  each of the 7 cases and asserts the documented "obvious right answer"
  actually happens. Each case is its own function, independent of the
  others, printing PASS/FAIL per case plus a final summary.
- KNOWN LIMITATIONS / ASSUMPTIONS: while writing case 5, found that
  model-design.md (draft)'s description ("the unconfirmed one should not
  even enter the ranking while a confirmed candidate exists") describes an
  EARLIER version of recommend.py's tier logic, before the later padding fix
  (made in this same project) that lets a confirmed tier with fewer than
  TOP_N_RESULTS rows pull in reachable-but-unconfirmed candidates to fill
  remaining slots, rather than reaching outside the time window. This file
  tests the CURRENT, correct behavior (confirmed always ranks ahead of
  unconfirmed, but unconfirmed can still fill out the remaining slots) —
  model-design.md (draft)'s case 5 wording should be updated to match.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_MODELING_ROOT = Path(__file__).resolve().parent.parent
for _sibling_dir in ("candidate_pool", "scoring", "recommend"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import recommend as R  # noqa: E402

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

DETOUR_FACTOR: float = 1.3
AVG_SPEED_KMPH: float = 35.0
EMPTY_SERVICES = pd.DataFrame({"unique_id": [], "service": []})


def case_1_single_option_within_range() -> bool:
    """"One blood bank within 100 km -> choose it." A single reachable
    candidate should be recommended directly, no ranking performed.
    """
    candidates = pd.DataFrame({"unique_id": ["only_one"], "lat": [19.00], "lon": [73.00], "blood_level": [2]})
    result, weights = R.recommend(candidates, EMPTY_SERVICES, 19.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = (
        len(result) == 1
        and result.iloc[0]["unique_id"] == "only_one"
        and result.iloc[0]["note"] == R.NOTE_ONLY_OPTION
        and weights == {}
    )
    print(f"case_1_single_option_within_range: {'PASS' if passed else 'FAIL'}")
    return passed


def case_2_only_matching_service_wins() -> bool:
    """"Two equally close, only one does C-sections, pregnant patient ->
    choose that one." With drive times tied, the confirmed facility with
    the matching services should outrank the one with none.
    """
    candidates = pd.DataFrame({
        "unique_id": ["has_services", "no_services"],
        "lat": [19.00, 19.00], "lon": [73.00, 73.00],  # identical location: drive time is tied
        "blood_level": [2, 2],
    })
    services = pd.DataFrame({
        "unique_id": ["has_services", "has_services", "has_services", "no_services"],
        "service": ["c_section", "obstetrics", "icu", "general_medicine"],
    })
    result, weights = R.recommend(candidates, services, 19.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = result.iloc[0]["unique_id"] == "has_services" and "needed_services" in weights
    print(f"case_2_only_matching_service_wins: {'PASS' if passed else 'FAIL'}")
    return passed


def case_3_none_in_window_shows_closest_with_warning() -> bool:
    """"None within the window -> three closest with a warning." When
    nobody is reachable, the closest candidates overall should still be
    returned, each flagged as outside the window.
    """
    candidates = pd.DataFrame({
        "unique_id": ["a", "b", "c"],
        "lat": [19.00, 19.00, 19.00], "lon": [73.00, 73.05, 73.10],
        "blood_level": [2, 2, 2],
    })
    # Patient far enough away that nothing is within the 60-minute window.
    result, weights = R.recommend(candidates, EMPTY_SERVICES, 5.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    expected_note = R.NOTE_OUTSIDE_WINDOW.format(top_n=R.TOP_N_RESULTS)
    passed = len(result) == 3 and (result["note"] == expected_note).all() and weights == {}
    print(f"case_3_none_in_window_shows_closest_with_warning: {'PASS' if passed else 'FAIL'}")
    return passed


def case_4_best_equipped_wins_in_dense_area() -> bool:
    """"City with many options, anemia -> best-equipped wins." With several
    similarly-close confirmed candidates, the one with more of the
    scenario's needed services should rank first, not just the nearest one.
    """
    candidates = pd.DataFrame({
        "unique_id": ["best_equipped", "partial", "nearest_but_bare"],
        "lat": [19.00, 19.00, 19.00], "lon": [73.02, 73.03, 73.00],  # nearest_but_bare is closest
        "blood_level": [2, 2, 2],
    })
    services = pd.DataFrame({
        "unique_id": ["best_equipped", "best_equipped", "partial"],
        "service": ["general_medicine", "icu", "general_medicine"],
    })
    result, weights = R.recommend(candidates, services, 19.00, 73.00, "severe_anemia", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = result.iloc[0]["unique_id"] == "best_equipped"
    print(f"case_4_best_equipped_wins_in_dense_area: {'PASS' if passed else 'FAIL'}")
    return passed


def case_5_confirmed_always_ranks_ahead_of_unconfirmed() -> bool:
    """"One confirmed facility (even with only 1 of 3 needed services) and
    one unconfirmed facility, both reachable -> the confirmed one should
    always be preferred." Verifies ranking PRIORITY, not exclusion — see
    this file's header for why "exclusion" is the stale, pre-padding-fix
    expectation.
    """
    candidates = pd.DataFrame({
        "unique_id": ["confirmed_partial", "unconfirmed"],
        "lat": [19.00, 19.00], "lon": [73.00, 73.02],
        "blood_level": [2, 2],
    })
    services = pd.DataFrame({"unique_id": ["confirmed_partial"], "service": ["icu"]})  # 1 of 3 needed services
    result, weights = R.recommend(candidates, services, 19.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = (
        result.iloc[0]["unique_id"] == "confirmed_partial"
        and bool(result.iloc[0]["needed_services_confirmed"])
        and (len(result) < 2 or not bool(result.iloc[1]["needed_services_confirmed"]))
    )
    print(f"case_5_confirmed_always_ranks_ahead_of_unconfirmed: {'PASS' if passed else 'FAIL'}")
    return passed


def case_6_all_unconfirmed_fallback_ranks_sensibly() -> bool:
    """"Every reachable candidate is unconfirmed -> check the fallback tier
    ranks sensibly on drive time and blood level alone, every result
    labeled 'not confirmed'." Closer + better blood level should still win
    even with no service data anywhere.
    """
    candidates = pd.DataFrame({
        "unique_id": ["near_good_blood", "far_poor_blood", "mid"],
        "lat": [19.00, 19.00, 19.00], "lon": [73.00, 73.08, 73.04],
        "blood_level": [3, 1, 2],
    })
    result, weights = R.recommend(candidates, EMPTY_SERVICES, 19.00, 73.00, "injury_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = (
        result.iloc[0]["unique_id"] == "near_good_blood"
        and (result["note"] == R.NOTE_UNCONFIRMED_FALLBACK).all()
        and "needed_services" not in weights
    )
    print(f"case_6_all_unconfirmed_fallback_ranks_sensibly: {'PASS' if passed else 'FAIL'}")
    return passed


def case_7_severity_boost_has_no_effect_in_fallback_tier() -> bool:
    """"Lower-urgency (anemia) case where only the fallback tier is
    reachable -> confirm the severity boost has no effect, documented, not
    a bug." The anemia boost targets needed_services, which the fallback
    tier never uses.
    """
    candidates = pd.DataFrame({
        "unique_id": ["a", "b", "c"],
        "lat": [19.00, 19.00, 19.00], "lon": [73.00, 73.05, 73.10],
        "blood_level": [3, 2, 1],
    })
    result, weights = R.recommend(candidates, EMPTY_SERVICES, 19.00, 73.00, "severe_anemia", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = "needed_services" not in weights and set(weights.keys()) == {"drive_min", "blood_level"}
    print(f"case_7_severity_boost_has_no_effect_in_fallback_tier: {'PASS' if passed else 'FAIL'}")
    return passed


ALL_CASES = (
    case_1_single_option_within_range,
    case_2_only_matching_service_wins,
    case_3_none_in_window_shows_closest_with_warning,
    case_4_best_equipped_wins_in_dense_area,
    case_5_confirmed_always_ranks_ahead_of_unconfirmed,
    case_6_all_unconfirmed_fallback_ranks_sensibly,
    case_7_severity_boost_has_no_effect_in_fallback_tier,
)


def main() -> int:
    """Entry point. Runs every case in ALL_CASES, prints a summary, and
    returns a process exit code (0 if every case passed, 1 otherwise).
    """
    assert len(ALL_CASES) >= 1, "ALL_CASES must not be empty"
    results = [case_fn() for case_fn in ALL_CASES]
    passed_count = sum(results)
    print(f"\n{passed_count}/{len(results)} hand-written test cases passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
