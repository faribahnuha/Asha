"""
==============================================================================
MODULE: sensitivity_sweep.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: formalizes model-design.md (draft)'s
  "Testing without ground truth" section 2 ("Sensitivity checks") — rerun
  the test cases with windows of 45/60/75 minutes and boosts of x1.5/2/3,
  report how often the top pick changes. That was a plan in prose; this is
  the actual, runnable version of it, against real sampled locations rather
  than only the hand-written cases.
- WHAT THIS FILE DOES: temporarily overrides candidate_pool.py's high-
  urgency time window and scoring.py's severity boost factor (restoring
  the originals afterward — never a permanent change to the pipeline),
  reruns a fixed set of real sampled locations at each setting, and
  measures how often the #1 recommendation changes versus the baseline
  (60 min, x2.0).
- KNOWN LIMITATIONS / ASSUMPTIONS: only the "high" urgency window is swept
  (45/60/75), matching model-design.md (draft)'s literal wording — the
  "lower" (anemia) window stays at 120 throughout, which doubles as a
  built-in sanity check: the anemia scenario's top pick should barely move
  under the window sweep, since its own window never changes.
==============================================================================
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELING_ROOT = _REPO_ROOT / "modeling"
for _sibling_dir in ("candidate_pool", "scoring", "recommend", "demo"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import candidate_pool as cp  # noqa: E402
import scoring  # noqa: E402
import recommend as R  # noqa: E402
import try_it_out as demo  # noqa: E402

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

RANDOM_SEED: int = 11
N_LOCATIONS: int = 40
BASELINE_HIGH_WINDOW: int = 60
BASELINE_BOOST_FACTOR: float = 2.0
WINDOW_VALUES_TO_TEST = (45, 60, 75)
BOOST_VALUES_TO_TEST = (1.5, 2.0, 3.0)
TEST_SCENARIOS = ("injury_bleeding", "severe_anemia")  # one high-urgency, one lower-urgency


@contextmanager
def temporary_high_window(minutes: int):
    """Temporarily override the "high" urgency time window, restoring the
    original value no matter how the `with` block exits.

    PARAMETERS:
        minutes: the window value to use for the duration of the block.
    """
    assert minutes > 0, "window must be positive"
    original_value = cp.TIME_WINDOW_MINUTES_BY_URGENCY["high"]
    cp.TIME_WINDOW_MINUTES_BY_URGENCY["high"] = minutes
    try:
        yield
    finally:
        cp.TIME_WINDOW_MINUTES_BY_URGENCY["high"] = original_value


@contextmanager
def temporary_boost_factor(factor: float):
    """Temporarily override the severity boost factor, restoring the
    original value no matter how the `with` block exits.

    PARAMETERS:
        factor: the boost multiplier to use for the duration of the block.
    """
    assert factor >= 1.0, "a boost factor below 1.0 would de-emphasize the boosted criterion, not boost it"
    original_value = scoring.SEVERITY_BOOST_FACTOR
    scoring.SEVERITY_BOOST_FACTOR = factor
    try:
        yield
    finally:
        scoring.SEVERITY_BOOST_FACTOR = original_value


def sample_locations(blood_points, scenario_key: str, n: int, rng: np.random.Generator):
    """Sample n jittered locations near real facilities — same reasoning as
    distance_only_sensitivity.py's sampler: realistic patient locations,
    not uniformly random across India.
    """
    pool = blood_points[blood_points["in_candidate_pool"] == True].dropna(subset=["lat", "lon"])  # noqa: E712
    assert len(pool) >= n, "not enough candidates to sample from"
    sampled = pool.sample(n=n, random_state=rng.integers(0, 1_000_000))
    return [
        (row["lat"] + rng.uniform(-0.1, 0.1), row["lon"] + rng.uniform(-0.1, 0.1))
        for _, row in sampled.iterrows()
    ]


def top_pick_for_each_location(blood_points, facility_services, scenario_key: str, locations) -> list:
    """Run recommend() for every location and return an identifying key for
    the #1 facility (or None if nothing was returned at all).

    Returns (lat, lon) tuples, not the unique_id column — ~85% of real
    candidates (directory-sourced blood banks) have unique_id = NaN, which
    would make two DIFFERENT facilities look identical to a same-ness
    comparison. Coordinates were already validated non-missing earlier in
    the pipeline (candidate_pool.drop_bad_coordinates), so they reliably
    identify a specific facility here.
    """
    top_picks = []
    for patient_lat, patient_lon in locations:
        nearby = demo.narrow_to_nearby(blood_points, scenario_key, patient_lat, patient_lon)
        if nearby.empty:
            top_picks.append(None)
            continue
        result, _ = R.recommend(nearby, facility_services, patient_lat, patient_lon, scenario_key,
                                 demo.ASSUMED_DETOUR_FACTOR, demo.ASSUMED_AVG_SPEED_KMPH)
        top_picks.append((round(float(result.iloc[0]["lat"]), 6), round(float(result.iloc[0]["lon"]), 6)) if len(result) >= 1 else None)
    assert len(top_picks) == len(locations), "must produce one top pick per location"
    return top_picks


def pct_changed_from_baseline(baseline_picks: list, variant_picks: list) -> float:
    """What fraction of locations got a different #1 pick under the variant
    setting compared to the baseline setting.
    """
    assert len(baseline_picks) == len(variant_picks), "baseline and variant must cover the same locations"
    assert len(baseline_picks) > 0, "cannot compute a percentage over zero locations"
    changed = sum(1 for b, v in zip(baseline_picks, variant_picks) if b != v)
    return 100.0 * changed / len(baseline_picks)


def run_sweep_for_scenario(blood_points, facility_services, scenario_key: str, locations) -> dict:
    """Run the baseline plus every window and boost variant for one
    scenario, returning a dict of % changed from baseline for each.
    """
    with temporary_high_window(BASELINE_HIGH_WINDOW), temporary_boost_factor(BASELINE_BOOST_FACTOR):
        baseline_picks = top_pick_for_each_location(blood_points, facility_services, scenario_key, locations)

    window_results = {}
    for window in WINDOW_VALUES_TO_TEST:
        with temporary_high_window(window), temporary_boost_factor(BASELINE_BOOST_FACTOR):
            picks = top_pick_for_each_location(blood_points, facility_services, scenario_key, locations)
        window_results[window] = pct_changed_from_baseline(baseline_picks, picks)

    boost_results = {}
    for boost in BOOST_VALUES_TO_TEST:
        with temporary_high_window(BASELINE_HIGH_WINDOW), temporary_boost_factor(boost):
            picks = top_pick_for_each_location(blood_points, facility_services, scenario_key, locations)
        boost_results[boost] = pct_changed_from_baseline(baseline_picks, picks)

    return {"window": window_results, "boost": boost_results}


def main() -> int:
    """Entry point. Runs the sweep for every scenario in TEST_SCENARIOS and
    prints a report. Returns 0 on success.
    """
    blood_points, facility_services = demo.load_datasets()
    rng = np.random.default_rng(RANDOM_SEED)

    for scenario_key in TEST_SCENARIOS:
        locations = sample_locations(blood_points, scenario_key, N_LOCATIONS, rng)
        results = run_sweep_for_scenario(blood_points, facility_services, scenario_key, locations)

        print(f"\n=== {scenario_key} (n={N_LOCATIONS}, baseline: {BASELINE_HIGH_WINDOW} min window, x{BASELINE_BOOST_FACTOR} boost) ===")
        print("Time window sweep (high-urgency window only; this scenario's own window is unaffected if it's 'lower'):")
        for window, pct in results["window"].items():
            marker = " (baseline)" if window == BASELINE_HIGH_WINDOW else ""
            print(f"  {window} min: {pct:.1f}% of top picks changed{marker}")
        print("Severity boost sweep:")
        for boost, pct in results["boost"].items():
            marker = " (baseline)" if boost == BASELINE_BOOST_FACTOR else ""
            print(f"  x{boost}: {pct:.1f}% of top picks changed{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
