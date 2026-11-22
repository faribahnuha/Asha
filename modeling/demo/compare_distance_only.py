"""
==============================================================================
MODULE: compare_distance_only.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: reads the same real dataset CSVs as
  try_it_out.py and generate_dashboard_data.py, and reuses
  generate_dashboard_data.SAMPLE_CASES so this comparison runs on the exact
  same locations/scenarios already shown in the dashboard. It does not
  change recommend.py or scoring.py — the "distance-only" ranking here is a
  separate, throwaway comparison function, not an alternate mode of the
  real model.
- WHAT THIS FILE DOES: answers "how different would the recommendation be
  if we ignored blood level and needed services, and ranked purely by
  distance?" For each sample case, it runs the real model AND a
  distance-only ranking over the same reachable candidates, then reports
  where the top 3 differ.
- KNOWN LIMITATIONS / ASSUMPTIONS: this is a one-off analysis script, not a
  pipeline stage and not part of model-design.md (draft)'s design — it
  exists to show, concretely, why the model doesn't rank on distance alone
  (see model-design.md (draft)'s "Why this model" table: "Cities and rural
  areas behave differently... weights are recomputed for each patient's own
  candidates" — distance-only ranking cannot do that).
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELING_ROOT = _REPO_ROOT / "modeling"
for _sibling_dir in ("candidate_pool", "scoring", "recommend", "demo"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import candidate_pool as cp  # noqa: E402
import recommend as R  # noqa: E402
import try_it_out as demo  # noqa: E402
import generate_dashboard_data as dashboard_data  # noqa: E402

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

TOP_N: int = R.TOP_N_RESULTS


def rank_by_distance_only(reachable_pool: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by drive_min alone, ignoring blood_level and
    needed_services entirely — the comparison this file exists to run, not
    a mode the real model supports.

    PARAMETERS:
        reachable_pool: must contain a "drive_min" column and at least one row.
    """
    assert "drive_min" in reachable_pool.columns, "reachable_pool is missing drive_min"
    assert len(reachable_pool) >= 1, "cannot rank an empty pool"
    return reachable_pool.nsmallest(TOP_N, "drive_min")


def run_full_model(blood_points: pd.DataFrame, facility_services: pd.DataFrame, scenario_key: str, patient_lat: float, patient_lon: float) -> pd.DataFrame:
    """Run the real model and return its top-N result.

    PARAMETERS: same as demo.narrow_to_nearby()/R.recommend().
    """
    nearby = demo.narrow_to_nearby(blood_points, scenario_key, patient_lat, patient_lon)
    assert len(nearby) >= 1, "no candidates found near this point at all"
    result, _ = R.recommend(
        nearby, facility_services, patient_lat, patient_lon, scenario_key, demo.ASSUMED_DETOUR_FACTOR, demo.ASSUMED_AVG_SPEED_KMPH
    )
    return result


def run_distance_only(blood_points: pd.DataFrame, facility_services: pd.DataFrame, scenario_key: str, patient_lat: float, patient_lon: float) -> pd.DataFrame:
    """Run the distance-only comparison over the SAME candidate pool and
    drive-time calculation the full model used, so the only variable that
    changes is whether blood_level/needed_services factor into the ranking.
    """
    nearby = demo.narrow_to_nearby(blood_points, scenario_key, patient_lat, patient_lon)
    clean_candidates = cp.drop_bad_coordinates(nearby)
    with_services = cp.attach_needed_services(clean_candidates, facility_services, scenario_key)
    with_services["drive_min"] = cp.compute_drive_minutes(
        patient_lat, patient_lon, with_services["lat"], with_services["lon"], demo.ASSUMED_DETOUR_FACTOR, demo.ASSUMED_AVG_SPEED_KMPH
    )
    reachable_pool = cp.filter_within_time_window(with_services, scenario_key)
    if reachable_pool.empty:
        return with_services.nsmallest(TOP_N, "drive_min")
    return rank_by_distance_only(reachable_pool)


def compare_one_case(blood_points: pd.DataFrame, facility_services: pd.DataFrame, label: str, scenario_key: str, patient_lat: float, patient_lon: float) -> dict:
    """Run both rankings for one sample case and summarize the difference.

    RETURNS: a dict with both top-3 lists (names only) and whether the
    #1 pick changed.
    """
    full_model_result = run_full_model(blood_points, facility_services, scenario_key, patient_lat, patient_lon)
    distance_only_result = run_distance_only(blood_points, facility_services, scenario_key, patient_lat, patient_lon)

    full_model_names = list(full_model_result["name"]) if "name" in full_model_result.columns else list(full_model_result["unique_id"])
    distance_only_names = list(distance_only_result["name"]) if "name" in distance_only_result.columns else list(distance_only_result["unique_id"])

    assert len(full_model_names) >= 1 and len(distance_only_names) >= 1, "both rankings must return at least one name"
    top_pick_changed = full_model_names[0] != distance_only_names[0]
    overlap_count = len(set(full_model_names) & set(distance_only_names))

    return {
        "label": label,
        "scenario_key": scenario_key,
        "full_model_top3": full_model_names,
        "distance_only_top3": distance_only_names,
        "top_pick_changed": top_pick_changed,
        "overlap_count": overlap_count,
    }


def print_comparison(comparison: dict) -> None:
    """Print one case's comparison in plain language.

    PARAMETERS:
        comparison: output of compare_one_case().
    """
    assert "label" in comparison and "full_model_top3" in comparison, "comparison dict is missing expected keys"
    print(f"\n{comparison['label']} ({comparison['scenario_key']})")
    print(f"  Full model:     {comparison['full_model_top3']}")
    print(f"  Distance only:  {comparison['distance_only_top3']}")
    changed_label = "YES — different #1 pick" if comparison["top_pick_changed"] else "no — same #1 pick"
    print(f"  Top pick changed: {changed_label}")
    print(f"  Overlap in top 3: {comparison['overlap_count']} / {TOP_N}")


def main() -> int:
    """Entry point. Runs the comparison for every sample case and prints a summary."""
    blood_points, facility_services = demo.load_datasets()
    assert len(dashboard_data.SAMPLE_CASES) >= 1, "no sample cases to compare"

    comparisons = []
    for label, scenario_key, patient_lat, patient_lon in dashboard_data.SAMPLE_CASES:
        comparison = compare_one_case(blood_points, facility_services, label, scenario_key, patient_lat, patient_lon)
        comparisons.append(comparison)
        print_comparison(comparison)

    changed_count = sum(1 for c in comparisons if c["top_pick_changed"])
    print(f"\nTop pick changed in {changed_count} / {len(comparisons)} sample cases.")
    assert len(comparisons) == len(dashboard_data.SAMPLE_CASES), "must produce one comparison per sample case"
    return 0


if __name__ == "__main__":
    sys.exit(main())
