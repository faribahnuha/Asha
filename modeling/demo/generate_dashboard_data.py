"""
==============================================================================
MODULE: generate_dashboard_data.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: calls the same pipeline try_it_out.py
  calls (candidate_pool.py -> scoring.py -> recommend.py) for a small, fixed
  set of sample cases, and writes a human-readable JSON summary of each
  result. It does not change or re-implement any scoring logic — it only
  adds a readable explanation on top of what recommend() already returns.
- WHAT THIS FILE DOES: runs the real model against the real dataset CSVs for
  a curated set of sample (scenario, location) pairs chosen to show the
  model's different behaviors (confirmed-tier ranking in a dense city,
  unconfirmed fallback, a region with no VF-linked data, a location with no
  reachable facility at all) — then writes dashboard_data.json for the demo
  dashboard artifact to read.
- KNOWN LIMITATIONS / ASSUMPTIONS: this is a presentation layer, not a test
  suite — model-design.md (draft)'s "Testing without ground truth" section
  describes the actual test cases this project needs; this file reuses that
  idea for display purposes only. Uses the same ASSUMED (not measured)
  detour_factor/avg_speed_kmph as try_it_out.py.
==============================================================================
"""

from __future__ import annotations

import json
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

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

OUTPUT_JSON_PATH: Path = Path(__file__).resolve().parent / "dashboard_data.json"

# Fixed, named sample cases. Each tuple is (label, scenario_key, lat, lon).
# Chosen to show different real behaviors of the pipeline, not chosen at
# random — see each case's comment for which behavior it demonstrates.
SAMPLE_CASES: tuple[tuple[str, str, float, float], ...] = (
    ("Mumbai (dense urban)", "injury_bleeding", 19.0760, 72.8777),  # many confirmed options
    ("Rural Chhattisgarh", "pregnancy_bleeding", 22.0084, 81.2374),  # few options, sparse area
    ("Delhi", "severe_anemia", 28.7041, 77.1025),  # dense, lower-urgency boost on needed_services
    ("Imphal, Manipur", "not_sure", 24.8170, 93.9368),  # a state with almost no VF-linked data
    ("Bengaluru", "pregnancy_bleeding", 12.9716, 77.5946),  # dense, obstetric-specific scenario
    ("Leh, Ladakh", "injury_bleeding", 34.1526, 77.5771),  # remote: likely nothing reachable in time
)

# Plain-language descriptions for the three blood_level tiers.
# Source: model-design.md (draft), "Criteria" table.
BLOOD_LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "Has blood on site (components and 24x7 status not confirmed)",
    2: "Has blood plus components (platelets, plasma)",
    3: "Has blood, components, and is open 24x7",
}

# Human-readable labels for the services used in SCENARIO_SERVICE_LISTS.
SERVICE_LABELS: dict[str, str] = {
    "c_section": "C-section",
    "obstetrics": "Obstetric care",
    "icu": "ICU",
    "emergency_surgery": "Emergency surgery",
    "trauma_care": "Trauma care",
    "general_medicine": "General medicine",
    "ed_24x7": "24/7 emergency department",
    "surgery": "Surgery",
}


def describe_services(unique_id: str, scenario_key: str, is_confirmed: bool, facility_services: pd.DataFrame) -> dict:
    """Build a human-readable breakdown of which of this scenario's needed
    services a facility has and lacks.

    PARAMETERS:
        unique_id: the facility's id. May not appear in facility_services.
        scenario_key: must be a key of cp.SCENARIO_SERVICE_LISTS.
        is_confirmed: whether this facility has any service data at all.
        facility_services: the full (unique_id, service) table.
    """
    assert scenario_key in cp.SCENARIO_SERVICE_LISTS, f"unknown scenario: {scenario_key}"
    needed_services = cp.SCENARIO_SERVICE_LISTS[scenario_key]

    if not is_confirmed:
        return {"has": [], "missing": [], "note": "No service data available for this facility"}

    facility_rows = facility_services[facility_services["unique_id"] == unique_id]
    present_services = set(facility_rows["service"])

    has = [SERVICE_LABELS.get(service, service) for service in needed_services if service in present_services]
    missing = [SERVICE_LABELS.get(service, service) for service in needed_services if service not in present_services]
    assert len(has) + len(missing) == len(needed_services), "every needed service must be has or missing, not both/neither"
    return {"has": has, "missing": missing, "note": ""}


def summarize_one_result(
    result: pd.DataFrame, weights: dict, facility_services: pd.DataFrame, scenario_key: str
) -> list[dict]:
    """Turn one recommend() result into a list of plain JSON-serializable
    facility summaries, ready to write out.

    PARAMETERS:
        result: output of recommend(); must have the columns recommend()
            always produces (unique_id, lat, lon, drive_min, score, note).
        weights: the second item recommend() returns.
        facility_services: passed through to describe_services().
        scenario_key: passed through to describe_services().
    """
    required_columns = ("unique_id", "lat", "lon", "drive_min", "score", "note")
    missing_columns = [column for column in required_columns if column not in result.columns]
    assert not missing_columns, f"result is missing columns: {missing_columns}"

    facility_summaries = []
    for _, row in result.iterrows():
        is_confirmed = bool(row.get("needed_services_confirmed", False))
        service_breakdown = describe_services(row["unique_id"], scenario_key, is_confirmed, facility_services)
        blood_level = int(row["blood_level"]) if pd.notna(row.get("blood_level")) else None

        facility_summaries.append(
            {
                "name": str(row.get("name", row["unique_id"])),
                "city": str(row.get("city", "")) if pd.notna(row.get("city")) else "",
                "state": str(row.get("state_std", "")) if pd.notna(row.get("state_std")) else "",
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "drive_min": round(float(row["drive_min"]), 1),
                "score": round(float(row["score"]), 3),
                "confirmed": is_confirmed,
                "blood_level": blood_level,
                "blood_level_description": BLOOD_LEVEL_DESCRIPTIONS.get(blood_level, "Unknown"),
                "services_has": service_breakdown["has"],
                "services_missing": service_breakdown["missing"],
                "services_note": service_breakdown["note"],
                "note": str(row.get("note", "")),
            }
        )
    assert len(facility_summaries) == len(result), "summary count must match result row count"
    return facility_summaries


def build_dashboard_data(blood_points: pd.DataFrame, facility_services: pd.DataFrame) -> list[dict]:
    """Run every SAMPLE_CASES entry through the real pipeline and return a
    list of JSON-serializable case summaries, in the same order as
    SAMPLE_CASES.
    """
    assert len(SAMPLE_CASES) >= 1, "SAMPLE_CASES must not be empty"

    all_cases = []
    for label, scenario_key, patient_lat, patient_lon in SAMPLE_CASES:
        nearby = demo.narrow_to_nearby(blood_points, scenario_key, patient_lat, patient_lon)
        case_entry = {
            "label": label,
            "scenario_key": scenario_key,
            "scenario_label": dict(demo.SCENARIO_MENU)[scenario_key],
            "patient_lat": patient_lat,
            "patient_lon": patient_lon,
            "candidates_considered": int(len(nearby)),
        }

        if nearby.empty:
            case_entry["facilities"] = []
            case_entry["outcome_note"] = "No candidates at all within the search area."
            all_cases.append(case_entry)
            continue

        result, weights = R.recommend(
            nearby,
            facility_services,
            patient_lat,
            patient_lon,
            scenario_key,
            demo.ASSUMED_DETOUR_FACTOR,
            demo.ASSUMED_AVG_SPEED_KMPH,
        )
        assert len(result) >= 1, "recommend() must return at least one row when candidates exist"

        case_entry["facilities"] = summarize_one_result(result, weights, facility_services, scenario_key)
        case_entry["weights"] = {key: round(float(value), 3) for key, value in weights.items()}
        case_entry["outcome_note"] = str(result.iloc[0].get("note", "")) if "note" in result.columns else ""
        all_cases.append(case_entry)

    assert len(all_cases) == len(SAMPLE_CASES), "must produce exactly one entry per sample case"
    return all_cases


def main() -> int:
    """Entry point. Writes OUTPUT_JSON_PATH and returns 0 on success."""
    blood_points, facility_services = demo.load_datasets()
    dashboard_data = build_dashboard_data(blood_points, facility_services)

    assert len(dashboard_data) == len(SAMPLE_CASES), "dashboard_data length mismatch"
    OUTPUT_JSON_PATH.write_text(json.dumps(dashboard_data, indent=2))
    print(f"Wrote {len(dashboard_data)} sample cases to {OUTPUT_JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
