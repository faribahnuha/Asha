"""
==============================================================================
MODULE: candidate_pool.py                                   (stage 1 of 3)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: this is the first stage of the modeling
  pipeline. It has no upstream file inside modeling/ — it reads directly from
  the cleaned tables described in model-design.md (draft)'s "Inputs" section
  (dataset/blood_points_v2.csv and dataset/facility_services_v2.csv, after
  whatever pre-processing/ eventually does to them). Its output is consumed
  by modeling/scoring/scoring.py.
- WHAT THIS FILE DOES: for one patient request (location + scenario), finds
  every blood-carrying facility reachable within that scenario's clinical
  time window, and splits them into a CONFIRMED tier (real service data
  exists for that facility) and an UNCONFIRMED tier (no service data exists
  at all). This is the two-tier design from model-design.md (draft)'s
  "Unknown vs. confirmed-absent" section — see that file before changing
  anything here.
- KNOWN LIMITATIONS / ASSUMPTIONS (this project's chosen third header line —
  see modeling/CLAUDE.md for why): drive time here is straight-line
  (haversine) distance converted to minutes via a detour factor and an
  average speed. This is a DELIBERATE, PERMANENT design choice, not a
  placeholder waiting on live routing or OSM data — this project does not
  use OSM/OSMnx or any routing engine, by decision, and that is not
  expected to change. detour_factor and avg_speed_kmph are still required
  parameters rather than hard-coded defaults, so that the specific values in
  use are always visible at the call site and never silently guessed by
  this file.
==============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# Declared once at import time; never reassigned afterward (see
# modeling/CLAUDE.md, "no dynamic allocation after initialization").
# -----------------------------------------------------------------------------

# Earth radius in kilometres, used by the haversine great-circle formula.
EARTH_RADIUS_KM: float = 6371.0

# India's approximate bounding box. Verified against dataset/blood_points_v2.csv
# while building model-design.md (draft): exactly one candidate row fell
# outside these bounds (a bad coordinate, not a data-entry policy), so this
# box is used as a correctness check, not a geography lesson.
INDIA_LAT_MIN: float = 6.0
INDIA_LAT_MAX: float = 38.0
INDIA_LON_MIN: float = 68.0
INDIA_LON_MAX: float = 98.0

# Clinical time window, in minutes, per urgency tier.
# Source: model-design.md (draft), "Time windows" section.
TIME_WINDOW_MINUTES_BY_URGENCY: dict[str, int] = {"high": 60, "lower": 120}

# Which urgency tier each responder-facing scenario maps to.
# Source: model-design.md (draft), "Time windows" section.
URGENCY_BY_SCENARIO: dict[str, str] = {
    "pregnancy_bleeding": "high",
    "injury_bleeding": "high",
    "not_sure": "high",
    "severe_anemia": "lower",
}

# The services counted toward "needed_services" for each scenario.
# Source: model-design.md (draft), "Criteria" -> "Services counted for each
# scenario". Tuples, not lists: this is fixed data, never mutated.
SCENARIO_SERVICE_LISTS: dict[str, tuple[str, ...]] = {
    "pregnancy_bleeding": ("c_section", "obstetrics", "icu"),
    "injury_bleeding": ("emergency_surgery", "trauma_care", "icu"),
    "severe_anemia": ("general_medicine", "icu"),
    "not_sure": ("ed_24x7", "surgery", "icu"),
}

# Safety rail, not a tuning knob — and NOT a cap on how many facilities the
# model is allowed to consider. model-design.md (draft) has exactly one hard
# gate (the time window); a dense city legitimately producing hundreds of
# in-window candidates must still have every one of them scored. This bound
# exists only to catch a real bug (e.g. a join fan-out duplicating rows), so
# it is set just above the full national pool size (dataset/
# blood_points_v2.csv, in_candidate_pool=True: 3,269 rows) — comfortably
# above any legitimate single-patient pool, which can be at most that many,
# but still a tripwire if something generates more rows than exist.
MAX_CANDIDATES_PER_REQUEST: int = 3500

# Column name used throughout the pipeline to carry the confirmed/
# unconfirmed flag described in model-design.md (draft).
CONFIRMED_FLAG_COLUMN: str = "needed_services_confirmed"
# Matches scoring.py's CONFIRMED_TIER_CRITERIA, which expects a column
# literally named "needed_services" — keep these two files in sync if either changes.
COUNT_COLUMN: str = "needed_services"


def is_within_india_bbox(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Return a boolean Series: True where (lat, lon) falls inside the
    verified India bounding box, False otherwise.

    PARAMETERS:
        lat: latitude values to check. Must be numeric, same length as lon.
        lon: longitude values to check. Must be numeric, same length as lat.

    KNOWN LIMITATION: this is a bounding-box sanity check, not a borders
    lookup — it will not catch a coordinate that is wrong but still happens
    to fall inside the box.
    """
    assert len(lat) == len(lon), "lat/lon length mismatch: not one row per candidate"
    assert len(lat) <= MAX_CANDIDATES_PER_REQUEST, "candidate table exceeds the fixed upper bound"
    in_lat_range = (lat >= INDIA_LAT_MIN) & (lat <= INDIA_LAT_MAX)
    in_lon_range = (lon >= INDIA_LON_MIN) & (lon <= INDIA_LON_MAX)
    return in_lat_range & in_lon_range


def drop_bad_coordinates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Explicit recovery action for the known bad-coordinate case: remove any
    candidate whose (lat, lon) falls outside India, instead of letting it
    silently corrupt a downstream drive-time calculation.

    PARAMETERS:
        candidates: must contain numeric "lat" and "lon" columns.

    RETURNS: a copy of `candidates` with out-of-bounds rows removed.
    """
    assert "lat" in candidates.columns and "lon" in candidates.columns, "missing lat/lon columns"
    assert len(candidates) <= MAX_CANDIDATES_PER_REQUEST, "candidate table exceeds the fixed upper bound"
    valid_mask = is_within_india_bbox(candidates["lat"], candidates["lon"])
    cleaned = candidates.loc[valid_mask].copy()
    assert len(cleaned) <= len(candidates), "filtering must not add rows"
    return cleaned


def compute_drive_minutes(
    patient_lat: float,
    patient_lon: float,
    facility_lat: pd.Series,
    facility_lon: pd.Series,
    detour_factor: float,
    avg_speed_kmph: float,
) -> pd.Series:
    """Estimate drive time, in minutes, from the patient to each facility
    using straight-line (haversine) distance, a measured detour factor, and
    a measured average speed.

    PARAMETERS:
        patient_lat, patient_lon: the responder's reported location.
        facility_lat, facility_lon: one value per candidate facility.
        detour_factor: road distance / straight-line distance. This project
            uses straight-line distance permanently (no OSM/live routing),
            so this is a configured constant, not a value pending later
            calibration. It is still a required parameter, not a hard-coded
            default, so the value in effect is always visible at the call
            site.
        avg_speed_kmph: assumed average travel speed, same reasoning as
            detour_factor above.

    KNOWN LIMITATION: a single detour_factor/avg_speed pair is applied to
    every facility in the call; model-design.md (draft) calls for separate
    urban/rural factors, which the caller is responsible for choosing
    before calling this function per area.
    """
    assert -90.0 <= patient_lat <= 90.0 and -180.0 <= patient_lon <= 180.0, "invalid patient coordinates"
    assert detour_factor >= 1.0, "a detour factor below 1.0 is not physically meaningful"
    assert avg_speed_kmph > 0.0, "average speed must be positive"

    lat1_rad = np.radians(patient_lat)
    lon1_rad = np.radians(patient_lon)
    lat2_rad = np.radians(facility_lat)
    lon2_rad = np.radians(facility_lon)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    haversine_term = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    straight_line_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine_term))

    drive_minutes = (straight_line_km * detour_factor / avg_speed_kmph) * 60.0
    assert bool((drive_minutes >= 0.0).all()), "drive time must never be negative"
    return drive_minutes


def build_needed_service_counts(facility_services: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """For every facility that has AT LEAST ONE row in `facility_services`
    (i.e. is VF-linked with real data), count how many of this scenario's
    needed services it has. Facilities with zero rows here are not
    represented in the output at all — the caller's join step is what turns
    "absent from this table" into "unconfirmed", not this function.

    PARAMETERS:
        facility_services: one row per (unique_id, service) pair, matching
            dataset/facility_services_v2.csv's schema.
        scenario: must be a key of SCENARIO_SERVICE_LISTS.
    """
    assert scenario in SCENARIO_SERVICE_LISTS, f"unknown scenario: {scenario}"
    assert "unique_id" in facility_services.columns and "service" in facility_services.columns, (
        "facility_services is missing required columns"
    )

    needed_services = SCENARIO_SERVICE_LISTS[scenario]
    matching_rows = facility_services[facility_services["service"].isin(needed_services)]
    counts_by_id = matching_rows.groupby("unique_id").size()

    all_known_ids = facility_services["unique_id"].unique()
    result = pd.DataFrame({"unique_id": all_known_ids})
    result[COUNT_COLUMN] = result["unique_id"].map(counts_by_id).fillna(0).astype(int)
    result[CONFIRMED_FLAG_COLUMN] = True

    assert result[COUNT_COLUMN].between(0, len(needed_services)).all(), "count out of expected range"
    assert bool(result[CONFIRMED_FLAG_COLUMN].all()), "every row here must be confirmed by construction"
    return result


def attach_needed_services(
    blood_candidates: pd.DataFrame, facility_services: pd.DataFrame, scenario: str
) -> pd.DataFrame:
    """Left-join needed-service counts onto the candidate table. Facilities
    with no matching row in `facility_services` get needed_services_count=0
    AND needed_services_confirmed=False — the unconfirmed state — rather
    than being dropped or treated as a confirmed zero.

    PARAMETERS:
        blood_candidates: one row per facility with blood, must have a
            "unique_id" column.
        facility_services: passed through to build_needed_service_counts().
        scenario: passed through to build_needed_service_counts().
    """
    assert "unique_id" in blood_candidates.columns, "blood_candidates is missing unique_id"
    assert len(blood_candidates) <= MAX_CANDIDATES_PER_REQUEST, "candidate table exceeds the fixed upper bound"

    service_counts = build_needed_service_counts(facility_services, scenario)
    merged = blood_candidates.merge(service_counts, on="unique_id", how="left")

    merged[COUNT_COLUMN] = merged[COUNT_COLUMN].fillna(0).astype(int)
    # .eq(True) turns NaN (no match in the join) into False directly, with no
    # fillna/astype downcast step that pandas would otherwise warn about.
    merged[CONFIRMED_FLAG_COLUMN] = merged[CONFIRMED_FLAG_COLUMN].eq(True)

    assert len(merged) == len(blood_candidates), "join must not change the row count"
    assert merged[CONFIRMED_FLAG_COLUMN].dtype == bool, "confirmed flag must be boolean"
    return merged


def filter_within_time_window(candidates: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Keep only candidates reachable within this scenario's clinical time
    window. This is the one hard gate in the model (model-design.md
    (draft), "Time windows") — nothing else in the pipeline removes a
    candidate from the pool.

    PARAMETERS:
        candidates: must have a numeric "drive_min" column.
        scenario: must be a key of URGENCY_BY_SCENARIO.
    """
    assert scenario in URGENCY_BY_SCENARIO, f"unknown scenario: {scenario}"
    assert "drive_min" in candidates.columns, "candidates is missing drive_min"

    urgency = URGENCY_BY_SCENARIO[scenario]
    window_minutes = TIME_WINDOW_MINUTES_BY_URGENCY[urgency]
    within_window = candidates[candidates["drive_min"] <= window_minutes].copy()

    assert len(within_window) <= len(candidates), "filtering must not add rows"
    return within_window


def split_confirmed_unconfirmed(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a reachable pool into (confirmed_tier, unconfirmed_tier) using
    CONFIRMED_FLAG_COLUMN. See model-design.md (draft), "Unknown vs.
    confirmed-absent" for why these are never merged back into one score.

    PARAMETERS:
        pool: must contain CONFIRMED_FLAG_COLUMN as a boolean column.
    """
    assert CONFIRMED_FLAG_COLUMN in pool.columns, "pool is missing the confirmed flag column"
    assert pool[CONFIRMED_FLAG_COLUMN].dtype == bool, "confirmed flag must be boolean"

    confirmed_tier = pool[pool[CONFIRMED_FLAG_COLUMN]].copy()
    unconfirmed_tier = pool[~pool[CONFIRMED_FLAG_COLUMN]].copy()

    assert len(confirmed_tier) + len(unconfirmed_tier) == len(pool), "split must account for every row"
    return confirmed_tier, unconfirmed_tier


def self_check() -> bool:
    """End-of-file checker. Builds a small, fixed, in-memory dataset (no
    file I/O, so this never silently passes because a path was wrong) and
    asserts this module's functions behave as documented. Returns True if
    every check passes, False the moment one does not — callers (including
    modeling/self_checks/run_all_checks.py) must check this return value.
    """
    blood_candidates = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C", "out_of_bounds"],
            "lat": [19.0, 28.6, 13.0, 45.0],  # row 4 is deliberately outside India
            "lon": [73.0, 77.2, 80.2, 45.0],
        }
    )
    facility_services = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "service": ["icu", "obstetrics", "emergency_surgery"],
        }
    )

    bbox_checked = drop_bad_coordinates(blood_candidates)
    check_1 = len(bbox_checked) == 3 and "out_of_bounds" not in set(bbox_checked["unique_id"])

    with_services = attach_needed_services(bbox_checked, facility_services, "pregnancy_bleeding")
    row_a = with_services[with_services["unique_id"] == "A"].iloc[0]
    row_c = with_services[with_services["unique_id"] == "C"].iloc[0]
    check_2 = bool(row_a[CONFIRMED_FLAG_COLUMN]) and row_a[COUNT_COLUMN] == 2  # A has "icu" and "obstetrics"
    check_3 = (not bool(row_c[CONFIRMED_FLAG_COLUMN])) and row_c[COUNT_COLUMN] == 0  # C: no data at all

    drive_minutes = compute_drive_minutes(19.0, 73.0, pd.Series([19.0, 28.6]), pd.Series([73.0, 77.2]), 1.3, 30.0)
    check_4 = drive_minutes.iloc[0] < 1.0 and drive_minutes.iloc[1] > 0.0

    with_services["drive_min"] = [0.0, 999.0, 999.0]  # only A is reachable; B and C are not
    windowed = filter_within_time_window(with_services, "pregnancy_bleeding")
    check_5 = len(windowed) == 1 and windowed.iloc[0]["unique_id"] == "A"

    confirmed_tier, unconfirmed_tier = split_confirmed_unconfirmed(with_services)
    check_6 = len(confirmed_tier) == 2 and len(unconfirmed_tier) == 1  # A and B are confirmed; C is not

    all_checks_passed = all([check_1, check_2, check_3, check_4, check_5, check_6])
    if not all_checks_passed:
        print("candidate_pool.self_check(): FAILED", [check_1, check_2, check_3, check_4, check_5, check_6])
    return all_checks_passed


if __name__ == "__main__":
    import sys

    passed = self_check()
    print("candidate_pool.self_check():", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)
