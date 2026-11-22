"""
==============================================================================
MODULE: try_it_out.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: this is NOT a pipeline stage. It is a
  user-facing demo that loads the real dataset/*.csv files and calls
  modeling/recommend/recommend.py exactly the way a future dashboard or API
  layer would. Nothing in modeling/ depends on this file; this file depends
  on all three pipeline stages.
- WHAT THIS FILE DOES: lets a person pick one of the four responder-facing
  scenarios and a location (typed coordinates, or a random real place in
  India), runs recommend(), and prints the top 3 facilities in plain
  language — including whether each one's capability is confirmed or not.
- KNOWN LIMITATIONS / ASSUMPTIONS: detour_factor and avg_speed_kmph below
  are ASSUMED placeholder values, not measured ones (see candidate_pool.py's
  header — this project's distance model is permanently straight-line, not
  OSM/live routing, but the specific factor/speed to use has still never
  been measured against real trips). Every result this script prints says
  so, so nobody mistakes a demo number for a validated one. "Random location
  in India" picks a real candidate facility's own coordinates (plus a small
  jitter) rather than a uniformly random point in the bounding box, so the
  demo never drops the patient in the ocean or across the border.
==============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELING_ROOT = _REPO_ROOT / "modeling"
for _sibling_dir in ("candidate_pool", "scoring", "recommend"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import candidate_pool as cp  # noqa: E402  (see sys.path setup above)
import recommend as R  # noqa: E402  (see sys.path setup above)

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

# Paths to the real dataset files this demo reads. Fixed, not configurable
# at the command line, so a typo in a path can never silently point this
# demo at the wrong data.
BLOOD_POINTS_CSV: Path = _REPO_ROOT / "dataset" / "blood_points_v2.csv"
FACILITY_SERVICES_CSV: Path = _REPO_ROOT / "dataset" / "facility_services_v2.csv"

# ASSUMED, not measured — see this file's header. Printed with every result
# so the assumption is visible, never hidden inside a "default".
ASSUMED_DETOUR_FACTOR: float = 1.3
ASSUMED_AVG_SPEED_KMPH: float = 35.0

# Small random offset applied to a real facility's coordinates when picking
# a "random location in India", so the patient isn't placed exactly on top
# of a facility every time. About 0.1 degrees is roughly 10-15 km.
RANDOM_LOCATION_JITTER_DEGREES: float = 0.1

# The four responder-facing scenarios, in the exact wording discussed for
# this project, mapped to the internal scenario keys candidate_pool.py uses.
SCENARIO_MENU: tuple[tuple[str, str], ...] = (
    ("pregnancy_bleeding", "Pregnant or just delivered, heavy bleeding"),
    ("injury_bleeding", "Accident or injury with major bleeding"),
    ("severe_anemia", "Severe weakness or anemia, no injury (including in pregnancy)"),
    ("not_sure", "Other blood-related / not sure"),
)

# Plain-language descriptions for the three blood_level tiers.
# Source: model-design.md (draft), "Criteria" table. Shared by
# generate_dashboard_data.py and modeling/api/app.py so both present
# capability the same way instead of drifting apart.
BLOOD_LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "Has blood on site (components and 24x7 status not confirmed)",
    2: "Has blood plus components (platelets, plasma)",
    3: "Has blood, components, and is open 24x7",
}

# Human-readable labels for the services used in cp.SCENARIO_SERVICE_LISTS.
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

# Kilometres per degree of latitude (and, at the equator, of longitude too).
# Used only to size the pre-filter box below — not a substitute for the
# haversine math candidate_pool.py already does on the actual candidates.
KM_PER_DEGREE_LATITUDE: float = 111.0

# The pre-filter box must be a SUPERSET of everything the real time-window
# gate could ever accept, never a smaller, arbitrary cutoff — otherwise this
# demo would silently exclude a facility the real model would have kept.
# This margin multiplies the model's own max-reachable-distance so rounding
# and the flat-box approximation (vs. true haversine) can't shrink it below
# that true reachable distance.
PREFILTER_BOX_SAFETY_MARGIN: float = 1.5

# recommend.py pads results out to TOP_N_RESULTS using the closest facility
# OUTSIDE the time window when fewer than that many are reachable (see
# recommend.py's _pad_to_top_n()). That padding candidate has to actually be
# present in the table this demo passes to recommend() — a box sized only
# to the time window has nothing beyond it to pad with. This is a second,
# independent floor on top of the time-window-derived box above, wide enough
# that a padding candidate can be found even in a sparse region: comfortably
# larger than any single scenario's reachable distance, not a guess at what
# "nearby" means.
MINIMUM_PADDING_SEARCH_KM: float = 500.0


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and lightly validate the two real dataset files this demo needs.

    RETURNS: (blood_points, facility_services), both non-empty.
    """
    assert BLOOD_POINTS_CSV.exists(), f"missing file: {BLOOD_POINTS_CSV}"
    assert FACILITY_SERVICES_CSV.exists(), f"missing file: {FACILITY_SERVICES_CSV}"

    blood_points = pd.read_csv(BLOOD_POINTS_CSV, low_memory=False)
    facility_services = pd.read_csv(FACILITY_SERVICES_CSV, low_memory=False)

    assert len(blood_points) > 0, "blood_points_v2.csv loaded empty"
    assert len(facility_services) > 0, "facility_services_v2.csv loaded empty"
    return blood_points, facility_services


def describe_services(unique_id: str, scenario_key: str, is_confirmed: bool, facility_services: pd.DataFrame) -> dict:
    """Build a human-readable breakdown of which of this scenario's needed
    services a facility has and lacks. Shared by generate_dashboard_data.py
    and modeling/api/app.py — the "why" behind a recommendation should look
    the same whether it's read from the static dashboard or a live API call.

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


def pick_random_location(blood_points: pd.DataFrame, random_seed: int | None) -> tuple[float, float]:
    """Pick a random real facility's coordinates, jittered slightly, as a
    stand-in patient location. Never picks a point outside the candidate
    pool's own known-good coordinates.

    PARAMETERS:
        blood_points: must have "lat", "lon", and "in_candidate_pool" columns.
        random_seed: pass a fixed value to get a reproducible "random" pick.
    """
    assert "lat" in blood_points.columns and "lon" in blood_points.columns, "missing lat/lon columns"
    pool = blood_points[blood_points["in_candidate_pool"] == True].dropna(subset=["lat", "lon"])  # noqa: E712
    assert len(pool) > 0, "no valid candidates to pick a random location from"

    random_generator = np.random.default_rng(random_seed)
    chosen_row = pool.sample(n=1, random_state=random_seed if random_seed is not None else None).iloc[0]
    jitter_lat = random_generator.uniform(-RANDOM_LOCATION_JITTER_DEGREES, RANDOM_LOCATION_JITTER_DEGREES)
    jitter_lon = random_generator.uniform(-RANDOM_LOCATION_JITTER_DEGREES, RANDOM_LOCATION_JITTER_DEGREES)

    patient_lat = float(chosen_row["lat"]) + jitter_lat
    patient_lon = float(chosen_row["lon"]) + jitter_lon
    assert cp.INDIA_LAT_MIN - 1 <= patient_lat <= cp.INDIA_LAT_MAX + 1, "jittered latitude drifted too far"
    assert cp.INDIA_LON_MIN - 1 <= patient_lon <= cp.INDIA_LON_MAX + 1, "jittered longitude drifted too far"
    return patient_lat, patient_lon


def compute_prefilter_box_degrees(scenario_key: str, patient_lat: float) -> tuple[float, float]:
    """Compute a (lat_degrees, lon_degrees) half-width box that is
    GUARANTEED to contain every facility the real time-window gate could
    ever accept for this scenario — derived from the model's own window,
    detour factor, and speed, not a flat guess. This is what keeps this
    demo's behavior the same as model-design.md (draft), not a separate,
    looser rule: the model never drops a facility for being one of "too
    many"; this box exists only so the demo doesn't load the whole country
    before the real time-window check runs.

    PARAMETERS:
        scenario_key: must be a key of cp.URGENCY_BY_SCENARIO.
        patient_lat: used to convert longitude degrees to km correctly —
            a degree of longitude covers fewer km farther from the equator.
    """
    assert scenario_key in cp.URGENCY_BY_SCENARIO, f"unknown scenario: {scenario_key}"
    assert -90.0 < patient_lat < 90.0, "invalid patient latitude"

    urgency = cp.URGENCY_BY_SCENARIO[scenario_key]
    window_minutes = cp.TIME_WINDOW_MINUTES_BY_URGENCY[urgency]
    time_window_reachable_km = (window_minutes * ASSUMED_AVG_SPEED_KMPH / (ASSUMED_DETOUR_FACTOR * 60.0)) * (
        PREFILTER_BOX_SAFETY_MARGIN
    )
    max_reachable_km = max(time_window_reachable_km, MINIMUM_PADDING_SEARCH_KM)

    lat_degrees = max_reachable_km / KM_PER_DEGREE_LATITUDE
    km_per_degree_longitude_here = KM_PER_DEGREE_LATITUDE * max(np.cos(np.radians(patient_lat)), 0.1)
    lon_degrees = max_reachable_km / km_per_degree_longitude_here

    assert lat_degrees > 0.0 and lon_degrees > 0.0, "computed box must have positive size"
    return lat_degrees, lon_degrees


def narrow_to_nearby(blood_points: pd.DataFrame, scenario_key: str, patient_lat: float, patient_lon: float) -> pd.DataFrame:
    """Pre-filter the national candidate table down to a bounding box around
    the patient, sized so it can only ever be a SUPERSET of what the real
    time-window gate would accept — never smaller. This is the "find nearby
    candidates" step model-design.md (draft) delegates to Databricks H3/SQL
    in production; here it's a plain lat/lon box, good enough for a local
    demo, as long as it never excludes anything the real model would keep.

    PARAMETERS:
        blood_points: must have "lat", "lon", "in_candidate_pool" columns.
        scenario_key: passed to compute_prefilter_box_degrees().
        patient_lat, patient_lon: the demo patient's location.
    """
    assert -90.0 <= patient_lat <= 90.0, "invalid patient latitude"
    assert -180.0 <= patient_lon <= 180.0, "invalid patient longitude"

    lat_degrees, lon_degrees = compute_prefilter_box_degrees(scenario_key, patient_lat)
    pool = blood_points[blood_points["in_candidate_pool"] == True].dropna(subset=["lat", "lon"])  # noqa: E712
    nearby = pool[
        (pool["lat"].between(patient_lat - lat_degrees, patient_lat + lat_degrees))
        & (pool["lon"].between(patient_lon - lon_degrees, patient_lon + lon_degrees))
    ].copy()

    # This should only ever fire on a real bug (see candidate_pool.py's
    # comment on MAX_CANDIDATES_PER_REQUEST) — never on legitimate density,
    # since the box above is already sized to the model's own reachability
    # math, not an arbitrary flat radius.
    assert len(nearby) <= cp.MAX_CANDIDATES_PER_REQUEST, (
        f"{len(nearby)} candidates found near this point, exceeding the fixed upper bound "
        f"({cp.MAX_CANDIDATES_PER_REQUEST}) — this indicates a bug, not real density"
    )
    return nearby


def format_result_table(result: pd.DataFrame) -> str:
    """Build a plain-text table of the top results for printing.

    PARAMETERS:
        result: output of recommend(); must have unique_id, drive_min, score,
            needed_services_confirmed, and note columns.
    """
    required_columns = ("drive_min", "score", "needed_services_confirmed", "note")
    missing = [column for column in required_columns if column not in result.columns]
    assert not missing, f"result is missing expected columns: {missing}"
    assert len(result) >= 1, "cannot format an empty result"

    lines = []
    for rank, (_, row) in enumerate(result.iterrows(), start=1):
        name = row.get("name", row.get("unique_id", "unknown facility"))
        place = ", ".join(str(row.get(field)) for field in ("city", "state_std") if pd.notna(row.get(field)))
        confirmed_label = "confirmed" if bool(row.get("needed_services_confirmed", False)) else "UNCONFIRMED"
        line = (
            f"{rank}. {name} ({place})\n"
            f"   drive time: ~{row['drive_min']:.0f} min | score: {row['score']:.3f} | "
            f"capability data: {confirmed_label}"
        )
        if row.get("note"):
            line += f"\n   note: {row['note']}"
        lines.append(line)
        assert rank <= R.TOP_N_RESULTS, "format_result_table() received more rows than TOP_N_RESULTS"
    return "\n".join(lines)


def run_one_query(
    blood_points: pd.DataFrame,
    facility_services: pd.DataFrame,
    scenario_key: str,
    patient_lat: float,
    patient_lon: float,
) -> None:
    """Run one recommend() call and print the result in plain language.

    PARAMETERS:
        scenario_key: must be one of the keys in SCENARIO_MENU.
    """
    valid_keys = [key for key, _ in SCENARIO_MENU]
    assert scenario_key in valid_keys, f"unknown scenario key: {scenario_key}"
    assert -90.0 <= patient_lat <= 90.0 and -180.0 <= patient_lon <= 180.0, "invalid patient coordinates"

    nearby = narrow_to_nearby(blood_points, scenario_key, patient_lat, patient_lon)
    lat_degrees, lon_degrees = compute_prefilter_box_degrees(scenario_key, patient_lat)
    print(f"\nPatient location: ({patient_lat:.4f}, {patient_lon:.4f})")
    print(f"Scenario: {dict(SCENARIO_MENU)[scenario_key]}")
    print(
        f"Candidates within this scenario's reachable area "
        f"(±{lat_degrees:.2f}° lat, ±{lon_degrees:.2f}° lon): {len(nearby)}"
    )
    print(
        f"Distance model: straight-line, detour factor {ASSUMED_DETOUR_FACTOR}x, "
        f"avg speed {ASSUMED_AVG_SPEED_KMPH} km/h (ASSUMED, not measured)"
    )

    if nearby.empty:
        print("No candidates at all within the search radius — nothing to recommend.")
        return

    result, weights = R.recommend(
        nearby, facility_services, patient_lat, patient_lon, scenario_key, ASSUMED_DETOUR_FACTOR, ASSUMED_AVG_SPEED_KMPH
    )
    assert len(result) >= 1, "recommend() must always return at least one row when candidates exist"

    print(f"\nTop {len(result)} result(s):")
    print(format_result_table(result))
    if weights:
        print(f"\nWeights used: {weights}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for this demo."""
    parser = argparse.ArgumentParser(description="Try the facility-recommendation model on real data.")
    parser.add_argument(
        "--scenario",
        choices=[key for key, _ in SCENARIO_MENU],
        help="Scenario key. If omitted, you'll be prompted to choose.",
    )
    parser.add_argument("--lat", type=float, help="Patient latitude. Omit together with --lon for a random spot.")
    parser.add_argument("--lon", type=float, help="Patient longitude. Omit together with --lat for a random spot.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for a reproducible random location.")
    return parser


def prompt_for_scenario() -> str:
    """Ask the user to pick a scenario from SCENARIO_MENU. Re-prompts on an
    invalid choice rather than crashing on bad input.
    """
    assert len(SCENARIO_MENU) >= 1, "SCENARIO_MENU must never be empty"
    print("\nWhich scenario?")
    for index, (_, label) in enumerate(SCENARIO_MENU, start=1):
        print(f"  {index}. {label}")

    max_attempts = 5  # fixed upper bound: never loop forever on bad input
    for _attempt in range(max_attempts):
        raw_choice = input(f"Enter 1-{len(SCENARIO_MENU)}: ").strip()
        if raw_choice.isdigit() and 1 <= int(raw_choice) <= len(SCENARIO_MENU):
            return SCENARIO_MENU[int(raw_choice) - 1][0]
        print("Not a valid choice, try again.")
    print("Too many invalid attempts; defaulting to scenario 1.")
    return SCENARIO_MENU[0][0]


def main() -> int:
    """Entry point. Returns a process exit code (0 success, 1 failure)."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    blood_points, facility_services = load_datasets()

    scenario_key = arguments.scenario if arguments.scenario is not None else prompt_for_scenario()

    if arguments.lat is not None and arguments.lon is not None:
        patient_lat, patient_lon = arguments.lat, arguments.lon
    elif arguments.lat is not None or arguments.lon is not None:
        print("Pass both --lat and --lon, or neither (for a random location).")
        return 1
    else:
        patient_lat, patient_lon = pick_random_location(blood_points, arguments.seed)

    run_one_query(blood_points, facility_services, scenario_key, patient_lat, patient_lon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
