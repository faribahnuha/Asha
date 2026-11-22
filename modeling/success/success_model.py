"""
==============================================================================
MODULE: success_model.py                    (v2 model: the three probability factors)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: consumes P(capable) per capability from
  modeling/capability (dataset/gold/capability_scores.csv), hospital and blood-bank coordinates
  from dataset/gold, and every assumption from dataset/ref/parameters.csv + scenarios.csv. Hands
  arrays of P(capable), P(blood in time) and P(arrives in time) to recommend_v2.py, which ranks.
- WHAT THIS FILE DOES: computes, for every candidate hospital and one patient request,
  P(success) = P(capable for the scenario) x P(right blood products available in time) x
  P(patient arrives within the clinical window). All distances are computed here from coordinates
  at query time (gold stores none).
- KNOWN LIMITATIONS / ASSUMPTIONS: the three factors are multiplied as if independent; requiring
  several capabilities multiplies their probabilities (conservative, since they are correlated);
  travel time is straight-line x detour / speed with a log-normal spread (no routing); blood stock,
  issue times and link trust are stated assumptions from parameters.csv, not measurements.
==============================================================================
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT: Path = Path(__file__).resolve().parents[2]
EARTH_RADIUS_KM: float = 6371.0
MAX_SITES: int = 20000              # bug tripwire: more rows than exist nationally means a join fanned out
MAX_BANKS: int = 5000
SCENARIOS_PATH: Path = ROOT / "dataset/ref/scenarios.csv"
PARAMETERS_PATH: Path = ROOT / "dataset/ref/parameters.csv"
_ERF = np.vectorize(math.erf)


def load_parameters(path: Path = PARAMETERS_PATH) -> dict[str, float]:
    """Every model assumption, by name, from the reference table."""
    assert path.exists(), f"missing {path}"
    params = {r["parameter"]: float(r["value"]) for r in csv.DictReader(open(path, encoding="utf-8"))}
    assert {"window_high_min", "road_detour_factor", "p_bank_has_stock", "abstain_threshold"} <= set(params), "parameters.csv is incomplete"
    return params


def load_scenarios(path: Path = SCENARIOS_PATH) -> dict[str, dict[str, object]]:
    """Scenario requirements: required/supporting capabilities, minimum blood level, window."""
    assert path.exists(), f"missing {path}"
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):   # 4 rows, fixed
        out[r["scenario"]] = dict(label=r["label"], window_parameter=r["window_parameter"],
                                  required=tuple(r["required_capabilities"].split("|")),
                                  supporting=tuple(r["supporting_capabilities"].split("|")),
                                  min_blood_level=int(r["min_blood_level"]), blood_products=r["blood_products"])
    assert len(out) == 4, "expected exactly 4 scenarios"
    return out


def haversine_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance from one point to many, in km."""
    assert -90 <= lat <= 90 and -180 <= lon <= 180, "invalid coordinates"
    assert lats.shape == lons.shape, "lat/lon arrays must match"
    p1, p2 = math.radians(lat), np.radians(lats)
    a = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(np.radians(lons - lon) / 2) ** 2
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    assert (d >= 0).all(), "distances must be non-negative"
    return d


def pairwise_km(lats_a: np.ndarray, lons_a: np.ndarray, lats_b: np.ndarray, lons_b: np.ndarray) -> np.ndarray:
    """Distance matrix (len(a) x len(b)) in km."""
    assert len(lats_a) <= MAX_SITES and len(lats_b) <= MAX_BANKS, "matrix larger than any real request"
    p1, p2 = np.radians(lats_a)[:, None], np.radians(lats_b)[None, :]
    dl = np.radians(lons_b)[None, :] - np.radians(lons_a)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    assert d.shape == (len(lats_a), len(lats_b)), "wrong matrix shape"
    return d


def area_is_rural(patient_lat: float, patient_lon: float, site_lats: np.ndarray, site_lons: np.ndarray,
                  params: dict[str, float]) -> bool:
    """Rural if few hospitals are within 10 km of the PATIENT (density proxy; no census data)."""
    assert len(site_lats) <= MAX_SITES, "site table larger than the national dataset"
    assert "rural_density_threshold" in params, "missing rural_density_threshold"
    n_near = int((haversine_km(patient_lat, patient_lon, site_lats, site_lons) <= 10).sum())
    rural = n_near <= params["rural_density_threshold"]
    assert isinstance(rural, (bool, np.bool_)), "must be a boolean"
    return bool(rural)


def travel_minutes(distance_km: np.ndarray, rural: bool, params: dict[str, float]) -> np.ndarray:
    """Median travel time: straight-line distance x detour factor / speed (urban or rural)."""
    assert (distance_km >= 0).all(), "distances must be non-negative"
    speed = params["speed_rural_kmph"] if rural else params["speed_urban_kmph"]
    assert speed > 0, "speed must be positive"
    minutes = distance_km * params["road_detour_factor"] / speed * 60.0
    return minutes


def p_within(median_minutes: np.ndarray, window_minutes: float, cv: float) -> np.ndarray:
    """P(actual time <= window) when actual time is log-normal around the median with spread cv."""
    assert window_minutes > 0 and cv > 0, "window and spread must be positive"
    assert (median_minutes >= 0).all(), "times must be non-negative"
    sigma = math.sqrt(math.log(1 + cv * cv))
    z = (math.log(window_minutes) - np.log(np.maximum(median_minutes, 0.1))) / sigma
    p = 0.5 * (1 + _ERF(z / math.sqrt(2)))
    return np.clip(p, 0.0, 1.0)


def window_minutes(scenario: dict[str, object], params: dict[str, float], hb: float | None) -> float:
    """Clinical window for the scenario, shortened when the patient's Hb is known to be very low."""
    assert scenario["window_parameter"] in params, "unknown window parameter"
    assert hb is None or 1.0 <= hb <= 20.0, "Hb must be in g/dL"
    w = params[str(scenario["window_parameter"])]
    if hb is not None and hb < params["hb_severe_threshold"]:
        w *= params["hb_severe_window_factor"]
    return float(w)


def p_capable_for_scenario(cap_probs: dict[str, np.ndarray], required: tuple[str, ...],
                           laq_mot: np.ndarray, laq_lr: np.ndarray, params: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """P(all required capabilities), after raising them where LaQshya independently certifies them."""
    assert all(c in cap_probs for c in required), "missing capability probabilities"
    assert laq_mot.shape == laq_lr.shape, "certification arrays must match"
    adjusted = {}
    for c in required:                                 # at most 3 required capabilities
        p = cap_probs[c].copy()
        if c in ("obstetric_care", "surgical_capacity", "anaesthesia"):
            p = np.where(laq_mot, np.maximum(p, params["laqshya_mot_floor"]), p)
        if c == "obstetric_care":
            p = np.where(laq_lr, np.maximum(p, params["laqshya_lr_obstetric_floor"]), p)
        adjusted[c] = p
    total = np.prod(np.vstack([adjusted[c] for c in required]), axis=0)
    assert ((total >= 0) & (total <= 1)).all(), "probability out of range"
    return total, adjusted


def self_check() -> bool:
    """Fixed in-memory checks of distances, travel time, window probability and capability combination."""
    params = {"speed_urban_kmph": 30.0, "speed_rural_kmph": 40.0, "road_detour_factor": 1.0, "rural_density_threshold": 1.0,
              "laqshya_mot_floor": 0.9, "laqshya_lr_obstetric_floor": 0.85, "hb_severe_threshold": 7.0,
              "hb_severe_window_factor": 0.5, "window_high_min": 60.0}
    d = haversine_km(28.6, 77.2, np.array([28.6, 28.7]), np.array([77.2, 77.2]))
    check_1 = d[0] < 1e-6 and 10.5 < d[1] < 11.6                     # 0.1 degree latitude ~ 11.1 km
    check_2 = abs(float(travel_minutes(np.array([30.0]), False, params)[0]) - 60.0) < 1e-9
    p = p_within(np.array([30.0, 60.0, 120.0]), 60.0, 0.3)
    check_3 = p[0] > 0.95 and abs(p[1] - 0.5) < 1e-9 and p[2] < 0.05
    total, _ = p_capable_for_scenario({"obstetric_care": np.array([0.2, 0.9]), "anaesthesia": np.array([0.5, 0.9])},
                                      ("obstetric_care", "anaesthesia"), np.array([True, False]), np.array([False, False]), params)
    check_4 = abs(total[0] - 0.81) < 1e-9 and abs(total[1] - 0.81) < 1e-9   # LaQshya floor lifts site 0 to 0.9 x 0.9
    check_5 = window_minutes({"window_parameter": "window_high_min"}, params, 6.0) == 30.0
    passed = all([check_1, check_2, check_3, check_4, check_5])
    if not passed:
        print("success_model.self_check(): FAILED", [check_1, check_2, check_3, check_4, check_5])
    return passed


if __name__ == "__main__":
    import sys

    ok = self_check()
    print("success_model.self_check():", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
