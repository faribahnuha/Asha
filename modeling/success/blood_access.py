"""
==============================================================================
MODULE: blood_access.py                       (v2 model: P(right blood products in time))
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: uses success_model.py's distance, travel-time and
  window-probability helpers; reads blood sources from dataset/gold/blood_sources.csv and in-house
  links from treatment_sites.csv (via recommend_v2.py). Returns P(blood in time) per candidate
  hospital plus the source that makes it possible, for recommend_v2.py to rank and explain.
- WHAT THIS FILE DOES: for each hospital, combines (a) its own blood bank (licensed directory link,
  or only claimed in its text) and (b) the nearest licensed banks that stock the scenario's products,
  each with an availability chance and a time-to-arrive, into one probability that the right blood
  is at that hospital within the clinical window.
- KNOWN LIMITATIONS / ASSUMPTIONS: nobody knows live stock, so every bank gets an assumed
  availability (parameters.csv); sources are treated as independent backups; blood moves by road
  at patient speed; at night, banks not listed as 24x7 are treated as closed and unknown hours as
  uncertain; a hospital's in-house bank known only from its own text gets reduced trust.
==============================================================================
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import success_model as sm

BLOOD_SOURCES_PATH: Path = sm.ROOT / "dataset/gold/blood_sources.csv"


def load_banks(path: Path = BLOOD_SOURCES_PATH) -> dict[str, np.ndarray]:
    """Licensed directory banks (levels 2 and 3) as parallel arrays."""
    assert path.exists(), f"missing {path}"
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["kind"] == "directory_blood_bank"]
    banks = dict(id=np.array([r["source_id"] for r in rows]), name=np.array([r["name"] for r in rows]),
                 lat=np.array([float(r["lat"]) for r in rows]), lon=np.array([float(r["lon"]) for r in rows]),
                 level=np.array([int(r["blood_level"]) for r in rows]), is_24x7=np.array([r["is_24x7"] for r in rows]),
                 phone=np.array([r["phone"] for r in rows]), location_confidence=np.array([r["location_confidence"] for r in rows]))
    assert 0 < len(rows) <= sm.MAX_BANKS, "unexpected number of directory banks"
    return banks


def hours_factor(is_24x7: np.ndarray, night: bool, params: dict[str, float]) -> np.ndarray:
    """1 by day; at night 1 if 24x7, 0 if not, and an assumed chance if hours are unknown."""
    assert set(np.unique(is_24x7)) <= {"yes", "no", "unknown", ""}, "unexpected 24x7 values"
    assert 0 <= params["p_stock_unknown_hours_night"] <= 1, "probability out of range"
    if not night:
        return np.ones(len(is_24x7))
    return np.where(is_24x7 == "yes", 1.0, np.where(is_24x7 == "no", 0.0, params["p_stock_unknown_hours_night"]))


def rh_factor(levels: np.ndarray, rh_negative: bool, params: dict[str, float]) -> np.ndarray:
    """Chance the bank also has Rh-negative stock (1 when the patient is not Rh-negative)."""
    assert levels.ndim == 1, "levels must be a vector"
    assert 0 <= params["p_rh_neg_stock_l3"] <= 1, "probability out of range"
    if not rh_negative:
        return np.ones(len(levels))
    return np.where(levels >= 3, params["p_rh_neg_stock_l3"], params["p_rh_neg_stock_l2"])


def prep_minutes(min_level: int, emergency: bool, params: dict[str, float]) -> float:
    """Time for the bank to issue the products: emergency release vs cross-match; plasma must thaw."""
    assert min_level in (1, 2, 3), "blood level must be 1, 2 or 3"
    assert params["blood_issue_emergency_min"] > 0, "issue time must be positive"
    issue = params["blood_issue_emergency_min"] if emergency else params["blood_issue_crossmatched_min"]
    total = max(issue, params["plasma_thaw_min"]) if min_level >= 3 else issue
    return float(total)


def in_house_availability(sites: dict[str, np.ndarray], min_level: int, night: bool, rh_negative: bool,
                          params: dict[str, float]) -> np.ndarray:
    """Chance each hospital can supply the products from its OWN bank (licensed link, else text claim)."""
    assert len(sites["in_house_level"]) == len(sites["p_blood_claim"]), "site arrays must match"
    assert min_level in (1, 2, 3), "blood level must be 1, 2 or 3"
    trust = np.select([sites["link_confidence"] == "high", sites["link_confidence"] == "medium", sites["link_confidence"] == "low"],
                      [params["link_trust_high"], params["link_trust_medium"], params["link_trust_low"]], 0.0)
    licensed_ok = sites["in_house_level"] >= max(min_level, 2)
    licensed = np.where(licensed_ok, trust * params["p_inhouse_stock"], 0.0)
    licensed *= hours_factor(sites["in_house_is_24x7"], night, params) * rh_factor(sites["in_house_level"], rh_negative, params)
    components = 1.0 if min_level <= 1 else params["p_components_unverified_bank"]
    claimed = sites["p_blood_claim"] * params["p_inhouse_stock"] * components
    claimed *= hours_factor(np.full(len(claimed), "unknown"), night, params) * rh_factor(np.full(len(claimed), 2), rh_negative, params)
    avail = np.where(sites["in_house_level"] > 0, licensed, claimed)     # a licensed link replaces the text claim
    assert ((avail >= 0) & (avail <= 1)).all(), "availability out of range"
    return avail


def external_access(sites: dict[str, np.ndarray], banks: dict[str, np.ndarray], min_level: int, ctx: dict[str, object],
                    params: dict[str, float]) -> dict[str, np.ndarray]:
    """For each hospital: probability-weighted access to its K nearest qualifying licensed banks."""
    assert len(sites["lat"]) <= sm.MAX_SITES, "too many candidate sites"
    assert min_level in (1, 2, 3), "blood level must be 1, 2 or 3"
    ok = banks["level"] >= max(min_level, 2)
    lat, lon, lvl, hrs = banks["lat"][ok], banks["lon"][ok], banks["level"][ok], banks["is_24x7"][ok]
    k = min(int(params["max_backup_banks"]), int(ok.sum()))
    n = len(sites["lat"])
    if k == 0:                                         # no bank anywhere stocks this level
        return dict(p_any=np.zeros(n), best_idx=np.full(n, -1), best_minutes=np.full(n, np.inf),
                    best_km=np.full(n, np.inf), n_backups_in_window=np.zeros(n, dtype=int))
    dist = sm.pairwise_km(sites["lat"], sites["lon"], lat, lon)
    nearest = np.argsort(dist, axis=1)[:, :k]
    d_k = np.take_along_axis(dist, nearest, axis=1)
    t_k = sm.travel_minutes(d_k, bool(ctx["rural"]), params) + float(ctx["prep"])
    cv = params["travel_time_cv_rural"] if ctx["rural"] else params["travel_time_cv_urban"]
    p_time = sm.p_within(t_k.ravel(), float(ctx["window"]), cv).reshape(t_k.shape)
    avail = params["p_bank_has_stock"] * hours_factor(hrs[nearest].ravel(), bool(ctx["night"]), params).reshape(t_k.shape)
    avail *= rh_factor(lvl[nearest].ravel(), bool(ctx["rh_negative"]), params).reshape(t_k.shape)
    p_none = np.prod(1 - avail * p_time, axis=1)
    out = dict(p_any=1 - p_none, best_idx=np.where(ok)[0][nearest[:, 0]], best_minutes=t_k[:, 0], best_km=d_k[:, 0],
               n_backups_in_window=(avail * p_time > 0.5).sum(axis=1))
    assert ((out["p_any"] >= 0) & (out["p_any"] <= 1)).all(), "probability out of range"
    return out


def p_blood_in_time(sites: dict[str, np.ndarray], banks: dict[str, np.ndarray], min_level: int, ctx: dict[str, object],
                    params: dict[str, float]) -> dict[str, np.ndarray]:
    """P(the scenario's blood products are at each hospital within the window), with its main source."""
    assert {"window", "rural", "night", "rh_negative", "emergency"} <= set(ctx), "incomplete request context"
    assert len(sites["lat"]) >= 1, "need at least one site"
    ctx = dict(ctx, prep=prep_minutes(min_level, bool(ctx["emergency"]), params))
    cv = params["travel_time_cv_rural"] if ctx["rural"] else params["travel_time_cv_urban"]
    a_in = in_house_availability(sites, min_level, bool(ctx["night"]), bool(ctx["rh_negative"]), params)
    p_in = a_in * sm.p_within(np.full(len(a_in), float(ctx["prep"])), float(ctx["window"]), cv)
    ext = external_access(sites, banks, min_level, ctx, params)
    total = 1 - (1 - p_in) * (1 - ext["p_any"])
    result = dict(p_blood=total, p_in_house=p_in, p_external=ext["p_any"], best_bank_idx=ext["best_idx"],
                  best_bank_minutes=ext["best_minutes"], best_bank_km=ext["best_km"], n_backups=ext["n_backups_in_window"],
                  prep_minutes=np.full(len(total), float(ctx["prep"])))
    assert ((total >= 0) & (total <= 1)).all(), "probability out of range"
    return result


def self_check() -> bool:
    """Fixed in-memory checks: licensed in-house beats nothing; night closes non-24x7 banks; level gating."""
    params = {"p_stock_unknown_hours_night": 0.5, "p_rh_neg_stock_l3": 0.8, "p_rh_neg_stock_l2": 0.5, "blood_issue_emergency_min": 15.0,
              "blood_issue_crossmatched_min": 45.0, "plasma_thaw_min": 25.0, "link_trust_high": 1.0, "link_trust_medium": 0.8,
              "link_trust_low": 0.5, "p_inhouse_stock": 1.0, "p_components_unverified_bank": 0.5, "p_bank_has_stock": 1.0,
              "max_backup_banks": 2.0, "speed_urban_kmph": 30.0, "speed_rural_kmph": 40.0, "road_detour_factor": 1.0,
              "travel_time_cv_urban": 0.2, "travel_time_cv_rural": 0.3}
    sites = dict(lat=np.array([20.0, 20.0]), lon=np.array([78.0, 78.0]), in_house_level=np.array([3, 0]),
                 link_confidence=np.array(["high", ""]), in_house_is_24x7=np.array(["yes", ""]), p_blood_claim=np.array([0.0, 0.0]))
    far_banks = dict(lat=np.array([25.0]), lon=np.array([78.0]), level=np.array([3]), is_24x7=np.array(["no"]))
    ctx = dict(window=60.0, rural=False, night=False, rh_negative=False, emergency=True)
    res = p_blood_in_time(sites, far_banks, 3, ctx, params)
    check_1 = res["p_blood"][0] > 0.95 and res["p_blood"][1] < 0.01     # in-house L3 vs only a bank 550 km away
    check_2 = prep_minutes(3, True, params) == 25.0 and prep_minutes(1, False, params) == 45.0
    check_3 = np.allclose(hours_factor(np.array(["yes", "no", "unknown"]), True, params), [1.0, 0.0, 0.5])
    near_l2 = dict(lat=np.array([20.01]), lon=np.array([78.0]), level=np.array([2]), is_24x7=np.array(["yes"]))
    res_l3 = p_blood_in_time(sites, near_l2, 3, ctx, params)            # an L2 bank cannot meet an L3 need
    check_4 = res_l3["p_external"][1] == 0.0 or not np.isfinite(res_l3["best_bank_km"][1])
    passed = all([check_1, check_2, check_3, check_4])
    if not passed:
        print("blood_access.self_check(): FAILED", [check_1, check_2, check_3, check_4])
    return passed


if __name__ == "__main__":
    import sys

    ok = self_check()
    print("blood_access.self_check():", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
