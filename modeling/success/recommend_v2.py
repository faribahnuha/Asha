"""
==============================================================================
MODULE: recommend_v2.py                       (v2 model: rank by chance of success, explain)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: loads gold data once (treatment_sites.csv + capability_scores.csv,
  blood_sources.csv) and, per request, calls success_model.py (P(capable), P(arrives)) and
  blood_access.py (P(blood in time)). Its output dict is what an API or UI displays. Replaces the
  entropy-TOPSIS recommend.py for the v2 design; the old pipeline is left untouched.
- WHAT THIS FILE DOES: takes patient coordinates + scenario (+ optional Hb, Rh-negative, night),
  scores every hospital within reach by P(success) = P(capable) x P(blood in time) x P(arrives),
  returns the top options with the evidence quotes and blood source behind each, says "no confident
  option" below a threshold, and explains why the nearest hospital was not chosen.
- KNOWN LIMITATIONS / ASSUMPTIONS: every caveat of success_model.py and blood_access.py applies;
  P(success) is a model estimate built from assumed parameters, so the UI must show it as a
  likelihood with a "call ahead" prompt, never as a guarantee.
==============================================================================
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np

import success_model as sm
import blood_access as ba

CAPABILITIES: tuple[str, ...] = ("obstetric_care", "surgical_capacity", "anaesthesia", "emergency_24x7",
                                 "trauma_surgery", "critical_care", "inpatient_medicine", "blood_service_claim")
EMERGENCY_WINDOW: str = "window_high_min"
CALL_AHEAD: tuple[str, ...] = ("Is {products} available right now?", "Is a surgeon and anaesthetist on duty now?",
                               "Do you need a replacement blood donor?", "Can you cross-match on arrival?")


def load_world() -> dict[str, object]:
    """Gold sites (with P(capable) per capability), licensed banks, parameters and scenarios."""
    sites = list(csv.DictReader(open(sm.ROOT / "dataset/gold/treatment_sites.csv", encoding="utf-8")))
    scores = {r["facility_id"]: r for r in csv.DictReader(open(sm.ROOT / "dataset/gold/capability_scores.csv", encoding="utf-8"))}
    assert sites and all(s["facility_id"] in scores for s in sites), "capability_scores.csv must cover every site"
    col = lambda name: np.array([s[name] for s in sites])
    arr = dict(id=col("facility_id"), name=col("name"), address=col("address"), city=col("city"), state=col("state"), phone=col("phone"),
               lat=col("lat").astype(float), lon=col("lon").astype(float), location_confidence=col("location_confidence"),
               in_house_level=col("in_house_directory_level").astype(int), link_confidence=col("in_house_link_confidence"),
               in_house_is_24x7=col("in_house_is_24x7"), in_house_source_id=col("in_house_source_id"),
               laq_mot=col("laqshya_mot_current") == "True", laq_lr=col("laqshya_lr_current") == "True",
               p_blood_claim=np.array([float(scores[s["facility_id"]]["p_blood_service_claim"]) for s in sites]))
    for c in CAPABILITIES:                             # 8 capabilities, fixed
        arr[f"p_{c}"] = np.array([float(scores[s["facility_id"]][f"p_{c}"]) for s in sites])
        arr[f"quotes_{c}"] = col(f"cap_{c}_quotes")
    for c in CAPABILITIES + ("neonatal_care", "ct_imaging"):   # evidence levels (0-3), used for display and evaluation
        arr[f"level_{c}"] = col(f"cap_{c}_level").astype(int)
    world = dict(sites=arr, banks=ba.load_banks(), params=sm.load_parameters(), scenarios=sm.load_scenarios())
    assert len(arr["id"]) <= sm.MAX_SITES, "more sites than the national dataset"
    return world


def subset(arr: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """The same site arrays restricted to the candidates in mask."""
    assert mask.dtype == bool, "mask must be boolean"
    out = {k: v[mask] for k, v in arr.items()}
    assert len(out["id"]) == int(mask.sum()), "subset size mismatch"
    return out


def score_candidates(world: dict[str, object], request: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Every factor for every hospital within reach of the patient."""
    params, scen = world["params"], world["scenarios"][request["scenario"]]
    assert request["scenario"] in world["scenarios"], "unknown scenario"
    allsites = world["sites"]
    d_all = sm.haversine_km(float(request["lat"]), float(request["lon"]), allsites["lat"], allsites["lon"])
    cand = subset(allsites, d_all <= params["max_search_km"])
    cand["km"] = d_all[d_all <= params["max_search_km"]]
    rural = sm.area_is_rural(float(request["lat"]), float(request["lon"]), allsites["lat"], allsites["lon"], params)
    window = sm.window_minutes(scen, params, request.get("hb"))
    cv = params["travel_time_cv_rural"] if rural else params["travel_time_cv_urban"]
    cand["minutes"] = sm.travel_minutes(cand["km"], rural, params)
    cand["p_arrive"] = sm.p_within(cand["minutes"], window, cv)
    caps = {c: cand[f"p_{c}"] for c in CAPABILITIES}
    cand["p_capable"], adjusted = sm.p_capable_for_scenario(caps, scen["required"], cand["laq_mot"], cand["laq_lr"], params)
    for c, p in adjusted.items():                     # at most 3 required capabilities
        cand[f"adj_{c}"] = p
    ctx = dict(window=window, rural=rural, night=bool(request.get("night", False)), rh_negative=bool(request.get("rh_negative", False)),
               emergency=scen["window_parameter"] == EMERGENCY_WINDOW)
    blood = ba.p_blood_in_time(cand, world["banks"], int(scen["min_blood_level"]), ctx, params) if len(cand["id"]) else {}
    cand.update(blood)
    cand["p_success"] = cand["p_capable"] * cand.get("p_blood", np.zeros(0)) * cand["p_arrive"]
    meta = dict(window_minutes=round(window, 1), area="rural" if rural else "urban", travel_cv=cv, candidates=int(len(cand["id"])))
    assert len(cand["p_success"]) == len(cand["id"]), "one score per candidate"
    return cand, meta


def shown(p: float, cap: float) -> float:
    """A probability as shown to users: never above `cap` (no estimate built on assumed parameters
    and web-scraped evidence justifies certainty). Ranking and abstention use the uncapped value."""
    assert 0.0 <= p <= 1.0 + 1e-9, "probability out of range"
    assert 0.5 <= cap <= 1.0, "cap must be between 0.5 and 1"
    return round(min(float(p), cap), 3)


def blood_source_text(cand: dict[str, np.ndarray], i: int, banks: dict[str, np.ndarray], cap: float = 1.0) -> dict[str, object]:
    """Where the blood would come from, in words, with the numbers behind it."""
    assert 0 <= i < len(cand["id"]), "index out of range"
    assert "p_in_house" in cand, "blood factors missing"
    if cand["p_in_house"][i] >= cand["p_external"][i] and cand["in_house_level"][i] > 0:
        source = f"Own blood bank, licensed in the government directory (level {cand['in_house_level'][i]}, link {cand['link_confidence'][i]})"
    elif cand["p_in_house"][i] >= cand["p_external"][i]:
        source = "Own blood bank claimed in the hospital's listing (not in the government directory; call to confirm)"
    else:
        b = int(cand["best_bank_idx"][i])
        source = (f"Nearest suitable licensed bank: {banks['name'][b]} ({cand['best_bank_km'][i]:.1f} km from the hospital, "
                  f"about {cand['best_bank_minutes'][i]:.0f} min including preparation)")
    out = dict(source=source, p_blood_in_time=shown(float(cand["p_blood"][i]), cap), backup_banks_in_time=int(cand["n_backups"][i]),
               preparation_minutes=float(cand["prep_minutes"][i]))
    assert out["p_blood_in_time"] <= 1.0, "probability out of range"
    return out


def explain(cand: dict[str, np.ndarray], i: int, scen: dict[str, object], world: dict[str, object], meta: dict[str, object]) -> dict[str, object]:
    """One recommendation card: the three factors, quotes behind each requirement, blood source, caveats."""
    assert 0 <= i < len(cand["id"]), "index out of range"
    sigma = math.sqrt(math.log(1 + float(meta["travel_cv"]) ** 2))
    cap = float(world["params"].get("display_probability_cap", 1.0))
    evidence = []
    for c in scen["required"]:                         # at most 3
        quotes = json.loads(str(cand[f"quotes_{c}"][i]) or "[]")
        evidence.append(dict(capability=c, p=shown(float(cand[f"adj_{c}"][i]), cap),
                             quote=quotes[0]["quote"] if quotes else None, field=quotes[0]["field"] if quotes else None,
                             independent_certification="LaQshya" if (cand["laq_mot"][i] and c in ("obstetric_care", "surgical_capacity", "anaesthesia")) else None))
    # supporting capabilities: modelled ones by P >= 0.5; unmodelled ones (NICU, CT) by moderate+ evidence level
    also = [c for c in scen["supporting"] if (f"p_{c}" in cand and float(cand[f"p_{c}"][i]) >= 0.5)
            or (f"level_{c}" in cand and int(cand[f"level_{c}"][i]) >= 2)]
    card = dict(facility_id=str(cand["id"][i]), name=str(cand["name"][i]), city=str(cand["city"][i]), state=str(cand["state"][i]),
                phone=str(cand["phone"][i]) or None, lat=float(cand["lat"][i]), lon=float(cand["lon"][i]),
                p_success=shown(float(cand["p_success"][i]), cap), p_capable=shown(float(cand["p_capable"][i]), cap),
                p_arrive=shown(float(cand["p_arrive"][i]), cap), straight_line_km=round(float(cand["km"][i]), 1),
                travel_minutes_estimate=round(float(cand["minutes"][i])),
                travel_minutes_80pct_range=[round(float(cand["minutes"][i]) * math.exp(-1.2816 * sigma)), round(float(cand["minutes"][i]) * math.exp(1.2816 * sigma))],
                evidence=evidence, also_has=also, blood=blood_source_text(cand, i, world["banks"], cap),
                location_confidence=str(cand["location_confidence"][i]),
                call_ahead=[q.format(products=scen["blood_products"]) for q in CALL_AHEAD])
    assert 0.0 <= card["p_success"] <= 1.0, "probability out of range"
    return card


def why_not(cand: dict[str, np.ndarray], i: int) -> str:
    """Plain reason a hospital lost: its weakest factor."""
    assert 0 <= i < len(cand["id"]), "index out of range"
    factors = {"its record gives little evidence it can treat this": cand["p_capable"][i],
               "the right blood products are unlikely to be there in time": cand["p_blood"][i],
               "it is unlikely to be reached within the time window": cand["p_arrive"][i]}
    reason = min(factors, key=factors.get)
    assert isinstance(reason, str), "reason must be text"
    return f"{cand['name'][i]} is closest ({cand['km'][i]:.1f} km) but {reason} (chance of success {cand['p_success'][i]:.0%})."


def nearest_bank(world: dict[str, object], lat: float, lon: float, min_level: int) -> dict[str, object] | None:
    """Nearest licensed bank stocking the scenario's level: a fallback anchor, since many district hospitals
    that house these banks are missing from the facility dataset."""
    assert min_level in (1, 2, 3), "blood level must be 1, 2 or 3"
    banks = world["banks"]
    ok = banks["level"] >= max(min_level, 2)
    if not ok.any():
        return None
    d = sm.haversine_km(lat, lon, banks["lat"][ok], banks["lon"][ok])
    j = int(np.argmin(d))
    idx = int(np.where(ok)[0][j])
    out = dict(name=str(banks["name"][idx]), straight_line_km=round(float(d[j]), 1), level=int(banks["level"][idx]),
               is_24x7=str(banks["is_24x7"][idx]), phone=str(banks["phone"][idx]) or None,
               location_confidence=str(banks["location_confidence"][idx]))
    assert out["straight_line_km"] >= 0, "distance must be non-negative"
    return out


def recommend(world: dict[str, object], lat: float, lon: float, scenario: str, hb: float | None = None,
              rh_negative: bool = False, night: bool = False, top_n: int = 3) -> dict[str, object]:
    """Top hospitals by chance of success, with explanations; abstains when nothing is likely enough."""
    assert 6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0, "patient coordinates must be inside India"
    assert scenario in world["scenarios"] and 1 <= top_n <= 10, "unknown scenario or bad top_n"
    request = dict(lat=lat, lon=lon, scenario=scenario, hb=hb, rh_negative=rh_negative, night=night)
    cand, meta = score_candidates(world, request)
    scen, params = world["scenarios"][scenario], world["params"]
    if len(cand["id"]) == 0:
        return dict(request=request, meta=meta, status="no_hospital_in_range", results=[], why_not_closest=None)
    best = float(cand["p_success"].max())
    status = "recommended" if best >= params["abstain_threshold"] else "no_confident_option"
    if status == "recommended":
        order = np.lexsort((cand["minutes"], -cand["p_success"]))[:top_n]
    else:
        # nothing is likely enough inside the window: rank the best definitive-care options reachable within
        # twice the window, so the list is "most capable reachable", not "least hopeless by a hair"
        cv = float(meta["travel_cv"])
        relaxed = cand["p_capable"] * cand["p_blood"] * sm.p_within(cand["minutes"], 2 * float(meta["window_minutes"]), cv)
        order = np.lexsort((cand["minutes"], -relaxed))[:top_n]
    nearest = int(np.argmin(cand["km"]))
    out = dict(request=request, meta=dict(meta, scenario_label=scen["label"], blood_products=scen["blood_products"]),
               status=status, results=[explain(cand, int(i), scen, world, meta) for i in order],
               why_not_closest=None if nearest in order.tolist() else why_not(cand, nearest))
    out["nearest_licensed_blood_bank"] = nearest_bank(world, lat, lon, int(scen["min_blood_level"]))
    if int(scen["min_blood_level"]) >= 3:              # whole blood nearby can still buy time to stabilise
        out["nearest_whole_blood_bank"] = nearest_bank(world, lat, lon, 2)
    if status == "no_confident_option":
        out["message"] = (f"No hospital within reach has at least a {params['abstain_threshold']:.0%} estimated chance of success within "
                          f"{meta['window_minutes']:.0f} minutes. Listed: the most capable hospitals reachable within twice that time. "
                          "Call ahead; stabilise at the nearest facility (or the hospital housing the nearest licensed blood bank) if needed.")
    assert len(out["results"]) <= top_n, "never return more than top_n"
    return out


def self_check() -> bool:
    """Fixed in-memory world: a capable hospital with in-house blood beats a closer clinic with nothing."""
    params = sm.load_parameters()
    scenarios = sm.load_scenarios()
    n = 2
    sites = dict(id=np.array(["good", "near"]), name=np.array(["Good Hospital", "Near Clinic"]), city=np.array(["X", "X"]),
                 state=np.array(["S", "S"]), phone=np.array(["", ""]), lat=np.array([25.60, 25.592]), lon=np.array([85.10, 85.10]),
                 location_confidence=np.array(["high", "high"]), in_house_level=np.array([3, 0]), link_confidence=np.array(["high", ""]),
                 in_house_is_24x7=np.array(["yes", ""]), in_house_source_id=np.array(["b1", ""]), laq_mot=np.zeros(n, bool),
                 laq_lr=np.zeros(n, bool), p_blood_claim=np.array([0.9, 0.0]))
    for c in CAPABILITIES:
        sites[f"p_{c}"] = np.array([0.95, 0.05])
        sites[f"quotes_{c}"] = np.array(['[{"field": "description", "quote": "We perform C-sections"}]', "[]"])
    banks = dict(id=np.array(["b1"]), name=np.array(["Good Hospital Blood Bank"]), lat=np.array([25.60]), lon=np.array([85.10]),
                 level=np.array([3]), is_24x7=np.array(["yes"]), phone=np.array([""]), location_confidence=np.array(["high"]))
    world = dict(sites=sites, banks=banks, params=params, scenarios=scenarios)
    res = recommend(world, 25.59, 85.10, "pregnancy_bleeding", top_n=1)   # top 1, so the closer clinic must be explained
    check_1 = res["results"][0]["facility_id"] == "good" and res["status"] == "recommended"
    check_2 = res["why_not_closest"] is not None and "Near Clinic" in res["why_not_closest"]
    check_3 = res["results"][0]["evidence"][0]["quote"] == "We perform C-sections"
    check_4 = res["results"][0]["blood"]["source"].startswith("Own blood bank, licensed")
    passed = all([check_1, check_2, check_3, check_4])
    if not passed:
        print("recommend_v2.self_check(): FAILED", [check_1, check_2, check_3, check_4])
    return passed


if __name__ == "__main__":
    import sys

    ok = self_check()
    print("recommend_v2.self_check():", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
