"""
==============================================================================
MODULE: evaluate_v2.py                         (cross-cutting evaluation tool, not a request stage)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: calls modeling/success/recommend_v2.py (new model) and the
  untouched entropy-TOPSIS pipeline (modeling/recommend/recommend.py) on the same simulated patients,
  plus two simple baselines, and scores every method's top pick on one shared yardstick built from
  gold data. Writes modeling/evaluation/results.json (report.md is written from it).
- WHAT THIS FILE DOES: label-free safety and blood-access metrics on 1,000 simulated requests,
  location and parameter stability, and a one-at-a-time sensitivity sweep over parameters.csv.
- KNOWN LIMITATIONS / ASSUMPTIONS: patients are simulated around listed facilities with random offsets
  (no population data), so the urban/rural mix is a design choice reported alongside every number;
  the yardstick uses gold evidence levels and median travel times, which are themselves estimates;
  "appropriate" means the record shows moderate-or-strong evidence (or LaQshya) for every required
  capability, not that the hospital was verified on the ground.
==============================================================================
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _d in ("candidate_pool", "scoring", "recommend", "demo", "success"):
    sys.path.append(str(ROOT / "modeling" / _d))
import recommend as v1  # noqa: E402
import try_it_out as v1_demo  # noqa: E402
import recommend_v2 as v2  # noqa: E402
import success_model as sm  # noqa: E402

SEED: int = 20261122
N_QUERIES: int = 1000
N_SENSITIVITY: int = 200
SCENARIOS: tuple[str, ...] = ("pregnancy_bleeding", "injury_bleeding", "severe_anemia", "not_sure")
LAQ_CAPS: frozenset[str] = frozenset({"obstetric_care", "surgical_capacity", "anaesthesia"})
DOMINANCE_TOLERANCE_MIN: float = 5.0
STANDALONE_BANK = re.compile(r"red cross|rotary|lions|voluntary|blood (bank|centre|center)$|^blood bank|samiti|charitable", re.I)


def simulate_queries(world: dict[str, object], n: int) -> list[dict[str, object]]:
    """Half the patients within ~5 km of a listed hospital (town-like), half up to ~35 km away (rural-like)."""
    assert n > 0 and n % 2 == 0, "n must be a positive even number"
    rng = np.random.default_rng(SEED)
    sites = world["sites"]
    anchors = rng.integers(0, len(sites["id"]), size=n)
    spread = np.where(np.arange(n) < n // 2, 0.045, 0.32)
    lat = np.clip(sites["lat"][anchors] + rng.uniform(-1, 1, n) * spread, 6.5, 37.0)
    lon = np.clip(sites["lon"][anchors] + rng.uniform(-1, 1, n) * spread, 68.5, 97.0)
    queries = [dict(qid=i, lat=float(lat[i]), lon=float(lon[i]), scenario=SCENARIOS[i % 4], design="near_town" if i < n // 2 else "offset")
               for i in range(n)]
    assert len(queries) == n, "one query per draw"
    return queries


def yardstick(cand: dict[str, np.ndarray], scen: dict[str, object], window: float) -> dict[str, np.ndarray]:
    """Shared, method-independent facts for every candidate: evidence, verified blood, time to treatment."""
    assert "minutes" in cand and "best_bank_minutes" in cand, "score_candidates output required"
    assert window > 0, "window must be positive"
    req = scen["required"]
    lvl = np.vstack([np.where(cand["laq_mot"] & (c in LAQ_CAPS), 3, cand[f"level_{c}"]) for c in req])
    min_level = int(scen["min_blood_level"])
    in_house = cand["in_house_level"] >= max(min_level, 2)
    blood_min = np.where(in_house, cand["prep_minutes"], cand["best_bank_minutes"])
    t_treat = np.maximum(cand["minutes"], blood_min)
    out = dict(n_moderate=(lvl >= 2).sum(axis=0), all_moderate=(lvl >= 2).all(axis=0), any_missing=(lvl == 0).any(axis=0),
               verified_blood_in_window=blood_min <= window, time_to_treatment=t_treat, in_window=t_treat <= window)
    assert len(out["time_to_treatment"]) == len(cand["id"]), "one value per candidate"
    return out


def dominated(ys: dict[str, np.ndarray], i: int) -> bool:
    """True if another candidate has at least as much evidence and is faster to treatment, and is better on one."""
    assert 0 <= i < len(ys["n_moderate"]), "index out of range"
    e, t = ys["n_moderate"], ys["time_to_treatment"]
    better = (e >= e[i]) & (t <= t[i] - DOMINANCE_TOLERANCE_MIN) | (e > e[i]) & (t <= t[i])
    better[i] = False
    result = bool(better.any())
    assert isinstance(result, bool), "must be boolean"
    return result


def v1_pick(bp: object, fs: object, q: dict[str, object]) -> dict[str, object] | None:
    """Old TOPSIS model's top pick, mapped to a treatment site where possible."""
    assert q["scenario"] in SCENARIOS, "unknown scenario"
    near = v1_demo.narrow_to_nearby(bp, q["scenario"], q["lat"], q["lon"])
    if near.empty:
        return None
    res, _ = v1.recommend(near, fs, q["lat"], q["lon"], q["scenario"], v1_demo.ASSUMED_DETOUR_FACTOR, v1_demo.ASSUMED_AVG_SPEED_KMPH)
    top = res.iloc[0]
    site = top["unique_id"] if isinstance(top["unique_id"], str) else (top["linked_vf_facility"] if isinstance(top.get("linked_vf_facility"), str) else None)
    out = dict(site=site, standalone_bank=site is None and bool(STANDALONE_BANK.search(str(top["name"]))), name=str(top["name"]))
    assert "site" in out, "pick must name a site or None"
    return out


def judge(pick_site: str | None, cand: dict[str, np.ndarray], ys: dict[str, np.ndarray], standalone: bool) -> dict[str, object]:
    """Score one method's pick on the shared yardstick."""
    assert len(cand["id"]) == len(ys["n_moderate"]), "yardstick must match candidates"
    idx = np.where(cand["id"] == pick_site)[0] if pick_site is not None else np.array([], dtype=int)
    if len(idx) == 0:                                  # a blood bank or place that is not a treatment site
        return dict(treatment_site=False, appropriate=False, unsafe=True, standalone_bank=standalone, verified_blood=False,
                    in_window=False, time_to_treatment=None, dominated=None, p_success=None)
    i = int(idx[0])
    out = dict(treatment_site=True, appropriate=bool(ys["all_moderate"][i]), unsafe=bool(ys["any_missing"][i]), standalone_bank=False,
               verified_blood=bool(ys["verified_blood_in_window"][i]), in_window=bool(ys["in_window"][i]),
               time_to_treatment=float(ys["time_to_treatment"][i]), dominated=dominated(ys, i), p_success=float(cand["p_success"][i]))
    assert out["time_to_treatment"] >= 0, "time must be non-negative"
    return out


def run_query(world: dict[str, object], bp: object, fs: object, q: dict[str, object]) -> dict[str, object]:
    """All four methods on one patient, judged on the same yardstick."""
    assert -90 <= q["lat"] <= 90, "bad latitude"
    scen = world["scenarios"][q["scenario"]]
    cand, meta = v2.score_candidates(world, dict(q, hb=None, night=False, rh_negative=False))
    if len(cand["id"]) == 0:
        return dict(q, area="none", skipped=True)
    ys = yardstick(cand, scen, float(meta["window_minutes"]))
    res = v2.recommend(world, q["lat"], q["lon"], q["scenario"])
    v2_site = res["results"][0]["facility_id"] if res["results"] else None
    any_ok = bool((ys["all_moderate"] & ys["in_window"]).any())
    nearest_blood = int(np.argmin(cand["km"] + np.where(cand["in_house_level"] > 0, 0, 1e6)))   # baseline A (sites with a licensed bank)
    claims_all = np.vstack([cand[f"level_{c}"] >= 1 for c in scen["required"]]).all(axis=0)
    base_b = int(np.argmin(np.where(claims_all, cand["km"], np.inf))) if claims_all.any() else None
    old = v1_pick(bp, fs, q)
    out = dict(q, area=meta["area"], skipped=False, v2_status=res["status"], feasible_option_exists=any_ok,
               v2=judge(v2_site, cand, ys, False),
               v1=judge(old["site"], cand, ys, old["standalone_bank"]) if old else None,
               nearest_licensed_blood=judge(str(cand["id"][nearest_blood]), cand, ys, False) if (cand["in_house_level"] > 0).any() else None,
               nearest_claiming_all=judge(str(cand["id"][base_b]), cand, ys, False) if base_b is not None else None)
    assert "v2" in out, "new model must be judged"
    return out


def summarise(rows: list[dict[str, object]], method: str, recommended_only: bool = False) -> dict[str, object]:
    """Rates for one method over a set of query results."""
    assert method in ("v2", "v1", "nearest_licensed_blood", "nearest_claiming_all"), "unknown method"
    sel = [r for r in rows if not r["skipped"] and r.get(method) and (not recommended_only or r["v2_status"] == "recommended")]
    if not sel:
        return dict(n=0)
    j = [r[method] for r in sel]
    rate = lambda key: round(float(np.mean([bool(x[key]) for x in j])), 3)
    tt = [x["time_to_treatment"] for x in j if x["time_to_treatment"] is not None]
    dom = [x["dominated"] for x in j if x["dominated"] is not None]
    out = dict(n=len(sel), appropriate=rate("appropriate"), unsafe=rate("unsafe"), not_a_treatment_site=round(1 - rate("treatment_site"), 3),
               standalone_blood_bank=rate("standalone_bank"), verified_blood_in_window=rate("verified_blood"),
               treatment_with_blood_in_window=rate("in_window"), median_minutes_to_treatment=round(float(np.median(tt)), 1) if tt else None,
               dominated=round(float(np.mean(dom)), 3) if dom else None)
    assert out["n"] > 0, "summary needs rows"
    return out


def abstention(rows: list[dict[str, object]]) -> dict[str, object]:
    """How often the new model abstains, and whether an appropriate in-window option actually existed."""
    sel = [r for r in rows if not r["skipped"]]
    assert sel, "no evaluable queries"
    abst = [r for r in sel if r["v2_status"] != "recommended"]
    rec = [r for r in sel if r["v2_status"] == "recommended"]
    out = dict(abstain_rate=round(len(abst) / len(sel), 3),
               abstained_but_appropriate_option_existed=round(float(np.mean([r["feasible_option_exists"] for r in abst])), 3) if abst else None,
               recommended_and_no_appropriate_option_existed=round(float(np.mean([not r["feasible_option_exists"] for r in rec])), 3) if rec else None,
               share_with_any_appropriate_option=round(float(np.mean([r["feasible_option_exists"] for r in sel])), 3))
    assert 0 <= out["abstain_rate"] <= 1, "rate out of range"
    return out


def stability(world: dict[str, object], queries: list[dict[str, object]]) -> dict[str, object]:
    """Share of top picks unchanged when the patient moves ~1 km, or speeds/detour shift by +/-20%."""
    assert len(queries) >= 10, "need queries"
    rng = np.random.default_rng(SEED + 1)
    top = lambda w, q, la, lo: (v2.recommend(w, la, lo, q["scenario"])["results"] or [{"facility_id": None}])[0]["facility_id"]
    base = [top(world, q, q["lat"], q["lon"]) for q in queries]
    ang = rng.uniform(0, 2 * np.pi, len(queries))
    moved = [top(world, q, q["lat"] + 0.009 * np.sin(a), q["lon"] + 0.009 * np.cos(a)) for q, a in zip(queries, ang)]
    out = dict(location_1km_unchanged=round(float(np.mean([a == b for a, b in zip(base, moved)])), 3))
    for name, factor in (("speeds_minus_20pct", 0.8), ("speeds_plus_20pct", 1.2), ("detour_plus_20pct", None)):
        p = dict(world["params"])
        if factor is None:
            p["road_detour_factor"] *= 1.2
        else:
            p["speed_urban_kmph"] *= factor
            p["speed_rural_kmph"] *= factor
        w = dict(world, params=p)
        out[f"{name}_unchanged"] = round(float(np.mean([top(w, q, q["lat"], q["lon"]) == b for q, b in zip(queries, base)])), 3)
    assert all(0 <= v <= 1 for v in out.values()), "rates out of range"
    return out


def sensitivity(world: dict[str, object], queries: list[dict[str, object]]) -> list[dict[str, object]]:
    """One-at-a-time sweep: each parameter at its low and high value; effect on abstention and top picks."""
    assert len(queries) >= 10, "need queries"
    import csv
    rows = list(csv.DictReader(open(sm.PARAMETERS_PATH, encoding="utf-8")))
    run = lambda w: [v2.recommend(w, q["lat"], q["lon"], q["scenario"]) for q in queries]
    base = run(world)
    base_top = [(r["results"] or [{"facility_id": None}])[0]["facility_id"] for r in base]
    out = [dict(parameter="(base)", value=None, abstain_rate=round(float(np.mean([r["status"] != "recommended" for r in base])), 3), top_pick_changed=0.0)]
    for row in rows:                                   # ~30 parameters, fixed table
        for which in ("sensitivity_low", "sensitivity_high"):
            p = dict(world["params"])
            p[row["parameter"]] = float(row[which])
            res = run(dict(world, params=p))
            tops = [(r["results"] or [{"facility_id": None}])[0]["facility_id"] for r in res]
            out.append(dict(parameter=row["parameter"], setting=which.split("_")[1], value=float(row[which]),
                            abstain_rate=round(float(np.mean([r["status"] != "recommended" for r in res])), 3),
                            top_pick_changed=round(float(np.mean([a != b for a, b in zip(tops, base_top)])), 3)))
    assert len(out) == 1 + 2 * len(rows), "two settings per parameter"
    return out


def main() -> dict[str, object]:
    """Run everything and write results.json."""
    t0 = time.time()
    world = v2.load_world()
    bp, fs = v1_demo.load_datasets()
    queries = simulate_queries(world, N_QUERIES)
    rows = [run_query(world, bp, fs, q) for q in queries]
    ok = [r for r in rows if not r["skipped"]]
    methods = ("v2", "v1", "nearest_licensed_blood", "nearest_claiming_all")
    by = lambda subset: {m: summarise(subset, m) for m in methods}
    results = dict(n_queries=len(rows), n_evaluable=len(ok), design="half within ~5 km of a listed hospital, half offset up to ~35 km",
                   area_mix={a: sum(r["area"] == a for r in ok) for a in ("urban", "rural")},
                   overall=by(ok), when_v2_recommends={m: summarise(ok, m, True) for m in methods},
                   by_scenario={s: by([r for r in ok if r["scenario"] == s]) for s in SCENARIOS},
                   by_area={a: by([r for r in ok if r["area"] == a]) for a in ("urban", "rural")},
                   abstention=abstention(ok), abstention_by_area={a: abstention([r for r in ok if r["area"] == a]) for a in ("urban", "rural")},
                   stability=stability(world, queries[: N_SENSITIVITY]), sensitivity=sensitivity(world, queries[: N_SENSITIVITY]))
    results["runtime_seconds"] = round(time.time() - t0, 1)
    (Path(__file__).resolve().parent / "results.json").write_text(json.dumps(results, indent=2))
    assert results["n_evaluable"] > 0.8 * N_QUERIES, "too many queries had no hospital in range"
    return results


def self_check() -> bool:
    """Dominance and judging logic on fixed in-memory data."""
    ys = dict(n_moderate=np.array([3, 3, 1]), time_to_treatment=np.array([20.0, 40.0, 10.0]))
    check_1 = dominated(ys, 1) and not dominated(ys, 0) and not dominated(ys, 2)
    cand = dict(id=np.array(["a"]), p_success=np.array([0.9]))
    full = dict(ys, n_moderate=np.array([3]), time_to_treatment=np.array([20.0]), all_moderate=np.array([True]),
                any_missing=np.array([False]), verified_blood_in_window=np.array([True]), in_window=np.array([True]))
    j = judge("a", cand, full, False)
    j_bank = judge(None, cand, full, True)
    check_2 = j["appropriate"] and not j["unsafe"] and j_bank["unsafe"] and j_bank["standalone_bank"]
    passed = check_1 and check_2
    if not passed:
        print("evaluate_v2.self_check(): FAILED", [check_1, check_2])
    return passed


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        sys.exit(0 if self_check() else 1)
    print(json.dumps(main(), indent=2))
