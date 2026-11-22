"""
==============================================================================
MODULE: app_v2.py                              (HTTP wrapper around the v2 chance-of-success model)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: wraps modeling/success/recommend_v2.py. It does not re-implement
  any scoring; it parses a request, calls recommend(), and serialises the result. Infrastructure glue,
  like the v1 app, so it is not held to modeling/CLAUDE.md's per-function assertion density.
- WHAT THIS FILE DOES: POST /recommend takes {lat, lon, scenario} (+ optional hb, rh_negative, night)
  and returns the top hospitals ranked by P(success). The response keeps every field the React UI
  already reads (name, city, state, lat, lon, drive_min, distance_km, score, confirmed, note, address,
  phone, services, weights, candidates_considered, scenario_label) and adds the v2 detail (the three
  probabilities, verbatim evidence quotes, blood source, call-ahead questions, abstention status).
- KNOWN LIMITATIONS / ASSUMPTIONS: `score` is now P(success), an estimate built on stated assumptions
  (dataset/ref/parameters.csv), not a guarantee; `weights` holds the top result's three factors
  because v2 has no criterion weights; service keys are mapped onto the UI's existing vocabulary.
==============================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_MODELING_ROOT = Path(__file__).resolve().parent.parent
for _sibling_dir in ("success", "capability"):
    _path = str(_MODELING_ROOT / _sibling_dir)
    if _path not in sys.path:
        sys.path.append(_path)

import recommend_v2 as R  # noqa: E402

API_TITLE = "Asha Facility Recommendation API (v2: chance of success)"
API_VERSION = "2.0.0"
WORLD = R.load_world()
SITE_INDEX = {fid: i for i, fid in enumerate(WORLD["sites"]["id"].tolist())}

app = FastAPI(title=API_TITLE, version=API_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class RecommendRequest(BaseModel):
    lat: float = Field(..., ge=6.0, le=38.0, description="Patient latitude (India)")
    lon: float = Field(..., ge=68.0, le=98.0, description="Patient longitude (India)")
    scenario: str = Field(..., description="pregnancy_bleeding | injury_bleeding | severe_anemia | not_sure")
    hb: Optional[float] = Field(None, ge=1.0, le=20.0, description="Patient haemoglobin in g/dL, if known")
    rh_negative: bool = Field(False, description="Patient is known to be Rh-negative")
    night: bool = Field(False, description="Request is at night (banks not listed as 24x7 treated as closed)")


class Evidence(BaseModel):
    capability: str
    p: float
    quote: Optional[str]
    field: Optional[str]
    independent_certification: Optional[str]


class FacilityResult(BaseModel):
    # fields the existing UI reads
    name: str
    city: str
    state: str
    lat: float
    lon: float
    drive_min: float
    distance_km: float
    score: float
    confirmed: bool
    note: str
    address: str
    phone: str
    services: list[str]
    # v2 detail
    facility_id: str
    p_success: float
    p_capable: float
    p_blood_in_time: float
    p_arrive_in_time: float
    travel_minutes_80pct_range: list[float]
    evidence: list[Evidence]
    blood_source: str
    backup_blood_banks_in_time: int
    location_confidence: str
    call_ahead: list[str]


class RecommendResponse(BaseModel):
    scenario_label: str
    candidates_considered: int
    results: list[FacilityResult]
    weights: dict[str, float]
    status: str
    message: Optional[str] = None
    why_not_closest: Optional[str] = None
    window_minutes: float
    area: str
    blood_products_needed: str
    nearest_licensed_blood_bank: Optional[dict] = None
    nearest_whole_blood_bank: Optional[dict] = None


def ui_services(i: int, card: dict) -> list[str]:
    """Map v2 capability probabilities onto the UI's existing service keys (P >= 0.5, or moderate+ evidence)."""
    s = WORLD["sites"]
    p = lambda c: float(s[f"p_{c}"][i])
    lvl = lambda c: int(s[f"level_{c}"][i])
    quotes = json.loads(str(s["quotes_obstetric_care"][i]) or "[]") + json.loads(str(s["quotes_surgical_capacity"][i]) or "[]")
    out = []
    if p("critical_care") >= 0.5: out.append("icu")
    if lvl("neonatal_care") >= 2: out.append("nicu")
    if any(q.get("service") == "csection" for q in quotes): out.append("c_section")
    if p("obstetric_care") >= 0.5: out.append("obstetrics")
    if p("emergency_24x7") >= 0.5 and p("surgical_capacity") >= 0.5: out.append("emergency_surgery")
    if p("trauma_surgery") >= 0.5: out.append("trauma_care")
    if p("surgical_capacity") >= 0.5: out.append("surgery")
    if p("inpatient_medicine") >= 0.5: out.append("general_medicine")
    if p("emergency_24x7") >= 0.5: out.append("ed_24x7")
    if card["blood"]["p_blood_in_time"] >= 0.5: out.append("transfusion")
    if int(s["in_house_level"][i]) > 0 or p("blood_service_claim") >= 0.5: out.append("blood_bank")
    if lvl("ct_imaging") >= 2: out.append("ct_scan")
    return out


def to_result(card: dict, status: str) -> FacilityResult:
    i = SITE_INDEX[card["facility_id"]]
    note = (f"Chance of success {card['p_success']:.0%} (capable {card['p_capable']:.0%} · blood in time "
            f"{card['blood']['p_blood_in_time']:.0%} · arrives in time {card['p_arrive']:.0%}). Blood: {card['blood']['source']}.")
    if status != "recommended":
        note = "Not a confident option. " + note
    return FacilityResult(
        name=card["name"], city=card["city"] or "", state=card["state"] or "", lat=card["lat"], lon=card["lon"],
        drive_min=float(card["travel_minutes_estimate"]), distance_km=float(card["straight_line_km"]), score=card["p_success"],
        confirmed=card["p_capable"] >= 0.5, note=note, address=str(WORLD["sites"]["address"][i] or ""),
        phone=card["phone"] or "", services=ui_services(i, card), facility_id=card["facility_id"], p_success=card["p_success"],
        p_capable=card["p_capable"], p_blood_in_time=card["blood"]["p_blood_in_time"], p_arrive_in_time=card["p_arrive"],
        travel_minutes_80pct_range=[float(x) for x in card["travel_minutes_80pct_range"]],
        evidence=[Evidence(**e) for e in card["evidence"]], blood_source=card["blood"]["source"],
        backup_blood_banks_in_time=card["blood"]["backup_banks_in_time"], location_confidence=card["location_confidence"],
        call_ahead=card["call_ahead"])


@app.get("/")
def root() -> dict[str, object]:
    return {"service": API_TITLE, "version": API_VERSION, "endpoints": {"health": "/health", "recommend": "POST /recommend", "docs": "/docs"}}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "v2 chance of success", "treatment_sites": str(len(WORLD["sites"]["id"])),
            "licensed_blood_banks": str(len(WORLD["banks"]["id"]))}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(request: RecommendRequest) -> RecommendResponse:
    if request.scenario not in WORLD["scenarios"]:
        raise HTTPException(status_code=422, detail=f"scenario must be one of {sorted(WORLD['scenarios'])}")
    res = R.recommend(WORLD, request.lat, request.lon, request.scenario, hb=request.hb,
                      rh_negative=request.rh_negative, night=request.night)
    if not res["results"]:
        raise HTTPException(status_code=404, detail="No hospital found within the search radius of this location")
    top = res["results"][0]
    return RecommendResponse(
        scenario_label=res["meta"]["scenario_label"], candidates_considered=res["meta"]["candidates"],
        results=[to_result(c, res["status"]) for c in res["results"]],
        weights={"p_capable": top["p_capable"], "p_blood_in_time": top["blood"]["p_blood_in_time"], "p_arrive_in_time": top["p_arrive"]},
        status=res["status"], message=res.get("message"), why_not_closest=res.get("why_not_closest"),
        window_minutes=res["meta"]["window_minutes"], area=res["meta"]["area"], blood_products_needed=res["meta"]["blood_products"],
        nearest_licensed_blood_bank=res.get("nearest_licensed_blood_bank"), nearest_whole_blood_bank=res.get("nearest_whole_blood_bank"))
