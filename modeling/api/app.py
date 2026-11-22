"""
==============================================================================
MODULE: app.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: this is an HTTP wrapper around
  modeling/recommend/recommend.py — it does not re-implement any scoring
  logic, it only parses a request, calls recommend(), and serializes the
  result. It is infrastructure glue, like modeling/demo/try_it_out.py, not a
  pipeline stage, so it is NOT held to modeling/CLAUDE.md's strict rules
  (bounded loops, assertion density, etc.) the way candidate_pool.py,
  scoring.py, and recommend.py are — those rules exist for the model logic
  itself, not for a web framework's request-handling glue around it.
- WHAT THIS FILE DOES: a small FastAPI app with one real endpoint,
  POST /recommend, that takes a patient location and scenario and returns
  the same top-3 result modeling/demo/try_it_out.py prints to a terminal,
  as JSON. Loads the dataset CSVs once at startup, not per-request.
- KNOWN LIMITATIONS / ASSUMPTIONS: uses the same ASSUMED (not measured)
  detour_factor/avg_speed_kmph as the rest of modeling/demo/ — see
  candidate_pool.py's header. This file is meant to be run with uvicorn
  locally, or deployed behind any ASGI-compatible host; see this repo's
  notes on what that does and doesn't mean for Vercel specifically.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELING_ROOT = _REPO_ROOT / "modeling"
for _sibling_dir in ("candidate_pool", "scoring", "recommend", "demo"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import recommend as R  # noqa: E402
import try_it_out as demo  # noqa: E402

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

API_TITLE: str = "Asha Facility Recommendation API"
API_VERSION: str = "0.1.0"

# Loaded once, at import time (process startup) — not reloaded per request.
# This is the thing that does NOT work the same way on every host; see the
# deployment notes in this repo's README / this session's chat for why.
BLOOD_POINTS: pd.DataFrame
FACILITY_SERVICES: pd.DataFrame
BLOOD_POINTS, FACILITY_SERVICES = demo.load_datasets()

app = FastAPI(title=API_TITLE, version=API_VERSION)

# A frontend hosted on a different domain (e.g. a Vercel-hosted UI calling
# this API on Render) is a cross-origin request by definition — without
# this, the browser blocks it silently and it looks like the API is broken
# when it isn't. "*" is deliberately permissive for a public, read-only,
# no-auth recommendation endpoint with no patient data stored; narrow this
# to specific origins if that stops being true.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    """Request body for POST /recommend."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Patient latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Patient longitude")
    scenario: str = Field(..., description="One of: " + ", ".join(key for key, _ in demo.SCENARIO_MENU))


class FacilityResult(BaseModel):
    """One ranked facility in the response, including the "why" behind it —
    the same detail modeling/demo/generate_dashboard_data.py bakes into the
    static dashboard, computed live here instead."""

    name: str
    city: str
    state: str
    lat: float
    lon: float
    drive_min: float
    score: float
    confirmed: bool
    blood_level: Optional[int]
    blood_level_description: str
    services_has: list[str]
    services_missing: list[str]
    services_note: str
    note: str


class RandomLocationResponse(BaseModel):
    """Response body for GET /random-location."""

    lat: float
    lon: float


class RecommendResponse(BaseModel):
    """Response body for POST /recommend."""

    scenario_label: str
    candidates_considered: int
    results: list[FacilityResult]
    weights: dict[str, float]


@app.get("/")
def root() -> dict[str, object]:
    """There is no page at the bare root path — this exists only so visiting
    the API's base URL in a browser shows what's actually available instead
    of looking broken. The real endpoints are /health and /recommend.
    """
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "endpoints": {"health": "/health", "recommend": "POST /recommend", "docs": "/docs"},
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Plain liveness check — confirms the process is up and data loaded,
    without running the model.
    """
    assert len(BLOOD_POINTS) > 0 and len(FACILITY_SERVICES) > 0, "datasets failed to load at startup"
    return {"status": "ok", "candidates_loaded": str(len(BLOOD_POINTS))}


def _build_facility_result(row: pd.Series, scenario_key: str) -> FacilityResult:
    """Turn one row of recommend()'s result into a FacilityResult, adding
    the human-readable blood-level description and service breakdown.

    PARAMETERS:
        row: one row from recommend()'s result DataFrame.
        scenario_key: passed to demo.describe_services().
    """
    is_confirmed = bool(row.get("needed_services_confirmed", False))
    service_breakdown = demo.describe_services(row["unique_id"], scenario_key, is_confirmed, FACILITY_SERVICES)
    blood_level = int(row["blood_level"]) if pd.notna(row.get("blood_level")) else None

    return FacilityResult(
        name=str(row.get("name", row["unique_id"])),
        city=str(row.get("city", "")) if pd.notna(row.get("city")) else "",
        state=str(row.get("state_std", "")) if pd.notna(row.get("state_std")) else "",
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        drive_min=round(float(row["drive_min"]), 1),
        score=round(float(row["score"]), 3),
        confirmed=is_confirmed,
        blood_level=blood_level,
        blood_level_description=demo.BLOOD_LEVEL_DESCRIPTIONS.get(blood_level, "Unknown"),
        services_has=service_breakdown["has"],
        services_missing=service_breakdown["missing"],
        services_note=service_breakdown["note"],
        note=str(row.get("note", "")),
    )


@app.get("/random-location", response_model=RandomLocationResponse)
def random_location_endpoint() -> RandomLocationResponse:
    """Pick a random real facility's coordinates (jittered slightly) to use
    as a stand-in patient location — never a point that could land in the
    ocean or across a border. Reuses demo.pick_random_location() exactly as
    the CLI demo does; no separate logic for the API.
    """
    patient_lat, patient_lon = demo.pick_random_location(BLOOD_POINTS, random_seed=None)
    return RandomLocationResponse(lat=patient_lat, lon=patient_lon)


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(request: RecommendRequest) -> RecommendResponse:
    """Run the real model for one patient request and return the top 3.

    PARAMETERS:
        request: validated by pydantic before this function ever runs —
            lat/lon range and scenario presence are already guaranteed.
    """
    valid_scenarios = [key for key, _ in demo.SCENARIO_MENU]
    if request.scenario not in valid_scenarios:
        raise HTTPException(status_code=422, detail=f"scenario must be one of {valid_scenarios}")

    nearby = demo.narrow_to_nearby(BLOOD_POINTS, request.scenario, request.lat, request.lon)
    if nearby.empty:
        raise HTTPException(status_code=404, detail="No candidates found anywhere near this location")

    result, weights = R.recommend(
        nearby, FACILITY_SERVICES, request.lat, request.lon, request.scenario,
        demo.ASSUMED_DETOUR_FACTOR, demo.ASSUMED_AVG_SPEED_KMPH,
    )
    assert len(result) >= 1, "recommend() must return at least one row when candidates exist"

    facility_results = [_build_facility_result(row, request.scenario) for _, row in result.iterrows()]

    return RecommendResponse(
        scenario_label=dict(demo.SCENARIO_MENU)[request.scenario],
        candidates_considered=int(len(nearby)),
        results=facility_results,
        weights={key: round(float(value), 3) for key, value in weights.items()},
    )
