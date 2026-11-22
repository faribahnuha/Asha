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

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

API_TITLE: str = "Asha Facility Recommendation API"
API_VERSION: str = "0.1.0"

# Loaded once, at import time (process startup) — not reloaded per request.
# This is the thing that does NOT work the same way on every host; see the
# deployment notes in this repo's README / this session's chat for why.
BLOOD_POINTS: pd.DataFrame
FACILITY_SERVICES: pd.DataFrame
BLOOD_POINTS, FACILITY_SERVICES = demo.load_datasets()

app = FastAPI(title=API_TITLE, version=API_VERSION)


class RecommendRequest(BaseModel):
    """Request body for POST /recommend."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Patient latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Patient longitude")
    scenario: str = Field(..., description="One of: " + ", ".join(key for key, _ in demo.SCENARIO_MENU))


class FacilityResult(BaseModel):
    """One ranked facility in the response."""

    name: str
    city: str
    state: str
    lat: float
    lon: float
    drive_min: float
    score: float
    confirmed: bool
    note: str


class RecommendResponse(BaseModel):
    """Response body for POST /recommend."""

    scenario_label: str
    candidates_considered: int
    results: list[FacilityResult]
    weights: dict[str, float]


@app.get("/health")
def health_check() -> dict[str, str]:
    """Plain liveness check — confirms the process is up and data loaded,
    without running the model.
    """
    assert len(BLOOD_POINTS) > 0 and len(FACILITY_SERVICES) > 0, "datasets failed to load at startup"
    return {"status": "ok", "candidates_loaded": str(len(BLOOD_POINTS))}


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

    facility_results = [
        FacilityResult(
            name=str(row.get("name", row["unique_id"])),
            city=str(row.get("city", "")) if pd.notna(row.get("city")) else "",
            state=str(row.get("state_std", "")) if pd.notna(row.get("state_std")) else "",
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            drive_min=round(float(row["drive_min"]), 1),
            score=round(float(row["score"]), 3),
            confirmed=bool(row.get("needed_services_confirmed", False)),
            note=str(row.get("note", "")),
        )
        for _, row in result.iterrows()
    ]

    return RecommendResponse(
        scenario_label=dict(demo.SCENARIO_MENU)[request.scenario],
        candidates_considered=int(len(nearby)),
        results=facility_results,
        weights={key: round(float(value), 3) for key, value in weights.items()},
    )
