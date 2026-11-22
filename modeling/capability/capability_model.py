"""
==============================================================================
MODULE: capability_model.py                     (P(capable) component, v2 model)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: reads one row of dataset/gold/treatment_sites.csv
  (built by pre-processing/gold.py) plus a capability name, turns it into a numeric feature
  vector, and applies the trained coefficients in modeling/capability/capability_model.json
  (written by train_capability.py). Its output, P(capable), is one of the three factors the
  chance-of-success ranker multiplies.
- WHAT THIS FILE DOES: feature construction + logistic-regression inference for "does this
  facility credibly provide this capability on site?". Pure numpy; no training code here.
- KNOWN LIMITATIONS / ASSUMPTIONS: trained on 218 AI-made labels (labels/RUBRIC.md) that
  measure whether the facility's OWN RECORD credibly supports the claim, not real-world truth.
  Independent sources (government blood bank directory, LaQshya) are deliberately NOT features,
  so they can be used to test this model; the ranker adds them afterwards as separate evidence.
  Probabilities for capabilities with few labels (8 capabilities x ~27 labels) are uncertain.
==============================================================================
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

# The scenario-level capabilities the model knows. Fixed data, never mutated.
CAPABILITIES: tuple[str, ...] = (
    "obstetric_care", "surgical_capacity", "anaesthesia", "emergency_24x7",
    "trauma_surgery", "critical_care", "inpatient_medicine", "blood_service_claim",
)
STRONG_FIELDS: frozenset[str] = frozenset({"description", "procedure", "capability"})

# Facility names that signal a single-specialty or non-clinical place (eye, dental, lab, IVF ...).
# Derived from the record itself (its own name), so it is a legitimate text feature.
NARROW_SPECIALTY = re.compile(
    r"\b(eye|netra|nethra|netralaya|vision|dental|dentist|teeth|smile|orthodontic|lab|labs|laboratory|"
    r"diagnostic|diagnostics|imaging|scans?|pathology|ivf|fertility|physio\w*|ayurved\w*|homoeo\w*|homeo\w*|"
    r"skin|derma\w*|hair|cosmetic|speech|hearing)\b", re.I)

FEATURE_NAMES: tuple[str, ...] = (
    "level_weak", "level_moderate", "level_strong", "log_n_fields",
    "quote_in_description", "quote_in_procedure", "quote_in_capability", "specialties_only",
    "is_hospital", "narrow_specialty_name", "beds_known", "log_beds",
    "has_official_site_source", "completeness_score", "log_flag_count",
) + tuple(f"cap_{c}" for c in CAPABILITIES)

MODEL_PATH: Path = Path(__file__).resolve().parent / "capability_model.json"


def _to_float(value: object, default: float = 0.0) -> float:
    """Parse a CSV cell to float; empty/invalid cells become `default`."""
    assert isinstance(default, float), "default must be a float"
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    assert not math.isnan(parsed), "parsed value must not be NaN"
    return parsed


def build_features(site: dict[str, str], capability: str) -> np.ndarray:
    """Feature vector for one (site row, capability). Order matches FEATURE_NAMES."""
    assert capability in CAPABILITIES, f"unknown capability: {capability}"
    assert "facility_id" in site, "site row must come from treatment_sites.csv"

    level = int(_to_float(site.get(f"cap_{capability}_level"), 0.0))
    n_fields = _to_float(site.get(f"cap_{capability}_n_fields"), 0.0)
    quotes = json.loads(site.get(f"cap_{capability}_quotes") or "[]")
    fields = {q["field"] for q in quotes}
    beds = _to_float(site.get("beds"), 0.0)
    values = [
        float(level == 1), float(level == 2), float(level == 3), math.log1p(n_fields),
        float("description" in fields), float("procedure" in fields), float("capability" in fields),
        float(bool(fields) and fields <= {"specialties", "equipment"}),
        float(site.get("facility_type") in ("hospital", "nursing_home")),
        float(bool(NARROW_SPECIALTY.search(site.get("name") or ""))),
        float(site.get("beds_known") == "True"), math.log1p(beds),
        float(site.get("has_official_site_source") == "True"),
        _to_float(site.get("completeness_score"), 0.0),
        math.log1p(_to_float(site.get("flag_count"), 0.0)),
    ] + [float(capability == c) for c in CAPABILITIES]
    vector = np.array(values, dtype=float)

    assert vector.shape == (len(FEATURE_NAMES),), "feature vector length must match FEATURE_NAMES"
    assert np.isfinite(vector).all(), "features must be finite"
    return vector


def standardize(matrix: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Scale columns with stored training means/stds (stds of 0 are left unscaled)."""
    assert matrix.shape[1] == len(means) == len(stds), "column count mismatch"
    assert (stds >= 0).all(), "standard deviations must be non-negative"
    safe = np.where(stds > 0, stds, 1.0)
    scaled = (matrix - means) / safe
    assert scaled.shape == matrix.shape, "scaling must keep the shape"
    return scaled


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    assert np.isfinite(z).all(), "logits must be finite"
    out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
    assert ((out >= 0) & (out <= 1)).all(), "probabilities must be in [0, 1]"
    return out


def load_model(path: Path = MODEL_PATH) -> dict[str, object]:
    """Load trained coefficients and scaling written by train_capability.py."""
    assert path.exists(), f"model file missing: {path} (run modeling/capability/train_capability.py)"
    model = json.loads(path.read_text())
    assert model["feature_names"] == list(FEATURE_NAMES), "model was trained on a different feature set"
    assert len(model["coef"]) == len(FEATURE_NAMES), "coefficient count must match features"
    return model


def predict_proba(sites: list[dict[str, str]], capability: str, model: dict[str, object]) -> np.ndarray:
    """P(capable) for each site row, for one capability."""
    assert capability in CAPABILITIES, f"unknown capability: {capability}"
    assert len(sites) >= 1, "need at least one site"
    matrix = np.vstack([build_features(s, capability) for s in sites])
    scaled = standardize(matrix, np.array(model["means"]), np.array(model["stds"]))
    probs = sigmoid(scaled @ np.array(model["coef"]) + float(model["intercept"]))  # type: ignore[arg-type]
    assert probs.shape == (len(sites),), "one probability per site"
    return probs


def self_check() -> bool:
    """Checks feature construction, scaling and the logistic function on fixed in-memory data."""
    strong_site = {"facility_id": "A", "name": "City Hospital", "facility_type": "hospital", "beds": "100",
                   "beds_known": "True", "has_official_site_source": "True", "completeness_score": "0.9",
                   "flag_count": "1", "cap_critical_care_level": "3", "cap_critical_care_n_fields": "3",
                   "cap_critical_care_quotes": json.dumps([{"field": "description", "quote": "ICU"}])}
    eye_site = {"facility_id": "B", "name": "Good Eye Clinic", "facility_type": "clinic", "beds": "",
                "beds_known": "False", "has_official_site_source": "False", "completeness_score": "0.5",
                "flag_count": "0", "cap_critical_care_level": "1", "cap_critical_care_n_fields": "1",
                "cap_critical_care_quotes": json.dumps([{"field": "specialties", "quote": "criticalCare"}])}
    f_strong, f_eye = build_features(strong_site, "critical_care"), build_features(eye_site, "critical_care")
    names = list(FEATURE_NAMES)
    check_1 = f_strong[names.index("level_strong")] == 1.0 and f_strong[names.index("quote_in_description")] == 1.0
    check_2 = f_eye[names.index("narrow_specialty_name")] == 1.0 and f_eye[names.index("specialties_only")] == 1.0
    check_3 = f_strong[names.index("cap_critical_care")] == 1.0 and f_strong[names.index("cap_anaesthesia")] == 0.0
    scaled = standardize(np.array([[1.0, 5.0], [3.0, 5.0]]), np.array([2.0, 5.0]), np.array([1.0, 0.0]))
    check_4 = np.allclose(scaled, [[-1.0, 0.0], [1.0, 0.0]])
    check_5 = np.allclose(sigmoid(np.array([0.0, 50.0, -50.0])), [0.5, 1.0, 0.0], atol=1e-9)
    passed = all([check_1, check_2, check_3, check_4, check_5])
    if not passed:
        print("capability_model.self_check(): FAILED", [check_1, check_2, check_3, check_4, check_5])
    return passed


if __name__ == "__main__":
    import sys

    ok = self_check()
    print("capability_model.self_check():", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
