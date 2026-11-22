# Prompt: Diagnose the Candidate-Pool Coverage Gap

This fills the context/task/rules template using only what has been verified against
`model-design.md`, `dataset/blood_points_v2.csv`, and `dataset/facility_services_v2.csv` in this
conversation. No numbers below are invented — each is a direct count from the data or a line
from the design doc. This is not an ML training task (no base model, no training stage), so
those template fields are translated to their closest equivalent in a non-ML decision-model
pipeline rather than left as ML jargon that doesn't apply here.

<context>
- **Model / architecture:** entropy-weighted TOPSIS multi-criteria ranking model (`model-design.md`,
  "Summary" + "Code" sections). Not a trained model — no weights are learned; `entropy_weights()`
  derives weights per-patient from the spread of candidate values, `severity_adjusted_weights()`
  applies a stated boost, `topsis()` scores closeness to an ideal. No neural network, no training
  run, no checkpoint.
- **Development stage:** candidate-pool and scoring-criteria design, pre-implementation. The
  pre-processing pipeline referenced by the doc (`pre-processing/bronze.sql`, `silver.py`,
  `gold.py`) exists only as empty 1-byte placeholder files — no cleaning/merging logic has been
  written yet, so nothing below has been tested end-to-end.
- **Dataset size & format:**
  - `dataset/blood_points_v2.csv`: 3,460 rows, 31 columns, CSV. 3,269 rows have
    `in_candidate_pool=True`. Of those, 475 are `place_kind=facility_with_blood_bank` (VF-linked)
    and 2,794 are `place_kind=directory_blood_bank` (directory-only).
  - `dataset/facility_services_v2.csv`: 46,807 rows, 11 columns, CSV, one row per
    (facility, service) pair. Covers only `unique_id`s that are VF-linked facilities — **0 rows**
    for directory-only blood banks except where `blood_points_v2.csv.linked_vf_facility` points to
    a VF record (157 of 2,794 directory-only candidates have this link; the other 2,637 have no
    path to service data at all).
  - Known imbalance: service-tag `evidence_strength` across all 46,807 rows is weak 22,534,
    strong 19,263, derived 3,138, moderate 1,872 — roughly half the service claims are weak
    web-text extractions, not verified capability.
  - Known missingness in `blood_points_v2.csv`: `has_components` is null for 561/3,460 rows,
    `is_24x7` is null for 1,041/3,460 rows. The existing `blood_level` column currently resolves
    both nulls to the lowest tier (1) rather than tracking them as unknown.
- **Hardware & budget constraints:** Databricks Free Edition, numpy/pandas only, no live
  routing/internet access in that environment, target build time of about 3 hours end-to-end
  (`model-design.md`, "Three-hour build plan" and "Databricks implementation" sections).
- **Current known failure modes** (no trained-model metrics exist; these are verified structural
  issues instead):
  1. The `needed_services` criterion currently has no way to distinguish "confirmed lacking this
     service" from "no service data exists for this facility." Both score as 0. Since 2,637 of
     3,269 candidates (81%) have zero path to service data, this systematically scores the
     majority of the candidate pool as if every scenario-relevant service were confirmed absent.
  2. 10 of 35 states/UTs in the candidate pool have zero VF-linked (service-tagged) facilities
     (e.g. Manipur: 1 facility, Mizoram: 2, Goa: 2, Chandigarh: 2, Uttarakhand: 4 — several other
     states/UTs have 0). Any scenario whose score depends on service tags has no confirmed
     signal to rank on in those regions.
  3. No test cases or sensitivity sweep have been implemented yet (`model-design.md`, "Testing
     without ground truth" describes ~10 hand-written cases and a boost/window sensitivity sweep,
     but none exist in the repo), so there is currently no way to measure how often the
     unconfirmed-vs-absent default actually changes the top-3 recommendation.
</context>

<task>
Diagnose why scoring unconfirmed service data as equivalent to confirmed-absent service data
under-serves 81% of the candidate pool (the directory-only blood banks with no
`facility_services_v2.csv` rows), and design a concrete fix: a schema change and a scoring-logic
change that let `needed_services` distinguish "unknown" from "confirmed absent" without
excluding directory-only facilities from any scenario, consistent with the time-window-only
gating already specified in `model-design.md`. Deliverable should be a revised `needed_services`
computation (or a replacement criterion) plus the data-schema change it depends on, not a full
reimplementation of the model.
</task>

<rules>
- Do not invent missing metrics, file paths, or library versions beyond what's listed in
  `model-design.md`, `dataset/blood_points_v2.csv`, and `dataset/facility_services_v2.csv`; ask
  before assuming a field, join key, or row count not already verified in this repo.
- Show step-by-step reasoning and trade-offs (e.g. unconfirmed-as-neutral vs. a separate
  confidence-weighted term vs. a two-stage score) before giving final code or schema changes.
- Include pass/fail checks or sanity tests — at minimum: coverage of the fix across the 2,794
  directory-only candidates, behavior in the 10 states/UTs with zero VF-linked facilities, and at
  least one of the hand-written test cases from `model-design.md`'s "Testing without ground
  truth" section — so the fix can be verified before running it against the full 3,269-row pool.
</rules>
