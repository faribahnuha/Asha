# P(capable) model: evaluation

Generated from `evaluation_report.json` by `modeling/capability/train_capability.py`
(re-run that script to refresh both files).

## Setup

- **Question:** does a facility's own record credibly show it provides a capability on site?
- **Labels:** 240 facility × capability claims, stratified across 8 capabilities and three evidence
  levels (strong / weak-or-moderate / none). 218 decided, 22 marked *unclear* and excluded.
  Labelled by an AI from the record text, blind to model scores, using `labels/RUBRIC.md`.
- **Model:** one shared L2-regularised logistic regression across capabilities, 23 features from
  the record text and record quality. Penalty chosen by facility-grouped 5-fold cross-validation
  (λ = 3).
- **Baselines:** *keyword* ("any mention = capable") and *evidence level* (share of true claims at
  each level, learned on the other folds).

## Accuracy (out of fold: no facility is scored by a model that saw it)

| | AUC | Brier | Log loss | Accuracy @0.5 | Calibration error |
|---|---|---|---|---|---|
| **Model** | **0.942** (95% CI 0.908–0.970) | **0.093** | **0.310** | **87.2%** | 0.049 |
| Evidence-level baseline | 0.796 | 0.159 | 0.481 | 70.6% | 0.035 |
| Keyword baseline | 0.726 | 0.247 | 1.169 | 74.8% | 0.242 |

Reweighted to the real mix of evidence levels in the 9,474 hospitals: model AUC 0.951,
accuracy 88.6%, calibration error 0.057.

### Calibration (does "80%" mean 80%?)

| Predicted | Labels | Mean predicted | Observed true |
|---|---|---|---|
| 0.0–0.2 | 70 | 0.04 | 0.07 |
| 0.2–0.4 | 12 | 0.30 | 0.00 |
| 0.4–0.6 | 16 | 0.51 | 0.50 |
| 0.6–0.8 | 27 | 0.71 | 0.63 |
| 0.8–1.0 | 93 | 0.92 | 0.95 |

### By capability (≈27 labels each, so wide uncertainty)

| Capability | Labels | True | Model AUC | Level baseline AUC |
|---|---|---|---|---|
| obstetric_care | 26 | 12 | 1.00 | 0.94 |
| surgical_capacity | 26 | 17 | 0.90 | 0.83 |
| anaesthesia | 28 | 23 | 1.00 | 1.00 |
| emergency_24x7 | 28 | 11 | 0.96 | 0.88 |
| trauma_surgery | 27 | 7 | 0.89 | 0.83 |
| critical_care | 26 | 15 | 0.84 | 0.80 |
| inpatient_medicine | 28 | 11 | 0.86 | 0.70 |
| blood_service_claim | 29 | 22 | 1.00 | 0.87 |

## Independent checks (sources the model never saw)

| Check | Result |
|---|---|
| Hospitals whose text claims blood: linked to a licensed government blood bank vs not | AUC 0.60; mean P 0.93 vs 0.89 |
| Hospitals with a government blood bank whose text never mentions blood | 163 |
| LaQshya-certified public hospitals vs other public hospitals, P(obstetric care) | AUC 0.68; mean P 0.58 vs 0.37 |
| Same, P(surgical capacity) | AUC 0.64 |

**Reading:** the text model ranks certified and licensed places higher, but only modestly. Text alone
cannot tell a licensed blood bank from an unlicensed claim, and it misses 163 hospitals that do have
a government-listed blood bank. That is why the ranker adds the directory link and LaQshya as
separate evidence on top of P(capable), instead of trusting text alone.

## What drives the predictions (standardised coefficients)

Largest effects: more fields mentioning the capability (+1.06), being a hospital (+0.94),
a single-specialty name such as eye/dental/lab/IVF (−0.74), moderate or strong evidence
(+0.63 / +0.60), and the capability appearing in the capability list (+0.54).

## Limits

- Labels are AI-made from the same text the features come from, so accuracy measures agreement
  with a careful reader of the record, not real-world truth. A 20-minute human spot-check of
  30 labels would add an agreement figure.
- The penalty was chosen on the same folds used for reporting (slightly optimistic).
- Per-capability results rest on about 27 labels each.
