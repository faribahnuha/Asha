# v2 model evaluation (Stage 6)

Produced by `modeling/evaluation/evaluate_v2.py` (raw numbers in `results.json`; re-run to refresh).

## Setup

- **1,000 simulated patients**, 250 per scenario. Half placed within ~5 km of a listed hospital
  (town-like), half offset up to ~35 km (rural-like). The model's own density test classed
  541 as urban and 459 as rural. There is no population data, so this mix is a design choice.
- **Four methods** on the same patients:
  - **v2**: chance-of-success model (this repo)
  - **v1**: the previous entropy-TOPSIS model
  - **Nearest licensed blood**: nearest hospital with a government-listed blood bank
  - **Nearest claiming all**: nearest hospital whose record mentions every required capability
- **One shared yardstick**, independent of any method, built from gold data:
  - *Appropriate*: moderate-or-strong evidence (or LaQshya) for **every** required capability
  - *Unsafe*: no evidence at all for at least one required capability, or not a treatment site
  - *Verified blood in window*: a licensed bank of the needed level is in-house, or its blood can
    arrive within the clinical window
  - *Treatment with blood in window*: the later of patient arrival and blood arrival is within the window
  - *Dominated*: another hospital has at least as much evidence and is faster to treatment (5-minute tolerance)

## Headline results (all 1,000 patients)

| | v2 | v1 TOPSIS | Nearest licensed blood | Nearest claiming all |
|---|---|---|---|---|
| **Appropriate destination** | **89.9%** | 43.6% | 37.0% | 40.1% |
| **Unsafe pick** | **1.7%** | 33.7% | 26.7% | 0.0%* |
| Not a treatment site (e.g. standalone blood bank) | **0%** | 21.2% | 0% | 0% |
| **Verified blood within window** | **96.5%** | 73.7% | 92.2% | 84.4% |
| **Treatment with blood within window** | **83.1%** | 71.4% | 69.0% | 81.1% |
| Median minutes to treatment with blood | 45.0 | 44.7 | 45.0 | 45.5 |
| Dominated by another option | **35.4%** | 56.0% | 61.3% | 63.2% |

\*"Nearest claiming all" is never "unsafe" by construction (it only picks places that mention every
capability), but only 40% of its picks have *credible* evidence.

**When v2 recommends** (80.6% of patients): 93.4% appropriate, 0.6% unsafe, 99% with treatment and
blood within the window, median 39.5 minutes. The other methods on the same patients: 40–52% appropriate.

## By area

| | Appropriate | Unsafe | Treatment + blood in window | Median min |
|---|---|---|---|---|
| Urban: v2 | **90.4%** | 1.7% | **95.6%** | 35.2 |
| Urban: v1 | 52.7% | 21.8% | 86.9% | 34.5 |
| Rural: v2 | **89.3%** | 1.7% | **68.4%** | 53.3 |
| Rural: v1 | 32.9% | 47.7% | 53.2% | 48.4 |

In rural areas, v1 sent 37% of patients to a place that is not a treatment site at all.

## By scenario (appropriate / unsafe)

| Scenario | v2 | v1 | Nearest licensed blood | Nearest claiming all |
|---|---|---|---|---|
| Pregnancy bleeding | **69.6% / 6.8%** | 16.4% / 59.6% | 13.4% / 51.6% | 28.9% / 0% |
| Injury bleeding | **93.2% / 0%** | 44.0% / 32.0% | 36.9% / 29.5% | 44.0% / 0% |
| Severe anaemia | **99.6% / 0%** | 66.0% / 11.2% | 57.6% / 0.4% | 40.8% / 0% |
| Not sure | **97.2% / 0%** | 48.0% / 32.0% | 40.6% / 25.1% | 46.8% / 0% |

Pregnancy is the hardest case: credible obstetric-surgery evidence is the scarcest in the data.

## Abstention ("no confident option")

| | Overall | Urban | Rural |
|---|---|---|---|
| v2 abstains | 19.4% | 5.9% | 35.3% |
| Of those, an appropriate in-window option existed (missed) | 22.7% | 15.6% | 24.1% |
| v2 recommended although no appropriate option existed | 2.0% | 2.4% | 1.3% |
| Patients with any appropriate in-window option | 83.4% | 92.8% | 72.3% |

The model rarely recommends when nothing appropriate exists (2%). It is more cautious than the
yardstick: about 1 in 4 abstentions had an option the yardstick would accept. The threshold is a
policy choice; see the sensitivity table.

## Stability (first 200 patients)

| Perturbation | Top pick unchanged |
|---|---|
| Patient moved ~1 km | **97.5%** |
| Road speeds −20% | 81.5% |
| Road speeds +20% | 85.5% |
| Detour factor +20% | 84.5% |

## Sensitivity: one parameter at a time (first 200 patients, base abstention 10.5%)

| Parameter (low / high) | Abstention low → high | Top pick changed (low / high) |
|---|---|---|
| abstain_threshold (0.3 / 0.7) | 6.5% → 17.0% | 2.5% / 0.5% |
| window_high_min (45 / 75 min) | 12.0% → 8.5% | 18.5% / 15.5% |
| speed_urban_kmph (15 / 35) | 11.5% → 10.5% | 26.5% / 19.0% |
| road_detour_factor (1.2 / 1.6) | 10.0% → 11.0% | 4.5% / 17.0% |
| speed_rural_kmph (30 / 50) | 12.5% → 9.0% | 5.0% / 3.0% |
| p_bank_has_stock (0.7 / 0.95) | 11.5% → about base | 3.5% / small |

Every other parameter moved abstention by ≤1 point and top picks by less than ~13%. Travel-time
assumptions (urban speed, detour, window) matter most for *which* hospital wins; the threshold
matters most for *whether* the model recommends.

## Limits

- Simulated patients, not real cases; the urban/rural mix is a design choice.
- The yardstick uses gold evidence levels and median travel times, which are themselves estimates.
  "Appropriate" means credible evidence in the facility's own record, not ground truth.
- "Dominated" ignores blood trust, 24×7 hours and backup banks, which v2 does use, so some v2
  "dominated" picks are deliberate trade-offs. It's still the metric with most room to improve (35%).
- No human ranking judgments yet (the blind 50-query comparison from the plan).
