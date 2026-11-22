# Facility Selection Criteria: Scored Preferences, Checked Against the Dataset

Source tables: `dataset/blood_points_v2.csv` (3,460 rows, 3,269 in `in_candidate_pool=True`),
`dataset/facility_services_v2.csv` (46,807 rows, covers only the 638 VF-linked facilities, 475
of which remain in the candidate pool).

**This table does not describe hard requirements.** Per `model-design.md`, the only hard gate in
the current design is the time window (drive time within 60 or 120 minutes, depending on
scenario — see bottom of this file). Nothing below removes a facility from the candidate pool.
For facilities with confirmed service data, everything below is scored via the `needed_services`
criterion (a count of scenario-relevant services the facility has), which feeds into entropy
weighting and the severity boost, same as drive time and blood level. A facility *confirmed* to
lack every item in a row scores a real 0 and can still be recommended if nothing better is
reachable in time.

A facility with *no service data at all* is never scored on `needed_services` — see the next
paragraph and `model-design.md`'s "Unknown vs. confirmed-absent" section for the full mechanism.

So read "preferred" below as "counts toward this scenario's `needed_services` score, for
facilities with confirmed service data," and "supplementary" as "would count toward
`needed_services` too, if the service list were extended to include it — currently it doesn't."

**Coverage caveat, now handled in `model-design.md` directly:** service tags (C-section, ICU,
trauma care, CT, 24/7 ED, transfusion, general medicine) exist only for the 475 VF-linked
facilities left in the candidate pool. The other 2,794 candidates (85% of the pool) are
directory-only blood banks with zero rows in `facility_services_v2.csv` — not "checked and
lacking," just never looked up. `model-design.md`'s `recommend()` now splits the reachable pool
into a confirmed tier (ranked on all three criteria, same as before) and an unconfirmed fallback
tier (ranked on drive time and blood level only — `needed_services` dropped from the math
entirely, never guessed at). The fallback tier is only used when nothing confirmed is reachable,
and every result from it is flagged "capability not confirmed — call ahead." So the 85% majority
of the pool is never penalized for a data-coverage gap by being scored as if it had zero
capability — it's simply ranked on what's actually known about it instead.

## Revised table

| Scenario | Preferred (counts toward `needed_services`) | Coverage in candidate pool | Unknown handling | Supplementary (not currently scored; could be added) | Notes |
|---|---|---|---|---|---|
| Pregnant / just delivered, heavy bleeding | `c_section` OR `obstetrics` | `c_section`: 52/475 VF facilities (11%); `obstetrics`: 401/475 (84%). Directory-only banks (2,794): 0 coverage | No tag present → score as unconfirmed/neutral, not 0; flag "obstetric capability not confirmed — call ahead" | `icu`, `nicu` | `obstetrics` is the more reliably tagged signal; `c_section` alone is too sparse to carry much weight on its own |
| Accident / injury, major bleeding | `emergency_surgery` OR `trauma_care` | Either tag: 390/475 (82%); both: 191/475 (40%) | Same as above | `ct_scan`, `icu` | Score on OR, not AND — requiring both roughly halves how many facilities get credit for no clear clinical reason |
| Severe weakness / anemia, no injury (incl. in pregnancy) | Being in the candidate pool at all (i.e. blood on site) | Pool membership: 3,269/3,460 — this is already guaranteed by candidacy, not a separate signal | N/A | `general_medicine` (465/475 VF facilities, 98%) | The literal `transfusion` tag (111/475, 23%) is too rare/under-extracted to weight meaningfully; blood-on-site already is the relevant signal here |
| — fallback behavior, not a scored criterion — | — | — | — | — | Lower-urgency window (120 min) already lets a nearer storage-only facility win when nothing better is reachable — this is handled by the time window + entropy weighting, not an extra rule to add |
| Other blood-related / not sure | `ed_24x7` (VF-tagged) OR `is_24x7` (from `blood_points_v2.csv`, covers the directory-only majority) | `ed_24x7`: 381/475 (80%) VF-only; `is_24x7`: 2,342/3,460 (68%) across the whole pool | No signal either way → unconfirmed/neutral | — (none defined) | Prefer `is_24x7` as the primary signal for this scenario — it's the field that actually has data for most of the pool, `ed_24x7` doesn't |

## What changed from the must-have/nice-to-have version

- Removed all exclusion language ("must have," excludes from pool). Nothing in `model-design.md`
  removes a candidate for lacking a service — only the time window does.
- Renamed columns to "preferred" / "supplementary" to match what the model actually does with
  these fields: add to a count that gets weighted, not gate eligibility.
- Kept the coverage and unknown-handling columns, because they still matter for *scoring
  fairly* — even without a hard gate, scoring a data-coverage gap as "0 services" rather than
  "unconfirmed" will still bias the ranking against the 85% of the pool with no service-tag data,
  just via the score instead of via exclusion.
- Dropped the separate "fallback rule" framing for anemia's storage-only case, since the model
  already produces that behavior through the 120-minute window and entropy weighting — it isn't
  a rule that needs to be added on top.

## For reference: the one actual hard gate in the current model

| Scenario | Urgency | Time window |
|---|---|---|
| Pregnant/just delivered, bleeding | High | 60 min |
| Accident/injury, bleeding | High | 60 min |
| Not sure / other | High | 60 min |
| Severe anemia, no injury | Lower | 120 min |

(`model-design.md`, "Time windows" section.) A facility outside this window is dropped from the
candidate pool regardless of how well it scores on everything else.

## Still open / needs a team decision

- Whether to implement the unconfirmed/neutral scoring for `needed_services` described above, or
  continue defaulting missing service data to 0 — this is a real scoring decision, not yet made
  in `model-design.md` or in code (the pre-processing scripts are still empty placeholders).
- Whether "obstetrics" and "emergency_surgery OR trauma_care" should be the service lists used in
  the `recommend()` function's `SERVICE_COLUMNS`-equivalent config, replacing the narrower
  `c_section`/AND-combined original list.
