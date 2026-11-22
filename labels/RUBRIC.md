# Capability labelling rulebook

Used to label facility x capability claims for training and testing P(capable).
Labels are made by reading the facility's own record (name, type, description, capability,
procedure, equipment, specialties) **without seeing any model score**.

## What is being labelled

For one facility and one capability: **does the record give credible evidence that this facility
itself provides this capability on site?**

| Label | Meaning |
|---|---|
| `1` | The record affirmatively says the facility provides it on site (a service, unit, procedure performed, or equipment it operates), and nothing in the record contradicts that |
| `0` | No such evidence; or the mention is not about on-site service (referrals, a visiting consultant, insurance or directory boilerplate, a doctor's general specialty with no facility service, marketing text about something else); or the facility type makes it implausible (e.g. an eye clinic "surgery" for the surgical-capacity capability) |
| `unclear` | The text could reasonably be read either way. Excluded from training and testing, and counted in the report |

## Capability definitions

| Capability | Counts as 1 | Does not count |
|---|---|---|
| `obstetric_care` | Deliveries, labour room, maternity ward, C-sections, obstetric department or obstetrician-led delivery services | Gynaecology only (IVF, fertility, PCOS, cosmetic gynae) with no deliveries; antenatal scans only |
| `surgical_capacity` | An operating theatre where major surgery is done (general, obstetric, orthopaedic surgery etc.) | Only minor procedures, dental, eye, cosmetic/laser, day-care endoscopy, or a surgeon who consults there with no OT |
| `anaesthesia` | Anaesthesia services or anaesthetists on staff, or surgery under general/spinal anaesthesia | Local anaesthesia for dental/minor procedures only |
| `emergency_24x7` | A 24-hour emergency/casualty department or round-the-clock emergency care | "Open 24 hours" pharmacy or OPD; ambulance only; trauma mentioned only as a specialty word |
| `trauma_surgery` | Trauma care, accident and emergency surgery, orthopaedic or neurosurgical treatment of injuries | Physiotherapy, sports injury rehab, orthopaedic consultation only |
| `critical_care` | ICU, CCU, MICU, SICU, HDU or ventilators operated by the facility | "Critical care" as marketing with no unit; NICU alone (that is neonatal care) |
| `inpatient_medicine` | Admissions/beds/wards with general medicine or physician-led inpatient care | Outpatient clinic only; diagnostic lab only |
| `blood_service_claim` | An in-house blood bank, blood centre, blood storage unit, or transfusions given on site | Blood tests / blood sugar / blood collection for lab tests; "blood donation camp" mentions only |

## Sampling

Stratified, not random, so rare capabilities and weak evidence are well represented:
per capability, about 12 strong-evidence claims, 12 weak/moderate, and 6 with no evidence at all.
Reports reweight to the real mix of evidence levels.

## Known limitation

These labels measure **whether the facility's own record credibly supports the claim**, not
real-world truth. Independent truth comes from the government blood bank directory (blood) and
LaQshya certification (obstetric surgery), which are held out from the text model and used only to
test it.
