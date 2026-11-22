"""
Draw the stratified labelling sample and print compact record excerpts for labelling.

Usage:
    python3 -I labels/sample_for_labeling.py <path/to/silver.vf_clean.csv>

Writes labels/label_sample.csv (one row per facility x capability to label) and
labels/label_sample_excerpts.txt (what the labeller reads). The labeller never sees a model score;
the evidence level is used only to stratify the sample and is NOT printed in the excerpts.
"""
import csv, json, os, random, sys, re

csv.field_size_limit(10**9)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPS = ["obstetric_care", "surgical_capacity", "anaesthesia", "emergency_24x7",
        "trauma_surgery", "critical_care", "inpatient_medicine", "blood_service_claim"]
PER_STRATUM = {"strong": 12, "weak_or_moderate": 12, "none": 6}
SEED = 20261122

def stratum(level):
    return "strong" if level == 3 else ("none" if level == 0 else "weak_or_moderate")

def excerpt(s, sv, cap):
    """Record text only. The capability's matching passages are added verbatim (so a mention deep in a long
    list is not missed), but WITHOUT the evidence level, so the labeller judges the text, not the score."""
    def clip(t, n):
        t = re.sub(r"\s+", " ", t or "").strip()
        return t[:n] + ("…" if len(t) > n else "")
    lists = []
    for col in ("capability_raw", "procedure_raw", "equipment_raw", "specialties_raw"):
        try:
            items = json.loads(sv[col]) if sv[col] and sv[col] != "null" else []
        except Exception:
            items = []
        items = [i for i in items if i][:5]
        if items:
            lists.append(f"{col[:-4]}: " + "; ".join(clip(i, 60) for i in items))
    mentions = json.loads(s[f"cap_{cap}_quotes"] or "[]")
    ment = "\n  matching passages: " + " || ".join(f"[{m['field']}] {clip(m['quote'], 150)}" for m in mentions) if mentions else ""
    desc = clip(sv["description_raw"], 350)
    return (f"{s['name']} | type={s['facility_type']} | operator={s['operator']} | {s['city']}, {s['state']}"
            f" | beds={s['beds'] or '?'}\n  description: {desc}\n  " + "\n  ".join(lists) + ment)

def main(silver_path):
    rnd = random.Random(SEED)
    sites = list(csv.DictReader(open(os.path.join(ROOT, "dataset", "gold", "treatment_sites.csv"), encoding="utf-8")))
    silver = {r["facility_id"]: r for r in csv.DictReader(open(silver_path, encoding="utf-8"))}
    rows, used = [], set()
    for cap in CAPS:
        pools = {k: [] for k in PER_STRATUM}
        for s in sites:
            pools[stratum(int(s[f"cap_{cap}_level"]))].append(s)
        for k, n in PER_STRATUM.items():
            pick = [s for s in pools[k] if (s["facility_id"], cap) not in used]
            for s in rnd.sample(pick, min(n, len(pick))):
                used.add((s["facility_id"], cap))
                rows.append(dict(item_id=len(rows) + 1, facility_id=s["facility_id"], capability=cap, stratum=k,
                                 stratum_size=len(pools[k]), label=None, note=None))
    rnd.shuffle(rows)                                   # labeller sees capabilities and strata mixed
    for i, r in enumerate(rows, 1):
        r["item_id"] = i
    by_id = {s["facility_id"]: s for s in sites}
    with open(os.path.join(ROOT, "labels", "label_sample.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(ROOT, "labels", "label_sample_excerpts.txt"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(f"### {r['item_id']} | capability={r['capability']}\n{excerpt(by_id[r['facility_id']], silver[r['facility_id']], r['capability'])}\n\n")
    print(f"{len(rows)} items written")

if __name__ == "__main__":
    main(sys.argv[1])
