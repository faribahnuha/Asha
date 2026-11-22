"""
Fix the two model-input files.

Usage:
    python3 -I fix_model_inputs.py <documents_dir>

Reads (from documents_dir):
    silver.vf_clean.csv                      clean facilities with evidence columns
    facility_services.csv                    original facility x service file (used only for comparison)
    blood_points.csv                         original blood points (Spark CSV: backslash-escaped quotes)
    India Blood Bank Directory (CSV).csv     original directory (to tell real coordinates from filled-in ones)

Writes:
    facility_services_v2.csv     one row per facility x service, each with a verbatim evidence quote
    blood_points_v2.csv          fixed name keys, states, merges, location source, unknown-as-NULL
    blood_points_merge_review.csv  every old and new merge, for hand review

All outputs use standard CSV quoting ("" inside quotes). Load with:
    read_files(path, format=>'csv', header=>true, multiLine=>true, escape=>'"')
"""
import csv, sys, os, re, json, math, collections, difflib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from silver import STATE_ALIAS, key as state_key, norm_phone, km, jlist  # reuse the same rules

csv.field_size_limit(10**9)
D = sys.argv[1] if len(sys.argv) > 1 else "."
P = lambda f: os.path.join(D, f)

def write(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def tf(v):
    return {"true": True, "false": False}.get(str(v).strip().lower())

silver = {r["facility_id"]: r for r in csv.DictReader(open(P("silver.vf_clean.csv"), encoding="utf-8"))}

# =========================================================================== 1. facility_services
# service name in the model  ->  how it is derived from silver evidence columns
SERVICES = {
    # existing services (same names as the original file)
    "c_section":         ["csection"],
    "obstetrics":        ["obstetrics"],
    "ed_24x7":           ["emergency_24x7"],
    "surgery":           ["general_surgery"],
    "emergency_surgery": None,                # derived: ed_24x7 AND (surgery OR operating_theatre)
    "icu":               ["icu"],
    "trauma_care":       ["trauma"],
    "general_medicine":  ["internal_medicine"],
    # added: needed by the 4 transfusion scenarios
    "operating_theatre": ["operating_theatre"],
    "anaesthesia":       ["anaesthesia"],
    "nicu":              ["nicu"],
    "ct_scan":           ["ct_scan"],
    "ventilator":        ["ventilator"],
    "inpatient":         ["inpatient"],
    "haematology":       ["haematology"],
    "orthopaedics":      ["orthopaedics"],
    "neurosurgery":      ["neurosurgery"],
    "blood_bank":        ["blood_bank"],
    "transfusion":       ["transfusion"],
    "blood_components":  ["blood_components"],
}
NON_CLINICAL = {"dentist", "pharmacy"}          # never receive clinical services
STRONG_FIELDS = {"description", "procedure", "capability"}

def strength(ev):
    fields = {e["field"] for e in ev}
    if len(fields) >= 2 and fields & STRONG_FIELDS:
        return "strong"
    if len(fields) >= 2 or fields & {"description", "procedure"}:
        return "moderate"
    return "weak"                                # one list field only, e.g. specialties

fs_rows, skipped = [], collections.Counter()
for fid, s in silver.items():
    if s["exclude_from_ranking"] == "True":
        skipped["excluded_in_silver"] += 1
        continue
    if s["facility_type_std"] in NON_CLINICAL:
        skipped["dentist_or_pharmacy"] += 1
        continue
    got = {}
    for svc, res_list in SERVICES.items():
        if res_list is None:
            continue
        ev = []
        for res in res_list:
            ev += json.loads(s[f"{res}_evidence"] or "[]")
        if ev:
            got[svc] = ev
    if "ed_24x7" in got and ("surgery" in got or "operating_theatre" in got):
        got["emergency_surgery"] = got["ed_24x7"][:1] + (got.get("surgery") or got["operating_theatre"])[:1]
    for svc, ev in got.items():
        fs_rows.append(dict(
            unique_id=fid, service=svc,
            evidence_strength=strength(ev) if svc != "emergency_surgery" else "derived",
            n_fields=len({e["field"] for e in ev}),
            evidence_field=ev[0]["field"], evidence_quote=ev[0]["quote"],
            all_evidence=json.dumps(ev, ensure_ascii=False),
            facility_type=s["facility_type_std"], state=s["state_std"],
            location_confidence=s["location_confidence"], review_color=s["review_color"]))

fs_cols = ["unique_id", "service", "evidence_strength", "n_fields", "evidence_field", "evidence_quote",
           "all_evidence", "facility_type", "state", "location_confidence", "review_color"]
write(P("facility_services_v2.csv"), fs_rows, fs_cols)

# compare with the original file (clean id,service lines only; the 25 text rows are junk)
pair_rx = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}),([a-z_0-9]+)$")
old_lines = open(P("facility_services.csv"), encoding="utf-8-sig").read().split("\n")[1:]
old = {m.groups() for m in (pair_rx.match(l) for l in old_lines) if m}
junk = sum(1 for l in old_lines if l and not pair_rx.match(l))
new = {(r["unique_id"], r["service"]) for r in fs_rows}
print("== facility_services")
print(f"  original: {len(old)} valid pairs + {junk} junk text rows | v2: {len(new)} pairs, {len({r['unique_id'] for r in fs_rows})} facilities")
print(f"  skipped facilities: {dict(skipped)}")
for svc in SERVICES:
    o = {p for p in old if p[1] == svc}; n = {p for p in new if p[1] == svc}
    print(f"  {svc:18s} old={len(o):5d} v2={len(n):5d} kept={len(o & n):5d} dropped(no evidence/non-clinical)={len(o - n):5d} added={len(n - o):5d}")

# =========================================================================== 2. blood_points
GENERIC_BB = set("""blood bank banks centre center hospital hospitals the and of dr govt government general district
medical college research institute trust society charitable voluntary memorial civil sub divisional
multispeciality multi speciality super specialty pvt ltd private limited m s ms""".split())

def name_key(n):                     # lowercase FIRST, then drop non-alphanumerics
    return re.sub(r"[^a-z0-9]", "", (n or "").lower())

CITY_ALIAS = {"bangalore": "bengaluru", "bombay": "mumbai", "gurgaon": "gurugram", "calcutta": "kolkata",
              "madras": "chennai", "trivandrum": "thiruvananthapuram", "mysore": "mysuru", "poona": "pune"}

def city_norm(c):
    c = re.sub(r"[^a-z]", "", (c or "").lower())
    return CITY_ALIAS.get(c, c)

def core_key(n, city=None, state=None):
    """Distinctive part of a name: generic words and the place's own city/state words removed
    ("Fortis Hospital at Ludhiana" -> "fortisat" ... -> "fortis")."""
    place = set(re.findall(r"[a-z]+", f"{city or ''} {state or ''}".lower().replace("-", " ")))
    place |= {CITY_ALIAS.get(p, p) for p in place} | {"at", "in", "cantt", "cant"}
    toks = [t for t in re.findall(r"[a-z0-9]+", (n or "").lower()) if t not in GENERIC_BB and t not in place]
    return "".join(toks)

def std_state(s):
    return STATE_ALIAS.get(state_key((s or "").replace("-", " ")))

rows = list(csv.DictReader(open(P("blood_points.csv"), encoding="utf-8-sig", newline=""),
                           escapechar="\\", doublequote=False))

# original directory, to identify filled-in coordinates
orig = collections.defaultdict(list)
for r in csv.DictReader(open(P("India Blood Bank Directory (CSV).csv"), encoding="utf-8-sig")):
    orig[(name_key(r["name"]), re.sub(r"\D", "", r["pin"]))].append(r)
    orig[(name_key(r["name"]), (r["city"] or "").strip().lower())].append(r)

coord_count = collections.Counter((r["lat"], r["lon"]) for r in rows if r["lat"])
out = []
for r in rows:
    o = dict(r)
    kind = r["place_kind"]
    o["name_key"] = name_key(r["name"])
    o["name_core"] = core_key(r["name"])                              # generic words removed
    o["_core_noplace"] = core_key(r["name"], r["city"], r["state"])   # ...and the place's own city/state words
    o["old_merged_into"] = r["merged_into"]
    lat = float(r["lat"]) if r["lat"] else None
    lon = float(r["lon"]) if r["lon"] else None
    flags = []
    if kind == "facility_with_blood_bank":
        fid = r["place_id"].split("|", 1)[1]
        s = silver.get(fid)
        o["unique_id"] = fid
        o["state_std"] = s["state_std"] if s else std_state(r["state"])
        o["location_source"] = "vf_record"
        o["location_confidence"] = s["location_confidence"] if s else None
        # the source never says whether a facility's own blood bank is 24x7 or does apheresis
        o["is_24x7"] = None
        o["apheresis"] = None
        o["has_components"] = True if (s and s["has_blood_components"] == "True") else None
        ev = []
        if s:
            for res in ("blood_bank", "transfusion", "blood_components"):
                ev += json.loads(s[f"{res}_evidence"] or "[]")
        o["evidence_field"] = ev[0]["field"] if ev else None
        o["evidence_quote"] = ev[0]["quote"] if ev else None
        o["evidence_strength"] = strength(ev) if ev else None
        if not ev:
            flags.append("no_blood_evidence_in_text")
        if s and s["exclude_from_ranking"] == "True":
            flags.append("excluded_in_silver")
        if s and "blood_bank_negation_present" in s["flag_list"] and not ev:
            flags.append("blood_bank_negated")
    else:
        o["unique_id"] = None
        o["state_std"] = std_state(r["state"])
        o["is_24x7"] = tf(r["is_24x7"])
        o["apheresis"] = tf(r["apheresis"])
        o["has_components"] = tf(r["has_components"])
        o["evidence_field"] = "blood_bank_directory"
        o["evidence_quote"] = (f"{r['name']}; components available: {r['has_components']}; "
                               f"apheresis: {r['apheresis'] or 'unknown'}; hours: {'24x7' if r['is_24x7'] == 'true' else (r['is_24x7'] or 'unknown')}")
        o["evidence_strength"] = "directory"
        cands = orig.get((o["name_key"], re.sub(r"\D", "", r["pincode"] or ""))) or orig.get((o["name_key"], (r["city"] or "").strip().lower())) or []
        src = cands[0] if cands else None
        if lat is None:
            o["location_source"] = "missing"
        elif src is None:
            o["location_source"] = "unmatched_to_directory"
        elif not src["lat"]:
            o["location_source"] = "filled_in"           # directory had no coordinates
        elif abs(float(src["lat"]) - lat) < 1e-4 and abs(float(src["lon"]) - lon) < 1e-4:
            o["location_source"] = "directory"
        else:
            o["location_source"] = "changed_from_directory"
        o["location_confidence"] = {"directory": "high", "changed_from_directory": "medium",
                                    "filled_in": "low", "unmatched_to_directory": "medium", "missing": None}[o["location_source"]]
    if not o["state_std"]:
        flags.append("state_unresolved")
    if lat is not None and coord_count[(r["lat"], r["lon"])] >= 3:
        flags.append("shared_coordinates")
        o["location_confidence"] = "low"
    if lat is None:
        flags.append("no_coordinates")
    # phones: one clean list
    raw_ph = jlist(r["phone"]) if r["phone"].strip().startswith("[") else re.split(r"[,/;]", r["phone"] or "")
    phones = []
    for p in raw_ph:
        n, _ = norm_phone(p)
        if n and n not in phones:
            phones.append(n)
    o["phone_std"] = ";".join(phones) or None
    o["_flags"] = flags
    o["_lat"], o["_lon"] = lat, lon
    out.append(o)

# ---- redo merges conservatively
#  A) VF facility <-> directory bank of the same hospital: link them (directory row stays the blood source,
#     VF row leaves the pool) when <= 1 km apart AND core names match (similarity >= 0.85 or one contains the other).
#  B) directory <-> directory duplicates: same name_key AND (same PIN or <= 0.3 km).
#  Generic names (empty core) never merge on name; they need <= 0.1 km and the same name_key.
def sim2(x, y):
    """Best similarity using either the plain core name or the core without city/state words."""
    return max(sim(x["name_core"], y["name_core"]), sim(x["_core_noplace"], y["_core_noplace"]))

def sim(a, b):
    if not a or not b or min(len(a), len(b)) < 4:      # "ram" vs "rama" is not evidence
        return 0.0
    short, long_ = sorted((a, b), key=len)
    # containment counts only if the shorter name is most of the longer one
    # ("mittal" inside "ashokmittal" is a different hospital; "kurjiholyfamily" inside "kurjiholyfamilypatna" is not)
    if len(short) >= 5 and short in long_ and len(short) >= 0.6 * len(long_):
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()

grid = collections.defaultdict(list)
for i, o in enumerate(out):
    if o["_lat"] is not None:
        grid[(round(o["_lat"], 1), round(o["_lon"], 1))].append(i)

def near(i, rad_km):
    o = out[i]
    res = []
    for dx in (-0.1, 0, 0.1):
        for dy in (-0.1, 0, 0.1):
            for j in grid.get((round(o["_lat"] + dx, 1), round(o["_lon"] + dy, 1)), []):
                if j != i and km(o["_lat"], o["_lon"], out[j]["_lat"], out[j]["_lon"]) <= rad_km:
                    res.append(j)
    return res

for o in out:
    o["merged_into"] = None
    o["linked_vf_facility"] = None
    o["merge_reason"] = None

for i, o in enumerate(out):
    if o["place_kind"] != "facility_with_blood_bank" or o["_lat"] is None:
        continue
    best, best_s = None, 0
    for j in near(i, 15.0):
        d = out[j]
        if d["place_kind"] != "directory_blood_bank":
            continue
        s = sim2(o, d)
        dist = km(o["_lat"], o["_lon"], d["_lat"], d["_lon"])
        same_city = city_norm(o["city"]) and city_norm(o["city"]) == city_norm(d["city"])
        # close names within 1 km; near-identical names within 5 km, or within 15 km in the same city
        # (VF and directory geocodes for the same hospital often differ by several km)
        if (s >= 0.85 and dist <= 1.0) or (s >= 0.95 and dist <= 5.0) or (s >= 0.95 and same_city and dist <= 15.0):
            if s > best_s:
                best, best_s = j, s
    if best is not None:
        d = out[best]
        o["merged_into"] = d["place_id"]
        o["merge_reason"] = f"same hospital as directory blood bank (name similarity {best_s:.2f}, {km(o['_lat'], o['_lon'], d['_lat'], d['_lon']):.2f} km)"
        d["linked_vf_facility"] = o["unique_id"] if not d["linked_vf_facility"] else d["linked_vf_facility"]

# same name in the same city but too far apart to merge automatically: usually the same hospital with one
# bad geocode. Not merged; recorded in possible_same_as for a quick human decision.
by_city = collections.defaultdict(list)
for j, d in enumerate(out):
    if d["place_kind"] == "directory_blood_bank":
        by_city[city_norm(d["city"])].append(j)
for o in out:
    o["possible_same_as"] = None
for i, o in enumerate(out):
    if o["place_kind"] != "facility_with_blood_bank" or o["merged_into"]:
        continue
    cands = [(sim2(o, out[j]), j) for j in by_city.get(city_norm(o["city"]), [])]
    cands = [c for c in cands if c[0] >= 0.95]
    if cands:
        s, j = max(cands)
        d = out[j]
        dist = km(o["_lat"], o["_lon"], d["_lat"], d["_lon"]) if o["_lat"] is not None and d["_lat"] is not None else None
        o["possible_same_as"] = d["place_id"] + (f" ({dist:.0f} km apart)" if dist is not None else "")

by_name = collections.defaultdict(list)
for i, o in enumerate(out):
    if o["place_kind"] == "directory_blood_bank":
        by_name[o["name_key"]].append(i)
for nk, idxs in by_name.items():
    if len(idxs) < 2:
        continue
    generic = not out[idxs[0]]["name_core"]
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            x, y = out[idxs[a]], out[idxs[b]]
            if x["merged_into"] or y["merged_into"]:
                continue
            same_pin = x["pincode"] and x["pincode"] == y["pincode"]
            dist = km(x["_lat"], x["_lon"], y["_lat"], y["_lon"]) if x["_lat"] is not None and y["_lat"] is not None else None
            ok = (dist is not None and dist <= 0.1) if generic else (same_pin or (dist is not None and dist <= 0.3))
            if ok:
                keep, drop = (x, y) if (x["_lat"] is not None, x["has_components"] is not None) >= (y["_lat"] is not None, y["has_components"] is not None) else (y, x)
                drop["merged_into"] = keep["place_id"]
                drop["merge_reason"] = "duplicate directory entry (same name, " + ("same PIN" if same_pin else f"{dist:.2f} km") + ")"

idx = {o["place_id"]: o for o in out}
review = []
for o in out:
    if o["old_merged_into"] or o["merged_into"] or o.get("possible_same_as"):
        tgt_old, tgt_new = idx.get(o["old_merged_into"]), idx.get(o["merged_into"])
        review.append(dict(
            place_id=o["place_id"], name=o["name"], city=o["city"],
            old_merged_into=o["old_merged_into"], old_target_name=tgt_old["name"] if tgt_old else None,
            new_merged_into=o["merged_into"], new_target_name=tgt_new["name"] if tgt_new else None,
            decision=("kept" if o["old_merged_into"] and o["old_merged_into"] == o["merged_into"]
                      else "undone" if o["old_merged_into"] and not o["merged_into"]
                      else "changed" if o["old_merged_into"] else "new" if o["merged_into"] else "possible_same"),
            new_reason=o["merge_reason"],
            old_name_similarity=round(sim2(o, tgt_old), 2) if tgt_old else None,
            old_distance_km=(round(km(o["_lat"], o["_lon"], tgt_old["_lat"], tgt_old["_lon"]), 1)
                             if tgt_old and o["_lat"] is not None and tgt_old["_lat"] is not None else None),
            possible_same_as=o.get("possible_same_as"), reviewer_ok=None))

# ---- final columns
for o in out:
    f = o.pop("_flags")
    if o["merged_into"]:
        f.append("merged")
    o["in_candidate_pool"] = not (o["merged_into"] or {"no_blood_evidence_in_text", "excluded_in_silver", "no_coordinates"} & set(f))
    o["flag_list"] = ";".join(sorted(set(f))) or None
    # blood level: 1 = facility's own blood bank (claimed in text, unverified),
    #              2 = licensed directory bank, whole blood only, 3 = directory bank with components
    if o["place_kind"] == "directory_blood_bank":
        o["blood_level"] = 3 if o["has_components"] else 2
    else:
        o["blood_level"] = 1
    o["lat"], o["lon"] = o.pop("_lat"), o.pop("_lon")
    o.pop("_core_noplace", None)

bp_cols = ["place_id", "place_kind", "unique_id", "linked_vf_facility", "name", "name_key", "name_core",
           "address", "city", "district", "state_std", "pincode", "lat", "lon", "location_source",
           "location_confidence", "blood_level", "has_components", "apheresis", "is_24x7", "category",
           "phone_std", "website", "evidence_field", "evidence_quote", "evidence_strength",
           "merged_into", "merge_reason", "possible_same_as", "in_candidate_pool", "flag_list"]
write(P("blood_points_v2.csv"), out, bp_cols)
write(P("blood_points_merge_review.csv"), review,
      ["place_id", "name", "city", "old_merged_into", "old_target_name", "new_merged_into", "new_target_name",
       "decision", "new_reason", "old_name_similarity", "old_distance_km", "possible_same_as", "reviewer_ok"])
print(f"  possible_same_as (same name + city, too far apart to merge automatically): "
      f"{sum(1 for o in out if o.get('possible_same_as'))}")

C = collections.Counter
print("\n== blood_points")
print(f"  rows: {len(out)} | in candidate pool: {sum(o['in_candidate_pool'] for o in out)}")
print(f"  by kind/level: {C((o['place_kind'], o['blood_level']) for o in out if o['in_candidate_pool']).most_common()}")
print(f"  location_source: {C(o['location_source'] for o in out).most_common()}")
print(f"  states: {len({o['state_std'] for o in out if o['state_std']})} official names, unresolved: {sum(1 for o in out if not o['state_std'])}")
print(f"  merges: {C(r['decision'] for r in review).most_common()}")
print(f"  flags: {C(fl for o in out for fl in (o['flag_list'] or '').split(';') if fl).most_common()}")
nk = collections.defaultdict(set)
for o in out:
    nk[o["name_key"]].add(re.sub(r"\s+", " ", o["name"].lower().strip()))
print(f"  name_key collisions across different names: {sum(1 for v in nk.values() if len(v) > 1)}")
