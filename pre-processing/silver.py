"""
Build silver.vf_clean from the bronze VF facility export.

Usage:
    python3 -I build_silver_vf_clean.py <bronze_vf.csv> <blood_bank_directory.csv> <out_dir>

Outputs (in out_dir):
    silver.vf_clean.csv                     one row per unique facility, cleaned + flagged
    silver.vf_clean_change_log.csv          every value changed or record removed, with reason
    silver.vf_clean_manual_review_sample.csv  stratified sample for human verification via source URLs

Principles:
    - Original evidence text (*_raw) is never edited; quotes come from it verbatim.
    - Provably wrong values -> NULL (logged). Doubtful values -> kept + flagged.
    - No values are invented. Nothing is deleted except exact duplicates and
      fragments of split records (both logged).
    - Pure Python standard library, so it runs locally or in a Databricks notebook.
"""
import csv, sys, re, json, math, collections, datetime, hashlib, random, difflib

csv.field_size_limit(10**9)
TODAY = datetime.date(2026, 11, 21)
NULLS = {"", "null", "none", "nan", "na", "n/a", "[]", '[""]', '["",""]', "{}"}

def isnull(v):
    return v is None or str(v).strip().lower() in NULLS

def clean(v):
    return None if isnull(v) else str(v).strip()

def jlist(v):
    """Parse a JSON-array string into a list of non-empty strings (order kept)."""
    if isnull(v):
        return []
    try:
        x = json.loads(v)
        if isinstance(x, list):
            return [str(i).strip() for i in x if i is not None and str(i).strip() != ""]
    except Exception:
        pass
    return [v.strip()]

def dedupe(items, key=lambda s: s.lower()):
    seen, out = set(), []
    for i in items:
        k = key(i)
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out

def to_float(v):
    try:
        return float(v)
    except Exception:
        return None

def to_int(v):
    f = to_float(v)
    return None if f is None else int(round(f))

def km(a1, o1, a2, o2):
    r = math.pi / 180
    x = (o2 - o1) * r * math.cos((a1 + a2) / 2 * r)
    y = (a2 - a1) * r
    return 6371 * math.hypot(x, y)

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# --------------------------------------------------------------------------- states
STATES = ["Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
          "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
          "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
          "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
          "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
          "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"]
key = lambda s: re.sub(r"[^a-z]", "", s.lower())
STATE_ALIAS = {key(s): s for s in STATES}
STATE_ALIAS.update({
    "up": "Uttar Pradesh", "uttarprades": "Uttar Pradesh", "mp": "Madhya Pradesh", "madhya": "Madhya Pradesh",
    "ts": "Telangana", "telengana": "Telangana", "gj": "Gujarat", "mh": "Maharashtra", "br": "Bihar",
    "cg": "Chhattisgarh", "chattisgarh": "Chhattisgarh", "uk": "Uttarakhand", "uttranchal": "Uttarakhand",
    "uttaranchal": "Uttarakhand", "dl": "Delhi", "nct": "Delhi", "nctofdelhi": "Delhi", "nctdelhi": "Delhi",
    "newdelhi": "Delhi", "delhincr": "Delhi", "ncrdelhi": "Delhi", "westdelhi": "Delhi", "eastdelhi": "Delhi",
    "southdelhi": "Delhi", "northwestdelhi": "Delhi", "southeastdelhiarea": "Delhi", "orissa": "Odisha",
    "tamilnadu": "Tamil Nadu", "pondicherry": "Puducherry", "utofpuducherry": "Puducherry",
    "jammukashmir": "Jammu and Kashmir", "jammuandkashmir": "Jammu and Kashmir", "kashmir": "Jammu and Kashmir",
    "jammujk": "Jammu and Kashmir", "srinagarkashmir": "Jammu and Kashmir", "punjabregion": "Punjab",
    "chandigarhut": "Chandigarh", "damananddiu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadranagarhavelidamandiu": "Dadra and Nagar Haveli and Daman and Diu",
})
STATE_RX = [(s, re.compile(r"\b" + re.escape(s.lower()).replace(r"\ ", r"\s*") + r"\b")) for s in STATES]
STATE_RX += [("Jammu and Kashmir", re.compile(r"jammu\s*(&|and)\s*kashmir|j&k")),
             ("Uttar Pradesh", re.compile(r"uttar\s*pradesh")), ("Delhi", re.compile(r"\bdelhi\b"))]

def state_from_field(raw):
    """Return (state or None, how) from the free-text state field."""
    if raw is None:
        return None, "missing"
    k = key(raw)
    if k in STATE_ALIAS:
        return STATE_ALIAS[k], "alias"
    hits = {s for s, rx in STATE_RX if rx.search(raw.lower())}
    if len(hits) == 1:
        return hits.pop(), "contains"
    if len(hits) > 1:
        return None, "multiple_states"
    return None, "not_a_state"

def states_from_pin(pin):
    """Possible states for a PIN, from India Post's prefix allocation (approximate at borders)."""
    if not pin:
        return set()
    p2, p3 = int(pin[:2]), int(pin[:3])
    if p3 == 160: return {"Chandigarh", "Punjab", "Haryana"}
    if p3 == 194: return {"Ladakh", "Jammu and Kashmir"}
    if p3 == 403: return {"Goa"}
    if p3 == 396: return {"Gujarat", "Dadra and Nagar Haveli and Daman and Diu"}
    if p3 in (605, 609): return {"Puducherry", "Tamil Nadu"}
    if p3 == 682: return {"Kerala", "Lakshadweep"}
    if p3 == 737: return {"Sikkim"}
    if p3 == 744: return {"Andaman and Nicobar Islands"}
    if p3 in (248, 249, 263): return {"Uttarakhand"}
    if p3 in (244, 246, 247, 262): return {"Uttar Pradesh", "Uttarakhand"}
    if 790 <= p3 <= 792: return {"Arunachal Pradesh"}
    if 793 <= p3 <= 794: return {"Meghalaya"}
    if p3 == 795: return {"Manipur"}
    if p3 == 796: return {"Mizoram"}
    if 797 <= p3 <= 798: return {"Nagaland"}
    if p3 == 799: return {"Tripura"}
    if 800 <= p3 <= 812 or 817 <= p3 <= 821 or 836 <= p3 <= 855: return {"Bihar", "Jharkhand"} if p3 in (812, 821, 836) else {"Bihar"}
    if 813 <= p3 <= 816 or 822 <= p3 <= 835: return {"Jharkhand"}
    m = {11: {"Delhi"}, 12: {"Haryana"}, 13: {"Haryana"}, 14: {"Punjab"}, 15: {"Punjab"}, 16: {"Punjab"},
         17: {"Himachal Pradesh"}, 18: {"Jammu and Kashmir"}, 19: {"Jammu and Kashmir"},
         30: {"Rajasthan"}, 31: {"Rajasthan"}, 32: {"Rajasthan"}, 33: {"Rajasthan"}, 34: {"Rajasthan"},
         36: {"Gujarat"}, 37: {"Gujarat"}, 38: {"Gujarat"}, 39: {"Gujarat"},
         40: {"Maharashtra"}, 41: {"Maharashtra"}, 42: {"Maharashtra"}, 43: {"Maharashtra"}, 44: {"Maharashtra"},
         45: {"Madhya Pradesh"}, 46: {"Madhya Pradesh"}, 47: {"Madhya Pradesh"}, 48: {"Madhya Pradesh"},
         49: {"Chhattisgarh"}, 50: {"Telangana"}, 51: {"Andhra Pradesh"}, 52: {"Andhra Pradesh"},
         53: {"Andhra Pradesh"}, 56: {"Karnataka"}, 57: {"Karnataka"}, 58: {"Karnataka"}, 59: {"Karnataka"},
         60: {"Tamil Nadu"}, 61: {"Tamil Nadu"}, 62: {"Tamil Nadu"}, 63: {"Tamil Nadu"}, 64: {"Tamil Nadu"},
         67: {"Kerala"}, 68: {"Kerala"}, 69: {"Kerala"}, 70: {"West Bengal"}, 71: {"West Bengal"},
         72: {"West Bengal"}, 73: {"West Bengal"}, 74: {"West Bengal"}, 75: {"Odisha"}, 76: {"Odisha"},
         77: {"Odisha"}, 78: {"Assam"}}
    if 20 <= p2 <= 28:
        return {"Uttar Pradesh"}
    return m.get(p2, set())

# --------------------------------------------------------------------------- vocabularies
GENERIC = set("""hospital hospitals clinic clinics centre center centres centers care health healthcare
multispeciality multispecialty multi speciality specialty super superspeciality the and of dr doctor doctors
medical nursing home pvt ltd private limited institute sciences science research diagnostic diagnostics lab
labs laboratory dental eye children women womens maternity polyclinic and memorial trust society city new
general services service unit india""".split())

DIRECTORY_DOMAINS = ["practo.com", "justdial.com", "business.site", "facebook.com", "instagram.com",
    "linkedin.com", "lybrate.com", "medindia.net", "onefivenine.com", "policyx.com", "hexahealth.com",
    "mappls.com", "docindia.org", "sulekha.com", "indiamart.com", "google.com", "goo.gl", "blogspot.",
    "healthinsuranceindia.org", "zurichkotak.com", "proptiger.com", "latlong.net", "quikr.com",
    "credihealth.com", "clinicspots.com", "drlogy.com", "bajajfinservhealth.in", "youtube.com", "x.com",
    "twitter.com", "wa.me", "whatsapp.com", "magicpin.in", "asklaila.com", "yellowpages", "joonsquare.com",
    "doctoriduniya.com", "kayawell.com", "orbitoasia.com", "docthub.com", "mediflix", "skedoc.com"]
FREE_EMAIL = {"gmail.com", "yahoo.com", "yahoo.co.in", "yahoo.in", "rediffmail.com", "hotmail.com",
              "outlook.com", "live.com", "icloud.com", "ymail.com", "aol.com", "protonmail.com"}

# Resource vocabulary for the 4 transfusion scenarios. Matched on raw text (case-insensitive);
# specialty camelCase tokens are split into words first.
RESOURCES = {
    "obstetrics":        r"obstetric|maternity|labou?r (room|ward|suite)|normal deliver|child ?birth|high[- ]risk pregnan|antenatal|delivery suite|caesarean|cesarean|c-?section|\blscs\b",
    "gynaecology":       r"gyn(a)?ecolog",
    "csection":          r"c-?section|caesarean|cesarean|\blscs\b|lower segment",
    "operating_theatre": r"operation theat(re|er)|operating (theat(re|er)|room)|modular ot\b|\bot complex|surgical suite|\boperation theat",
    "anaesthesia":       r"an(a)?esthe(sia|siology|siologist|tist|tic)",
    "emergency_24x7":    r"24 ?[x×/\*] ?7|24 ?hours? (emergency|casualty|service)|round[- ]the[- ]clock|emergency (department|services?|care|room|ward|medicine)|casualty",
    "trauma":            r"trauma|polytrauma|accident (and|&) emergency|road traffic accident",
    "general_surgery":   r"general surg|laparoscop",
    "orthopaedics":      r"orthop(a)?edic|fracture",
    "neurosurgery":      r"neuro ?surg",
    "ct_scan":           r"\bct\b|ct[- ]scan|computed tomography|\bcat scan",
    "icu":               r"\bicu\b|intensive care|critical care|\bccu\b|\bmicu\b|\bsicu\b",
    "nicu":              r"\bnicu\b|neonatal intensive|neonatolog|\bsncu\b|special newborn care",
    "ventilator":        r"ventilator|mechanical ventilation",
    "internal_medicine": r"internal medicine|general medicine|general physician",
    "haematology":       r"ha?ematolog",
    "inpatient":         r"in-?patient|\bipd\b|\bbeds?\b|\bwards?\b|admission",
    "blood_bank":        r"blood ?bank|blood cent(er|re)|blood storage|\bbsu\b",
    "transfusion":       r"transfusion",
    "blood_components":  r"component separation|blood components?|\bprbc\b|packed (red )?cells?|\bffp\b|fresh frozen plasma|platelet (transfusion|concentrate)|cryoprecipitate|apheresis",
    "dialysis":          r"dialysis",
    "mri":               r"\bmri\b|magnetic resonance",
    "xray":              r"x-?ray|radiograph",
    "ultrasound":        r"ultrasound|ultrasonograph|sonograph|\busg\b",
}
RES_RX = {k: re.compile(v, re.I) for k, v in RESOURCES.items()}
EQUIPMENT_RESOURCES = ["ct_scan", "mri", "xray", "ultrasound", "ventilator", "dialysis"]
NEG_RX = re.compile(r"\b(no|not|without|lacks?|lacking|unavailable|refer(red|s)? (to|for|out)|does not|don't|doesn't)\b[^.;]{0,40}$", re.I)
TEXT_FIELDS = ["description", "capability", "procedure", "equipment", "specialties"]

def camel_words(s):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)

# --------------------------------------------------------------------------- load bronze
def load_bronze(path):
    rdr = csv.reader(open(path, encoding="utf-8-sig", newline=""))
    hdr = next(rdr)
    return hdr, list(rdr)

def classify(hdr, rows):
    src = hdr.index("source")
    heads, tails, texts, complete = [], [], [], []
    for i, r in enumerate(rows):
        if UUID.match(r[0]):
            (heads if r[src] == "null" else complete).append(i)
        elif all(x == "null" for x in r[1:]):
            texts.append(i)
        else:
            tails.append(i)
    return heads, tails, texts, complete

SPEC_RX = re.compile(r'^\[\s*"[a-z][A-Za-z]+"')

def align_tail(hdr, t):
    """Map a shifted tail fragment back onto the columns after `description`."""
    di, si = hdr.index("description"), hdr.index("specialties")
    pos = next((i for i in range(1, len(t)) if SPEC_RX.match(t[i])), None)
    if pos is None:
        return None
    off = si - pos
    fields, desc_parts = {}, [t[0]]
    for j in range(1, len(t)):
        k = j + off
        if k <= di:
            if t[j] != "null":
                desc_parts.append(t[j])
        elif k < len(hdr):
            fields[hdr[k]] = t[j]
    lat, lon = to_float(fields.get("latitude")), to_float(fields.get("longitude"))
    ok = (lat is not None and lon is not None and 6 <= lat <= 37.5 and 68 <= lon <= 97.5
          and str(fields.get("source_urls", "")).startswith('["http'))
    return {"fields": fields, "desc_tail": ",".join(desc_parts), "valid": ok}

def name_tokens(name):
    return {t for t in re.findall(r"[a-z]{3,}", (name or "").lower()) if t not in GENERIC}

def repair_split_records(hdr, rows, heads, tails, log, grid):
    """Attach each truncated facility to a tail that matches its name AND its own stated location.

    The head row keeps its address, city, state and PIN (they come before `description`),
    so a tail is accepted only if its coordinates fall in the head's state. Pairs are
    assigned greedily, best score first, one tail per head.
    """
    aligned = {i: align_tail(hdr, rows[i]) for i in tails}
    H = {c: hdr.index(c) for c in ("name", "unique_id", "address_stateOrRegion", "address_zipOrPostcode", "address_city")}
    cands = []
    for h in heads:
        r = rows[h]
        ht = name_tokens(r[H["name"]])
        st, _ = state_from_field(clean(r[H["address_stateOrRegion"]]))
        pin = re.sub(r"\s", "", r[H["address_zipOrPostcode"]])
        pst = states_from_pin(pin) if re.fullmatch(r"[1-9]\d{5}", pin) else set()
        head_states = ({st} if st else set()) | pst
        city = re.sub(r"[^a-z]", "", r[H["address_city"]].lower())
        for t, a in aligned.items():
            if not a or not a["valid"]:
                continue
            blob = re.sub(r"[^a-z]", "", " ".join(rows[t]).lower())
            score = sum(1 for tok in ht if tok in blob)
            need = min(2, max(1, math.ceil(len(ht) / 2)))
            if score < need:
                continue
            lat, lon = float(a["fields"]["latitude"]), float(a["fields"]["longitude"])
            cst, share, _ = state_from_coords(grid, lat, lon)
            geo_ok = bool(head_states) and cst in head_states and share >= 3 / 7
            if not geo_ok and head_states:
                geo_ok = any(nearest_state_km(grid, lat, lon, s) < 40 for s in head_states)
            if not geo_ok:
                continue
            bonus = 1 if city and len(city) >= 4 and city in blob else 0
            cands.append((score + bonus, h, t, score, len(ht)))
    repaired, used_h, used_t = {}, set(), set()
    for tot, h, t, sc, n in sorted(cands, reverse=True):
        if h in used_h or t in used_t:
            continue
        used_h.add(h); used_t.add(t)
        repaired[h] = (t, aligned[t], sc, n)
        log.append(dict(facility_id=rows[h][H["unique_id"]], field="(record)", old_value="truncated_head",
                        new_value=f"repaired from bronze record #{t + 2}", rule_id="R00_split_repair",
                        reason=f"tail matched {sc}/{n} distinctive name tokens; its coordinates fall in the head's stated state"))
    return repaired, aligned

# --------------------------------------------------------------------------- blood-bank kNN (state-from-coordinates proxy)
def load_bloodbank_points(path):
    pts = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig", newline="")):
        a, o = to_float(r.get("lat")), to_float(r.get("lon"))
        if a is None or o is None or not (6 <= a <= 37.5 and 68 <= o <= 97.5):
            continue
        st = STATE_ALIAS.get(key(r["state"].replace("-", " ")))
        if st:
            pts.append((a, o, st))
    grid = collections.defaultdict(list)
    for p in pts:
        grid[(int(p[0]), int(p[1]))].append(p)
    return grid

def state_from_coords(grid, lat, lon, k=7):
    cand, rad = [], 1
    while len(cand) < k and rad <= 6:
        cand = [p for dx in range(-rad, rad + 1) for dy in range(-rad, rad + 1) for p in grid.get((int(lat) + dx, int(lon) + dy), [])]
        rad += 1
    if not cand:
        return None, 0.0, None
    cand.sort(key=lambda p: km(lat, lon, p[0], p[1]))
    near = cand[:k]
    votes = collections.Counter(p[2] for p in near)
    st, n = votes.most_common(1)[0]
    return st, n / len(near), km(lat, lon, near[0][0], near[0][1])

def nearest_state_km(grid, lat, lon, state, rad=4):
    """Distance to the nearest blood bank listed in `state` (inf if none within ~rad degrees)."""
    best = float("inf")
    for dx in range(-rad, rad + 1):
        for dy in range(-rad, rad + 1):
            for p in grid.get((int(lat) + dx, int(lon) + dy), []):
                if p[2] == state:
                    best = min(best, km(lat, lon, p[0], p[1]))
    return best

# --------------------------------------------------------------------------- phones / web / email
def norm_phone(p):
    d = re.sub(r"\D", "", p or "")
    if d.startswith("91") and len(d) in (12, 14):
        d = d[2:]
    if d.startswith("0") and len(d) == 11:
        d = d[1:]
    if re.fullmatch(r"1800\d{6,7}", d):
        return "+91" + d, "tollfree"
    if len(d) == 10 and d[0] in "6789":
        return "+91" + d, "mobile"
    if len(d) == 10 and d[0] in "12345":
        return "+91" + d, "landline"
    return None, "invalid"

def domain_of(u):
    if isnull(u):
        return None
    u = u.strip().lower()
    if u in ("true", "false"):
        return None
    u = re.sub(r"^[a-z]+://", "", u)
    d = re.split(r"[/?#]", u)[0]
    d = re.sub(r"^www\d?\.", "", d).split(":")[0]
    return d if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", d) else None

def is_directory(d):
    return d is not None and any(x in d for x in DIRECTORY_DOMAINS)

EMAIL_RX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

# --------------------------------------------------------------------------- main build
def main(vf_path, bb_path, out_dir):
    hdr, rows = load_bronze(vf_path)
    log = []
    heads, tails, texts, complete = classify(hdr, rows)
    grid = load_bloodbank_points(bb_path)
    repaired, _ = repair_split_records(hdr, rows, heads, tails, log, grid)
    idc = hdr.index("unique_id")

    for i in tails + texts:
        log.append(dict(facility_id=None, field="(record)", old_value=rows[i][0][:80], new_value="removed",
                        rule_id="R01_fragment", reason=f"bronze record #{i + 2} is a fragment of a split record, not a facility"))

    # assemble one dict per facility record (complete + heads, with repairs applied)
    recs = []
    for i in sorted(complete + heads):
        d = dict(zip(hdr, rows[i]))
        d["_bronze_row"] = i + 2
        d["_split_status"] = "complete"
        if i in heads:
            if i in repaired:
                t, a, _, _ = repaired[i]
                for k, v in a["fields"].items():
                    d[k] = v
                d["description"] = d["description"].rstrip("\n") + "\n" + a["desc_tail"]
                d["_split_status"] = "repaired"
                d["_tail_row"] = t + 2
            else:
                d["_split_status"] = "truncated_unrepaired"
        recs.append(d)

    # exact duplicates (same unique_id, identical values)
    seen, uniq = {}, []
    for d in recs:
        sig = hashlib.md5(json.dumps([d[c] for c in hdr]).encode()).hexdigest()
        if (d["unique_id"], sig) in seen:
            log.append(dict(facility_id=d["unique_id"], field="(record)", old_value=f"bronze record #{d['_bronze_row']}",
                            new_value="removed", rule_id="R02_exact_duplicate",
                            reason=f"identical to bronze record #{seen[(d['unique_id'], sig)]}"))
            continue
        seen[(d["unique_id"], sig)] = d["_bronze_row"]
        uniq.append(d)

    out = []
    def chg(fid, field, old, new, rule, reason):
        log.append(dict(facility_id=fid, field=field, old_value=old, new_value=new, rule_id=rule, reason=reason))

    for d in uniq:
        fid = d["unique_id"]
        o = collections.OrderedDict()
        flags = []
        o["facility_id"] = fid
        o["cluster_id"] = clean(d["cluster_id"])
        o["bronze_row"] = d["_bronze_row"]
        o["split_status"] = d["_split_status"]
        o["source_types_list"] = json.dumps(dedupe(jlist(d["source_types"])))
        o["n_source_ids"] = len(set(jlist(d["source_ids"])))

        # ---- name
        name = clean(d["name"])
        o["name_raw"] = d["name"]
        o["name_norm"] = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())).strip() or None
        toks = name_tokens(name)
        o["name_distinctive_tokens"] = json.dumps(sorted(toks))
        if not name or len(name) < 3 or re.fullmatch(r"[\d\s+()-]+", name) or re.search(r"https?://|@", name):
            flags.append("name_invalid")
        elif not toks:
            flags.append("name_generic")
        if name and (len(name) > 80 or re.search(r"\b[1-9]\d{5}\b|\bfloor\b|\bcomplex\b|\broad\b", name, re.I)):
            flags.append("name_has_address_text")
        words = re.findall(r"[a-z0-9]+", (name or "").lower())
        acronym = "".join(w[0] for w in words if w not in ("the", "and", "of", "dr", "government", "govt", "sri", "shri", "new"))
        compact = "".join(words)

        # ---- facility / operator type
        ft = (clean(d["facilityTypeId"]) or "unknown").lower()
        ft_std = {"farmacy": "pharmacy"}.get(ft, ft)
        if ft_std != ft:
            chg(fid, "facilityTypeId", ft, ft_std, "R10_type_typo", "spelling fixed")
        o["facility_type_std"] = ft_std
        ln = (name or "").lower()
        if (re.search(r"pharmacy|chemist|medical store|druggist", ln) and ft_std != "pharmacy") or \
           (re.search(r"\bhospital\b", ln) and ft_std in ("pharmacy", "doctor")) or \
           (re.search(r"dental|dentist", ln) and ft_std not in ("dentist", "hospital", "clinic")):
            flags.append("facility_type_conflict")
        op = (clean(d["operatorTypeId"]) or "unknown").lower()
        o["operator_std"] = "public" if op in ("public", "government") else op
        affil = dedupe(jlist(d["affiliationTypeIds"]))
        o["affiliation_list"] = json.dumps(affil)
        govt_name = re.search(r"\b(government|govt|civil hospital|district hospital|sadar|aiims|esic|esi hospital|primary health cent|community health cent|\bphc\b|\bchc\b|sub[- ]?district hospital)", ln)
        # affiliation "government" on private hospitals usually means empanelment (CGHS/PM-JAY),
        # not ownership, so only the name triggers a conflict.
        if o["operator_std"] == "private" and govt_name:
            flags.append("operator_conflict")

        # ---- contact: phones
        plist = dedupe(jlist(d["phone_numbers"]), key=lambda s: re.sub(r"\D", "", s))
        normed = [norm_phone(p) for p in plist]
        o["phone_list_std"] = json.dumps(dedupe([p for p, k in normed if p], key=lambda s: s))
        o["n_phones"] = len(json.loads(o["phone_list_std"]))
        o["n_invalid_phones"] = sum(1 for p, k in normed if not p)
        op_std, op_kind = norm_phone(clean(d["officialPhone"]) or "")
        o["official_phone_raw"] = d["officialPhone"]
        o["official_phone_std"] = op_std
        o["official_phone_kind"] = op_kind if clean(d["officialPhone"]) else None
        if clean(d["officialPhone"]) and not op_std:
            flags.append("official_phone_invalid")
        if o["n_phones"] >= 4:
            flags.append("many_phones")

        # ---- email
        e_raw = clean(d["email"])
        o["email_raw"] = d["email"]
        emails = []
        if e_raw and "protected" in e_raw.lower():
            chg(fid, "email", e_raw, None, "R20_email_obfuscated", "scraper captured an email-protection placeholder, not an address")
        elif e_raw:
            emails = [x for x in re.split(r"[,;\s]+", e_raw) if x]
        valid_emails = [x.lower() for x in emails if EMAIL_RX.match(x)]
        if emails and not valid_emails:
            flags.append("email_invalid")
        o["email_list_std"] = json.dumps(dedupe(valid_emails))
        e_dom = valid_emails[0].split("@")[1] if valid_emails else None
        o["email_domain"] = e_dom
        o["email_kind"] = None if not e_dom else ("free_provider" if e_dom in FREE_EMAIL else "organisation")

        # ---- websites
        ow = clean(d["officialWebsite"])
        o["official_website_raw"] = d["officialWebsite"]
        w_dom = domain_of(ow)
        if ow and not w_dom:
            flags.append("website_invalid")
        o["official_website_domain"] = w_dom
        o["website_kind"] = None if not w_dom else ("directory_or_social" if is_directory(w_dom) else "own_domain")
        o["website_list_domains"] = json.dumps(dedupe([x for x in (domain_of(u) for u in jlist(d["websites"])) if x]))
        o["website_reachable"] = "not_checked_offline"
        if w_dom and o["website_kind"] == "own_domain":
            wd = re.sub(r"[^a-z]", "", w_dom.split(".")[0])
            def fuzzy_in(t):
                if t in wd or (len(wd) >= 4 and wd in t):
                    return True
                n = len(t)
                return n >= 5 and any(difflib.SequenceMatcher(None, t, wd[i:i + n]).ratio() >= 0.8 for i in range(0, max(1, len(wd) - n + 2)))
            acr_hit = len(acronym) >= 3 and (wd.startswith(acronym[:3]) or acronym[:4] in wd)
            wd_full = re.sub(r"[^a-z0-9]", "", w_dom.split(".")[0])
            acr_hit = acr_hit or (len(wd_full) >= 4 and (wd_full in compact or compact.startswith(wd_full[:5])))
            o["website_matches_name"] = (any(fuzzy_in(t) for t in toks) or acr_hit) if (toks or acronym) else None
            if o["website_matches_name"] is False:
                flags.append("website_name_mismatch")
            if e_dom and o["email_kind"] == "organisation":
                o["email_matches_website"] = (e_dom == w_dom or e_dom.endswith("." + w_dom) or w_dom.endswith("." + e_dom))
                if not o["email_matches_website"]:
                    flags.append("email_website_mismatch")
            else:
                o["email_matches_website"] = None
        else:
            o["website_matches_name"] = None
            o["email_matches_website"] = None

        # ---- source URLs (offline association check)
        urls = dedupe(jlist(d["source_urls"]), key=lambda s: s)
        o["source_url_list"] = json.dumps(urls)
        o["n_source_urls"] = len(urls)
        doms = [domain_of(u) for u in urls]
        o["source_domains"] = json.dumps(dedupe([x for x in doms if x]))
        o["share_directory_sources"] = round(sum(1 for x in doms if is_directory(x)) / len(urls), 2) if urls else None
        o["has_official_site_source"] = bool(w_dom and any(x and (x == w_dom or x.endswith("." + w_dom)) for x in doms))
        city_tok = re.sub(r"[^a-z]", "", (clean(d["address_city"]) or "").lower())
        informative = [u for u in urls if re.search(r"[a-z]{4,}", re.sub(r"^https?://[^/]+", "", u.lower()))]
        def url_mentions(u):
            s = re.sub(r"[^a-z]", "", u.lower())
            stems = {t[:-1] if t.endswith("s") and len(t) >= 5 else t for t in toks} | {t[:6] for t in toks if len(t) >= 8}
            return any(t in s for t in stems) or (len(acronym) >= 3 and acronym in s)
        o["n_informative_source_urls"] = len(informative)
        o["source_url_name_match_share"] = round(sum(1 for u in informative if url_mentions(u)) / len(informative), 2) if informative and toks else None
        o["source_url_city_match"] = any(city_tok and len(city_tok) >= 4 and city_tok in re.sub(r"[^a-z]", "", u.lower()) for u in urls) if urls else None
        if informative and toks and o["source_url_name_match_share"] == 0 and not o["source_url_city_match"]:
            flags.append("sources_unrelated_to_name")
        o["source_url_verification"] = "not_checked_offline"

        # ---- address / city
        lines = [clean(d[c]) for c in ("address_line1", "address_line2", "address_line3")]
        o["address_raw"] = ", ".join(x for x in lines if x) or None
        o["address_norm"] = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (o["address_raw"] or "").lower())).strip() or None
        if not o["address_raw"]:
            flags.append("address_missing")
        city = clean(d["address_city"])
        o["city_raw"] = d["address_city"]
        CITY_ALIAS = {"bangalore": "Bengaluru", "bombay": "Mumbai", "gurgaon": "Gurugram", "trivandrum": "Thiruvananthapuram",
                      "calcutta": "Kolkata", "madras": "Chennai", "mysore": "Mysuru", "poona": "Pune", "baroda": "Vadodara"}
        o["city_std"] = CITY_ALIAS.get((city or "").lower(), city.title() if city else None)
        if not city:
            flags.append("city_missing")

        # ---- PIN
        pin_raw = clean(d["address_zipOrPostcode"])
        o["pin_raw"] = d["address_zipOrPostcode"]
        pin = re.sub(r"\s", "", pin_raw or "")
        pin = pin if re.fullmatch(r"[1-9]\d{5}", pin) else None
        if pin and pin != pin_raw:
            chg(fid, "address_zipOrPostcode", pin_raw, pin, "R30_pin_format", "removed spaces")
        if pin_raw and not pin:
            chg(fid, "address_zipOrPostcode", pin_raw, None, "R31_pin_invalid", "not a valid 6-digit Indian PIN")
            flags.append("pin_invalid")
        if not pin_raw:
            flags.append("pin_missing")
        o["pin_std"] = pin
        addr_pins = set(re.findall(r"\b[1-9]\d{2}\s?\d{3}\b", o["address_raw"] or ""))
        addr_pins = {p.replace(" ", "") for p in addr_pins}
        if pin and addr_pins and pin not in addr_pins:
            flags.append("address_pin_conflict")

        # ---- coordinates
        lat, lon = to_float(d["latitude"]), to_float(d["longitude"])
        o["lat_raw"], o["lon_raw"] = d["latitude"], d["longitude"]
        if lat is not None and lon is not None and not (6 <= lat <= 37.5 and 68 <= lon <= 97.5):
            if 6 <= lon <= 37.5 and 68 <= lat <= 97.5:
                chg(fid, "latitude/longitude", f"{lat},{lon}", f"{lon},{lat}", "R40_coords_swapped", "latitude and longitude were swapped")
                lat, lon = lon, lat
            else:
                chg(fid, "latitude/longitude", f"{lat},{lon}", None, "R41_coords_outside_india", "point lies outside India")
                flags.append("coords_outside_india")
                lat = lon = None
        if lat is None or lon is None:
            lat = lon = None
            if "coords_outside_india" not in flags:
                flags.append("coords_missing")
        o["lat"], o["lon"] = lat, lon
        if lat is not None and ((lat * 64).is_integer() or (lon * 64).is_integer()):
            flags.append("low_precision_coords")
        try:
            cj = json.loads(d["coordinates"])["coordinates"]
            if lat is not None and (abs(cj[1] - lat) > 1e-4 or abs(cj[0] - lon) > 1e-4) and "coords_swapped" not in flags:
                flags.append("coordinates_field_conflict")
        except Exception:
            pass

        # ---- state (field -> PIN -> coordinates)
        s_raw = clean(d["address_stateOrRegion"])
        o["state_raw"] = d["address_stateOrRegion"]
        f_state, how = state_from_field(s_raw)
        pin_states = states_from_pin(pin)
        c_state, c_share, c_dist = (state_from_coords(grid, lat, lon) if lat is not None else (None, 0.0, None))
        o["state_from_pin"] = json.dumps(sorted(pin_states))
        o["state_from_coords_proxy"] = c_state if c_share >= 5 / 7 else None
        if f_state:
            o["state_std"], o["state_source"] = f_state, "field_" + how
        elif len(pin_states) == 1:
            o["state_std"], o["state_source"] = next(iter(pin_states)), "pin_prefix"
        elif o["state_from_coords_proxy"]:
            o["state_std"], o["state_source"] = o["state_from_coords_proxy"], "coords_proxy"
        else:
            o["state_std"], o["state_source"] = None, None
        if how in ("not_a_state", "multiple_states") and s_raw:
            flags.append("state_field_not_a_state")
        if o["state_std"] and s_raw and o["state_std"] != s_raw:
            chg(fid, "address_stateOrRegion", s_raw, o["state_std"], "R50_state_std", f"standardised via {o['state_source']}")
        if o["state_std"] and pin_states and o["state_std"] not in pin_states:
            flags.append("pin_state_mismatch")
        # strict: all 7 nearest blood banks are in another state AND none of the stated state within 40 km
        # (if the PIN agrees with the stated state, only flag when the point is > 100 km from that state:
        #  near-border points are usually right and the proxy is wrong)
        if o["state_std"] and lat is not None and c_state and c_state != o["state_std"] and c_share == 1.0:
            d_state = nearest_state_km(grid, lat, lon, o["state_std"])
            if d_state > (100 if o["state_std"] in pin_states else 40):
                flags.append("coords_state_mismatch")
        if not o["state_std"]:
            flags.append("state_unresolved")

        # ---- evidence text (raw, never edited) + parsed lists
        o["description_raw"] = d["description"]
        for c in ("capability", "procedure", "equipment", "specialties"):
            o[f"{c}_raw"] = d[c]
        spec = dedupe(jlist(d["specialties"]))
        o["specialties_list"] = json.dumps(spec)
        o["n_specialties"] = len(spec)
        for c in ("capability", "procedure", "equipment"):
            lst = dedupe(jlist(d[c]))
            o[f"{c}_list"] = json.dumps(lst)
            o[f"n_{c}"] = len(lst)
        if len(spec) > 30:
            flags.append("too_many_specialties")
        desc = clean(d["description"]) or ""
        if len(desc) < 40:
            flags.append("description_short")
        if d["_split_status"] != "complete":
            flags.append("description_incomplete")

        # ---- resource evidence (verbatim quotes with field of origin)
        field_items = {
            "description": [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", d["description"] or "") if s.strip()],
            "capability": jlist(d["capability"]), "procedure": jlist(d["procedure"]),
            "equipment": jlist(d["equipment"]), "specialties": spec,
        }
        for res, rx in RES_RX.items():
            ev, negated = [], False
            for fld, items in field_items.items():
                for it in items:
                    probe = camel_words(it) if fld == "specialties" else it
                    m = rx.search(probe)
                    if not m:
                        continue
                    if NEG_RX.search(probe[:m.start()]):
                        negated = True
                        continue
                    ev.append({"field": fld, "quote": it[:300]})
            ev = dedupe(ev, key=lambda e: (e["field"], e["quote"].lower()))
            o[f"has_{res}"] = bool(ev)
            o[f"{res}_n_fields"] = len({e["field"] for e in ev})
            o[f"{res}_evidence"] = json.dumps(ev[:5], ensure_ascii=False)
            if negated:
                flags.append(f"{res}_negation_present")
        eq_text = " ".join(field_items["equipment"]).lower()
        o["equipment_mentioned_elsewhere"] = json.dumps(
            [r for r in EQUIPMENT_RESOURCES if o[f"has_{r}"] and not RES_RX[r].search(eq_text)])

        # ---- scale and history
        for src_c, dst_c, hi, rule in (("numberDoctors", "num_doctors", 5000, "R60"), ("capacity", "capacity_beds", 6000, "R61")):
            v = to_int(d[src_c])
            o[f"{dst_c}_raw"] = d[src_c]
            if v is not None and v <= 0:
                chg(fid, src_c, d[src_c], None, rule + "_zero", "0 or negative used as a missing value")
                v = None
            elif v is not None and v > hi:
                chg(fid, src_c, d[src_c], None, rule + "_impossible", f"exceeds {hi}, larger than any Indian hospital")
                flags.append("implausible_scale")
                v = None
            o[dst_c] = v
        if o["num_doctors"] and o["facility_type_std"] in ("clinic", "dentist", "doctor") and o["num_doctors"] > 100:
            flags.append("implausible_scale")
        if o["num_doctors"] and o["capacity_beds"] and o["num_doctors"] > 10 * o["capacity_beds"] + 50:
            flags.append("implausible_scale")
        # infrastructure-heavy claims from facility types that cannot host them
        if ft_std in ("dentist", "pharmacy", "doctor") and any(o[f"has_{r}"] for r in ("blood_bank", "blood_components", "icu", "nicu", "ventilator")):
            flags.append("implausible_capability")
        y = to_int(d["yearEstablished"])
        o["year_established_raw"] = d["yearEstablished"]
        if y is not None and (y < 1800 or y > TODAY.year):
            chg(fid, "yearEstablished", d["yearEstablished"], None, "R62_year_invalid", "outside 1800..current year")
            y = None
        o["year_established"] = y
        o["scale_missing"] = o["num_doctors"] is None and o["capacity_beds"] is None and y is None

        # ---- recency
        rp = clean(d["recency_of_page_update"])
        try:
            rd = datetime.date.fromisoformat(rp[:10]) if rp else None
        except Exception:
            rd = None
        if rd and rd > TODAY:
            chg(fid, "recency_of_page_update", rp, None, "R63_future_date", "date is in the future")
            flags.append("recency_future_date")
            rd = None
        o["page_last_updated"] = rd.isoformat() if rd else None
        o["page_age_days"] = (TODAY - rd).days if rd else None

        # ---- web presence (signals that the place exists; not evidence of care)
        o["n_social_platforms"] = to_int(d["distinct_social_media_presence_count"])
        o["has_staff_listed"] = clean(d["affiliated_staff_presence"]) == "true"
        o["has_custom_logo"] = clean(d["custom_logo_presence"]) == "true"
        o["n_facts"] = to_int(d["number_of_facts_about_the_organization"])
        o["last_social_post_date"] = clean(d["post_metrics_most_recent_social_media_post_date"])
        o["n_followers"] = to_int(d["engagement_metrics_n_followers"])

        o["_flags"] = flags
        out.append(o)

    # ---- shared official phone (one number on many unrelated facilities)
    by_phone = collections.defaultdict(set)
    for o in out:
        if o["official_phone_std"]:
            by_phone[o["official_phone_std"]].add(o["name_norm"])
    for o in out:
        n = len(by_phone.get(o["official_phone_std"], ()))
        o["official_phone_shared_by"] = n if o["official_phone_std"] else None
        if n >= 3:
            o["_flags"].append("shared_phone")

    # ---- internal location consistency: distance from the median point of facilities
    #      in the same city, and in the same 3-digit PIN area (needs >= 5 facilities).
    #      In Databricks, replace the PIN check with the India Post PIN centroid table.
    def medians(keyf):
        acc = collections.defaultdict(list)
        for o in out:
            k = keyf(o)
            if k and o["lat"] is not None:
                acc[k].append((o["lat"], o["lon"]))
        res = {}
        for k, v in acc.items():
            if len(v) >= 5:
                la = sorted(p[0] for p in v)[len(v) // 2]
                lo = sorted(p[1] for p in v)[len(v) // 2]
                res[k] = (la, lo)
        return res
    city_med = medians(lambda o: (o["city_std"], o["state_std"]) if o["city_std"] else None)
    pin_med = medians(lambda o: o["pin_std"][:3] if o["pin_std"] else None)
    for o in out:
        cm = city_med.get((o["city_std"], o["state_std"]))
        pm = pin_med.get(o["pin_std"][:3]) if o["pin_std"] else None
        o["km_from_city_median"] = round(km(o["lat"], o["lon"], *cm), 1) if cm and o["lat"] is not None else None
        o["km_from_pin_area_median"] = round(km(o["lat"], o["lon"], *pm), 1) if pm and o["lat"] is not None else None
        if o["km_from_city_median"] is not None and o["km_from_city_median"] > 50:
            o["_flags"].append("far_from_city")
        if o["km_from_pin_area_median"] is not None and o["km_from_pin_area_median"] > 80:
            o["_flags"].append("far_from_pin_area")

    # ---- probable duplicates: same name + state, same city or PIN, and
    #      (< 1 km apart, same PIN, overlapping address words, or same non-shared phone)
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x
    groups = collections.defaultdict(list)
    for idx, o in enumerate(out):
        if o["name_norm"]:
            groups[(o["name_norm"], o["state_std"])].append(idx)
    for g in groups.values():
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                x, y = out[g[a]], out[g[b]]
                same_place = (x["city_std"] and x["city_std"] == y["city_std"]) or (x["pin_std"] and x["pin_std"] == y["pin_std"])
                if not same_place:
                    continue
                aw = set(re.findall(r"[a-z0-9]{3,}", x["address_norm"] or "")) - GENERIC
                bw = set(re.findall(r"[a-z0-9]{3,}", y["address_norm"] or "")) - GENERIC
                addr_overlap = len(aw & bw) / max(1, len(aw | bw)) >= 0.4
                near = x["lat"] is not None and y["lat"] is not None and km(x["lat"], x["lon"], y["lat"], y["lon"]) < 1.0
                same_pin = x["pin_std"] and x["pin_std"] == y["pin_std"]
                same_ph = x["official_phone_std"] and x["official_phone_std"] == y["official_phone_std"] and "shared_phone" not in x["_flags"]
                if addr_overlap or near or same_pin or same_ph:
                    parent[find(g[a])] = find(g[b])
    members = collections.defaultdict(list)
    for idx in range(len(out)):
        if idx in parent:
            members[find(idx)].append(idx)

    def completeness(o):
        keys = [o["name_raw"], o["lat"], o["pin_std"], o["official_phone_std"], len(o["description_raw"] or "") >= 40,
                o["n_specialties"], o["n_capability"], o["n_procedure"], o["n_equipment"], o["num_doctors"],
                o["capacity_beds"], o["year_established"]]
        return round(sum(1 for k in keys if k) / len(keys), 2)

    for o in out:
        o["completeness_score"] = completeness(o)
        o["dup_group_id"] = None
        o["is_canonical"] = True
    for root, idxs in members.items():
        if len(idxs) < 2:
            continue
        gid = "dup_" + out[root]["facility_id"][:8]
        # keep the record whose location is internally consistent, then the most complete one
        best = max(idxs, key=lambda i: ("far_from_city" not in out[i]["_flags"] and "far_from_pin_area" not in out[i]["_flags"],
                                        out[i]["completeness_score"], out[i]["n_source_urls"]))
        for i in idxs:
            out[i]["dup_group_id"] = gid
            out[i]["is_canonical"] = (i == best)
            out[i]["_flags"].append("duplicate")
            if i != best:
                chg(out[i]["facility_id"], "(record)", "active", "excluded_from_ranking", "R70_probable_duplicate",
                    f"same name and state as {out[best]['facility_id']} with same address, phone or <200 m; kept the more complete record")

    # ---- final status columns
    for o in out:
        f = o.pop("_flags")
        no_location = o["lat"] is None and o["pin_std"] is None
        no_services = not (o["n_specialties"] or o["n_capability"] or o["n_procedure"]) and len(o["description_raw"] or "") < 40
        unusable = ("name_invalid" in f) or no_location or no_services or o["split_status"] == "truncated_unrepaired"
        verify_later = {"website_name_mismatch", "email_website_mismatch", "sources_unrelated_to_name", "pin_state_mismatch",
                        "coords_state_mismatch", "facility_type_conflict", "operator_conflict", "implausible_scale",
                        "implausible_capability", "shared_phone", "address_pin_conflict", "state_unresolved",
                        "far_from_city", "far_from_pin_area", "name_has_address_text"}
        o["flag_unusable"] = unusable
        o["unusable_reason"] = ";".join(r for r, c in (("name_invalid", "name_invalid" in f), ("no_location", no_location),
                                        ("no_service_info", no_services), ("truncated_record", o["split_status"] == "truncated_unrepaired")) if c) or None
        o["flag_duplicate"] = "duplicate" in f
        o["needs_manual_verification"] = bool(verify_later & set(f))
        o["review_color"] = "orange" if unusable else ("purple" if o["flag_duplicate"] else ("yellow" if o["needs_manual_verification"] else None))
        o["exclude_from_ranking"] = unusable or not o["is_canonical"]
        o["location_confidence"] = ("low" if o["lat"] is None or {"coords_state_mismatch", "low_precision_coords", "pin_state_mismatch", "far_from_city", "far_from_pin_area"} & set(f)
                                    else "medium" if {"pin_missing", "pin_invalid", "address_pin_conflict", "state_field_not_a_state"} & set(f)
                                    else "high")
        o["flag_list"] = json.dumps(sorted(set(f)))
        o["flag_count"] = len(set(f))
        o["silver_built_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    # ---- write
    cols = list(out[0].keys())
    with open(f"{out_dir}/silver.vf_clean.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    with open(f"{out_dir}/silver.vf_clean_change_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["facility_id", "field", "old_value", "new_value", "rule_id", "reason"])
        w.writeheader()
        w.writerows(log)

    # stratified manual-review sample: orange / purple / yellow / clean, plus all blood/C-section claims sample
    random.seed(42)
    strata = collections.defaultdict(list)
    for o in out:
        strata[o["review_color"] or "clean"].append(o)
    sample = []
    for s, n in (("orange", 25), ("purple", 25), ("yellow", 50), ("clean", 50)):
        sample += random.sample(strata[s], min(n, len(strata[s])))
    sc = ["facility_id", "name_raw", "city_std", "state_std", "pin_std", "lat", "lon", "official_website_raw",
          "official_phone_std", "email_list_std", "source_url_list", "review_color", "flag_list"]
    with open(f"{out_dir}/silver.vf_clean_manual_review_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sc + ["reviewer", "name_ok", "location_ok", "phone_ok", "website_ok", "services_ok", "notes"])
        w.writeheader()
        for o in sample:
            w.writerow({k: o[k] for k in sc})

    # summary
    c = collections.Counter()
    for o in out:
        for fl in json.loads(o["flag_list"]):
            c[fl] += 1
    print("bronze records:", len(rows), "| facilities in silver:", len(out),
          "| repaired:", sum(o["split_status"] == "repaired" for o in out),
          "| truncated unrepaired:", sum(o["split_status"] == "truncated_unrepaired" for o in out),
          "| fragments removed:", len(tails) + len(texts), "| exact duplicates removed:", len(recs) - len(uniq))
    print("review_color:", collections.Counter(o["review_color"] for o in out))
    print("excluded from ranking:", sum(o["exclude_from_ranking"] for o in out))
    print("changes logged:", len(log))
    print("flags:", c.most_common())

if __name__ == "__main__":
    main(*sys.argv[1:4])
