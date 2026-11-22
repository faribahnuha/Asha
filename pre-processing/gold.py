"""
Build the gold tables for the chance-of-success model.

Usage:
    python3 -I pre-processing/gold.py <path/to/silver.vf_clean.csv>

Reads:
    <silver.vf_clean.csv>                       cleaned VF facilities (coordinates, raw text, evidence, flags)
    dataset/blood_points_v2.csv                 blood sources (directory blood banks + hospital claims)
    dataset/external/laqshya_2025-07-31.csv     LaQshya certified labour rooms / maternity OTs (pre-processing/laqshya.py)
Writes (dataset/gold/):
    treatment_sites.csv      one row per possible destination hospital, combined capabilities with evidence
    blood_sources.csv        one row per blood source with blood level, hours, trust and location quality
    site_blood_links.csv     hospital <-> its own directory blood bank (re-matched against ALL hospitals,
                             by name + same PIN/city, never by distance)
    laqshya_matches.csv      LaQshya rows matched to hospitals, for hand checking
    gold_build_report.json   counts and checks from this run

Design decisions (see the gold section of the model design doc):
    * Single services are combined into scenario-level capabilities (C-section alone is 5.5% covered).
    * Dropped as too thin or not varying: year established, page last updated, number of source URLs.
    * Evidence level uses the FULL field count from silver (fixes the 5-quote cap undercount).
    * Unknown stays unknown: beds/doctors get a *_known flag; is_24x7 keeps "unknown".
    * No distances are stored. Gold keeps coordinates only; the model computes every distance
      (patient -> hospital, bank -> hospital, local density) at query time.
"""
import csv, sys, os, re, json, math, collections, difflib, datetime

csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TODAY = datetime.date(2026, 11, 22)
OUT = os.path.join(ROOT, "dataset", "gold")
STRONG_FIELDS = {"description", "procedure", "capability"}
NON_CLINICAL = {"dentist", "pharmacy"}

# ----------------------------------------------------------------------------- combined capabilities
# capability -> silver resource columns that count as evidence for it (any one is enough)
CAPABILITIES = {
    "obstetric_care":      ["obstetrics", "csection"],
    "surgical_capacity":   ["operating_theatre", "general_surgery", "csection"],
    "anaesthesia":         ["anaesthesia"],
    "emergency_24x7":      ["emergency_24x7", "trauma"],
    "trauma_surgery":      ["trauma", "orthopaedics", "neurosurgery", "general_surgery"],
    "critical_care":       ["icu", "ventilator"],
    "neonatal_care":       ["nicu"],
    "inpatient_medicine":  ["internal_medicine", "inpatient", "haematology"],
    "ct_imaging":          ["ct_scan"],
    "blood_service_claim": ["blood_bank", "transfusion", "blood_components"],
}
LEVEL_NAME = {0: "none", 1: "weak", 2: "moderate", 3: "strong"}

def evidence_level(n_fields, quotes):
    """3 strong: >=2 fields incl. description/procedure/capability; 2 moderate: >=2 fields, or one field
    that is description/procedure; 1 weak: one list field only; 0 none. n_fields is the FULL count."""
    if n_fields <= 0:
        return 0
    fields = {q["field"] for q in quotes}
    if n_fields >= 2 and (fields & STRONG_FIELDS):
        return 3
    if n_fields >= 2 or (fields & {"description", "procedure"}):
        return 2
    return 1

def capability_columns(s):
    """Combined capability columns for one silver row."""
    out = {}
    for cap, members in CAPABILITIES.items():
        best_lvl, best_nf, quotes, present = 0, 0, [], []
        for res in members:
            nf = int(s[f"{res}_n_fields"] or 0)
            q = json.loads(s[f"{res}_evidence"] or "[]")
            lvl = evidence_level(nf, q)
            if lvl:
                present.append(res)
                quotes += [dict(x, service=res) for x in q]
            if (lvl, nf) > (best_lvl, best_nf):
                best_lvl, best_nf = lvl, nf
        quotes.sort(key=lambda x: (x["field"] not in STRONG_FIELDS, x["field"]))
        out[f"cap_{cap}_level"] = best_lvl
        out[f"cap_{cap}_n_fields"] = best_nf
        out[f"cap_{cap}_services"] = "|".join(present) or None
        out[f"cap_{cap}_quotes"] = json.dumps(quotes[:3], ensure_ascii=False) if quotes else None
    return out

# ----------------------------------------------------------------------------- name matching helpers
GENERIC = set("""blood bank banks centre center hospital hospitals the and of dr govt government general district
medical college research institute trust society charitable voluntary memorial civil sub divisional area
multispeciality multi speciality super specialty superspeciality pvt ltd private limited m s ms care health
healthcare clinic nursing home sciences""".split())
PUBLIC_WORDS = re.compile(r"\b(civil|district|sadar|general|government|govt|area hospital|sub[- ]?district|taluk|chc|community health|dh|sdh)\b", re.I)
INSTITUTION = re.compile(PUBLIC_WORDS.pattern[:-3] + r"|medical college)\b", re.I)
CHAIN = re.compile(r"\b(apollo|fortis|max|manipal|narayana|medanta|columbia asia|yashoda|aster|amri|medica super|"
                   r"care hospitals?|kims hospitals?|rainbow|motherhood|cloudnine|sparsh|alchemist|unity|wockhardt|"
                   r"global hospitals?|asian institute|sunrise)\b", re.I)
CITY_ALIAS = {"bangalore": "bengaluru", "bombay": "mumbai", "gurgaon": "gurugram", "calcutta": "kolkata",
              "madras": "chennai", "trivandrum": "thiruvananthapuram", "mysore": "mysuru", "poona": "pune"}

def city_norm(c):
    c = re.sub(r"[^a-z]", "", (c or "").lower())
    return CITY_ALIAS.get(c, c)

def tokens(n, place=()):
    return [t for t in re.findall(r"[a-z0-9]+", (n or "").lower()) if t not in GENERIC and t not in place]

def core(n, place=()):
    return "".join(tokens(n, place))

def sim(a, b):
    if not a or not b or min(len(a), len(b)) < 4:
        return 0.0
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 5 and short in long_ and len(short) >= 0.6 * len(long_):
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def place_words(city, state):
    p = set(re.findall(r"[a-z]+", f"{city or ''} {state or ''}".lower().replace("-", " ")))
    return p | {CITY_ALIAS.get(x, x) for x in p} | {"at", "in", "cantt"}


# ----------------------------------------------------------------------------- builders
def build_sites(silver):
    sites = []
    for fid, s in silver.items():
        if s["exclude_from_ranking"] == "True" or not s["lat"] or s["facility_type_std"] in NON_CLINICAL:
            continue
        shared = int(s["official_phone_shared_by"] or 0)
        row = dict(
            facility_id=fid, name=s["name_raw"], address=s["address_raw"], city=s["city_std"], state=s["state_std"], pin=s["pin_std"],
            lat=float(s["lat"]), lon=float(s["lon"]), location_confidence=s["location_confidence"],
            facility_type=s["facility_type_std"], operator=s["operator_std"],
            phone=s["official_phone_std"] if s["official_phone_std"] and shared < 3 else None,
            beds=int(s["capacity_beds"]) if s["capacity_beds"] else None, beds_known=bool(s["capacity_beds"]),
            doctors=int(s["num_doctors"]) if s["num_doctors"] else None, doctors_known=bool(s["num_doctors"]),
            has_official_site_source=s["has_official_site_source"] == "True",
            share_directory_sources=float(s["share_directory_sources"]) if s["share_directory_sources"] else None,
            completeness_score=float(s["completeness_score"]), flag_count=int(s["flag_count"]),
            needs_manual_verification=s["needs_manual_verification"] == "True", flag_list=s["flag_list"],
        )
        row.update(capability_columns(s))
        sites.append(row)
    return sites

LAQ_GENERIC = {"women", "womens", "child", "children", "maternity", "combined", "taluka", "taluk", "fru", "esic",
               "sadar", "rural", "satellite", "mother", "zonal", "headquarters", "base", "referral", "esi",
               "dch", "dwh", "ch", "uphc", "phc", "mch", "gh", "ah", "govt", "hospiral", "and", "for"}
DH_WORDS = re.compile(r"\b(district|civil|sadar|sub[- ]?district|taluk|taluka|area hospital|headquarters|"
                      r"government|govt|state general|zonal|regional|community health|chc)\b", re.I)
NOT_ALLOPATHIC = re.compile(r"homoeo|homeo|ayurved|unani|siddha|dental|eye|netralaya", re.I)

def match_laqshya(sites, laq_rows):
    """Public-facility match. Same state AND the same place (LaQshya district or a place word in its name must
    appear in the hospital's city or name), AND distinctive name words agree. Names that are only a place
    (e.g. 'DH Durg') also need a public-hospital word in the VF name. Generic words like 'women', 'taluka',
    'ESIC' never count as distinctive."""
    by_state = collections.defaultdict(list)
    for s in sites:
        if s["operator"] != "private" and not NOT_ALLOPATHIC.search(s["name"] or ""):
            by_state[(s["state"] or "").lower()].append(s)
    matches = []
    place_stop = {"nagar", "north", "south", "east", "west", "city", "rural", "urban", "dehat", "new", "old",
                  "district", "hospital", "government", "govt", "medical", "college", "civil", "general", "and"}
    words3 = lambda text: {CITY_ALIAS.get(w, w) for w in re.findall(r"[a-z]{3,}", (text or "").lower())} - place_stop
    for r in laq_rows:
        place = place_words(r["district"], r["state"])
        rt_all = {t for t in tokens(r["facility_name"]) if len(t) >= 3} - LAQ_GENERIC
        district_words = words3(r["district"])
        laq_name_words = words3(r["facility_name"])
        for s in by_state.get(r["state"].lower(), []):
            st_all = {t for t in tokens(s["name"]) if len(t) >= 3} - LAQ_GENERIC
            vf_city, vf_name = words3(s["city"]), words3(s["name"])
            # same place: the LaQshya district appears in the hospital's city/name, or the hospital's city
            # appears in the LaQshya facility name ("Sadar Hospital Jamshedpur")
            if not ((district_words & (vf_city | vf_name)) or (vf_city & laq_name_words)):
                continue
            if not st_all and not rt_all:
                continue
            rt, st_d = rt_all - place, st_all - place_words(s["city"], s["state"])
            if rt and st_d:     # both have distinctive words: they must agree
                ok = rt <= st_all or st_d <= rt_all
                rule = "distinctive_name_same_place"
            elif not rt:        # LaQshya name is only a place/type ("DH Durg"): the hospital TYPE must agree
                if r["facility_type"] == "MC":
                    ok = bool(re.search(r"medical college", s["name"], re.I))
                else:
                    ok = bool(DH_WORDS.search(s["name"])) and not re.search(r"medical college|veterinary", s["name"], re.I)
                rule = "place_name_public_word"
            else:
                ok, rule = False, None
            if ok:
                matches.append(dict(facility_id=s["facility_id"], vf_name=s["name"], vf_city=s["city"],
                                    laqshya_name=r["facility_name"], laqshya_district=r["district"],
                                    laqshya_type=r["facility_type"], department=r["department"], status=r["status"],
                                    certified_on=r["certified_on"], valid_until=r["valid_until"],
                                    is_current=r["valid_until"] >= TODAY.isoformat(), match_rule=rule, reviewer_ok=None))
    return matches

def apply_laqshya(sites, matches):
    by = collections.defaultdict(list)
    for m in matches:
        by[m["facility_id"]].append(m)
    for s in sites:
        ms = by.get(s["facility_id"], [])
        mot = [m for m in ms if m["department"] == "Maternity OT"]
        lr = [m for m in ms if m["department"] == "Labour Room"]
        s["laqshya_mot_current"] = any(m["is_current"] for m in mot)
        s["laqshya_mot_ever"] = bool(mot)
        s["laqshya_lr_current"] = any(m["is_current"] for m in lr)
        s["laqshya_lr_ever"] = bool(lr)

def build_blood_sources(blood_rows):
    out = []
    for b in blood_rows:
        if b["in_candidate_pool"] != "True" or not b["lat"]:
            continue
        directory = b["place_kind"] == "directory_blood_bank"
        out.append(dict(
            source_id=b["place_id"], kind=b["place_kind"], name=b["name"], city=b["city"], state=b["state_std"],
            pin=re.sub(r"\D", "", b["pincode"] or "") or None,
            lat=float(b["lat"]), lon=float(b["lon"]), blood_level=int(b["blood_level"]),
            has_components={"True": True, "False": False}.get(b["has_components"]),
            apheresis={"True": True, "False": False}.get(b["apheresis"]),
            is_24x7={"True": "yes", "False": "no"}.get(b["is_24x7"], "unknown"),
            source_trust="licensed_directory" if directory else "hospital_text_claim",
            category=b["category"] or None, location_source=b["location_source"],
            location_confidence=b["location_confidence"], phone=b["phone_std"] or None,
            vf_facility_id=b["unique_id"] or None, evidence_field=b["evidence_field"], evidence_quote=b["evidence_quote"]))
    return out

def link_sites_to_banks(sites, sources):
    """Re-match every hospital (not just the 638 that claimed a blood bank) to a directory bank.
    Match by NAME and PLACE only (same PIN or same city), never by distance, so a bad coordinate in either
    dataset can neither create nor block a link."""
    dirs = [b for b in sources if b["kind"] == "directory_blood_bank"]
    by_state = collections.defaultdict(list)
    for b in dirs:
        by_state[(b["state"] or "").lower()].append(b)
    links = {}
    for s in sites:
        p_s = place_words(s["city"], s["state"])
        cs = core(s["name"], p_s)                        # distinctive part, place words removed
        best = None
        for b in by_state.get((s["state"] or "").lower(), []):
            same_pin = bool(s["pin"]) and s["pin"] == b["pin"]
            same_city = bool(city_norm(s["city"])) and city_norm(s["city"]) == city_norm(b["city"])
            if not (same_pin or same_city):
                continue
            cb = core(b["name"], place_words(b["city"], b["state"]))
            # compare distinctive parts only, so a shared city name can't make two hospitals look alike
            sm = sim(cs, cb) if (cs and cb) else 0.0
            rule = None
            if sm >= 0.85 and same_pin:
                rule = "name>=0.85_same_pin"
            elif sm >= 0.90 and same_city:
                rule = "name>=0.90_same_city"
            elif not cs and not cb and same_pin and INSTITUTION.search(s["name"]) and INSTITUTION.search(b["name"]):
                # both names generic ("District Hospital X"): same PIN, and any place words in the names must agree
                ps, pb = set(tokens(s["name"])), set(tokens(b["name"]))
                if not ps or not pb or ps & pb:
                    rule, sm = "generic_public_name_same_pin", 0.0
            if rule and (best is None or (b["blood_level"], sm) > (best["blood_level"], best["name_similarity"])):
                # without distance, a chain brand in a big city can match the wrong branch: mark it low
                conf = "high" if same_pin else ("low" if CHAIN.search(b["name"]) else "medium")
                best = dict(facility_id=s["facility_id"], source_id=b["source_id"], source_name=b["name"],
                            blood_level=b["blood_level"], is_24x7=b["is_24x7"], name_similarity=round(sm, 2),
                            same_pin=same_pin, same_city=same_city, link_rule=rule, link_confidence=conf)
        if best:
            links[s["facility_id"]] = best
    return links

def apply_in_house(sites, links):
    """In-house blood as plain facts on each site. Distances to other banks are NOT stored: the model computes
    them from coordinates at query time."""
    for s in sites:
        link = links.get(s["facility_id"])
        s["in_house_directory_level"] = link["blood_level"] if link else 0
        s["in_house_source_id"] = link["source_id"] if link else None
        s["in_house_is_24x7"] = link["is_24x7"] if link else None
        s["in_house_link_confidence"] = link["link_confidence"] if link else None
        s["in_house_text_claim_level"] = s["cap_blood_service_claim_level"]

def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    cols = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

# ----------------------------------------------------------------------------- main
def main(silver_path):
    silver = {r["facility_id"]: r for r in csv.DictReader(open(silver_path, encoding="utf-8"))}
    blood_rows = list(csv.DictReader(open(os.path.join(ROOT, "dataset", "blood_points_v2.csv"), encoding="utf-8")))
    laq = list(csv.DictReader(open(os.path.join(ROOT, "dataset", "external", "laqshya_2025-07-31.csv"), encoding="utf-8")))

    sites = build_sites(silver)
    laq_matches = match_laqshya(sites, laq)
    apply_laqshya(sites, laq_matches)
    sources = build_blood_sources(blood_rows)
    links = link_sites_to_banks(sites, sources)
    apply_in_house(sites, links)

    write("treatment_sites.csv", sites)
    write("blood_sources.csv", sources)
    write("site_blood_links.csv", list(links.values()))
    write("laqshya_matches.csv", laq_matches)
    stale = os.path.join(OUT, "site_blood_access.csv")      # distance table from the earlier design
    if os.path.exists(stale):
        os.remove(stale)

    # ---- checks
    raw = lambda f: " ".join(silver[f][c] for c in ("description_raw", "capability_raw", "procedure_raw", "equipment_raw", "specialties_raw"))
    bad_quotes = n_quotes = 0
    for s in sites:
        r = raw(s["facility_id"])
        for cap in CAPABILITIES:
            for q in json.loads(s[f"cap_{cap}_quotes"] or "[]"):
                n_quotes += 1
                if q["quote"] not in r and json.dumps(q["quote"])[1:-1] not in r:
                    bad_quotes += 1
    old_links = {r["linked_vf_facility"] for r in blood_rows if r["linked_vf_facility"]}
    C = collections.Counter
    report = dict(
        built_at=datetime.datetime.now().isoformat(timespec="seconds"),
        treatment_sites=len(sites), blood_sources=len(sources),
        blood_sources_by_level=dict(C(b["blood_level"] for b in sources)),
        capability_coverage={cap: dict(C(LEVEL_NAME[s[f"cap_{cap}_level"]] for s in sites)) for cap in CAPABILITIES},
        evidence_quotes_checked=n_quotes, evidence_quotes_not_verbatim=bad_quotes,
        directory_links_before=len(old_links), directory_links_after=len(links),
        directory_links_by_rule=dict(C(l["link_rule"] for l in links.values())),
        directory_links_by_confidence=dict(C(l["link_confidence"] for l in links.values())),
        laqshya_rows=len(laq), laqshya_matches=len(laq_matches),
        laqshya_matched_sites=len({m["facility_id"] for m in laq_matches}),
        laqshya_mot_current_sites=sum(s["laqshya_mot_current"] for s in sites),
        sites_with_in_house_directory_blood=sum(1 for s in sites if s["in_house_directory_level"] > 0),
        dropped_features=["year_established", "page_last_updated", "n_source_urls"],
        not_precomputed="all distances (hospital-bank, density, nearest bank): computed from coordinates at query time",
    )
    json.dump(report, open(os.path.join(OUT, "gold_build_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    assert bad_quotes == 0, "every evidence quote must appear verbatim in the facility's raw text"
    assert len({s["facility_id"] for s in sites}) == len(sites), "one row per facility"

if __name__ == "__main__":
    main(sys.argv[1])
