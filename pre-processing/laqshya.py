"""
Parse the NHSRC LaQshya certification list (labour rooms and maternity operating theatres).

Usage:
    python3 -I pre-processing/laqshya.py dataset/external/LaQshya_LR-MOT_2025-07-31.pdf \
        dataset/external/laqshya_2025-07-31.csv

Source: https://qps.nhsrcindia.org/sites/default/files/2025-08/LaQshya(LR-MOT)-%2031st%20July%2025.pdf
(public list, data as of 31 July 2025). Needs `pdftotext` (poppler) on the PATH.

Why it matters: LaQshya certifies PUBLIC labour rooms and maternity operating theatres against a
national quality standard. It is the only source we have that confirms obstetric surgery
independently of scraped web text, and it covers exactly the public hospitals whose web listings
rarely mention C-sections.

Limits: public facilities only; certificates expire (we keep both dates); facility names in the
PDF are merged across rows, so names are re-attached heuristically and every row records how
its name was found (`name_source`).
"""
import csv, re, subprocess, sys, datetime

DEPTS = ("Labour Room", "Maternity OT")
ROW_START = re.compile(r"^\s*\d+\s+\S")

def split_row(line):
    """Split a table row on runs of 2+ spaces. Returns None if the line is not a data row.
    Layout: S.NO | State | District | Type | [Name] | Department | Yes | cert date | valid date."""
    if not ROW_START.match(line) or not any(d in line for d in DEPTS):
        return None
    toks = re.split(r"\s{2,}", line.strip())
    m = re.match(r"^(\d+)\s+(.+)$", toks[0])          # S.NO glued to State by a single space
    if m:
        toks = [m.group(1), m.group(2)] + toks[1:]
    di = next((i for i, t in enumerate(toks) if t in DEPTS), None)
    if di is None or di < 3 or len(toks) < di + 3:
        return None
    if di == 3:                                        # District and Type glued by a single space
        dist, _, ftype = toks[2].rpartition(" ")
        toks = toks[:2] + [dist, ftype] + toks[3:]
        di = 4
    name = " ".join(toks[4:di]) if di > 4 else ""
    return dict(sno=toks[0], state=toks[1], district=toks[2], ftype=toks[3], name=name, dept=toks[di],
                d1=toks[-2], d2=toks[-1])

class _RowMatcher:                                     # keeps the old ROW.match() call sites readable
    def match(self, line):
        return split_row(line)
ROW = _RowMatcher()
DATE_FORMATS = ("%d-%b-%y", "%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y")

def parse_date(s):
    s = s.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None

def is_name_line(line):
    t = line.strip()
    return bool(t) and not ROW.match(line) and not t.startswith(("S.NO", "Type of", "Facility", "Result")) \
        and not re.fullmatch(r"[\d\s/-]+", t)

def main(pdf_path, out_path):
    text = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True, check=True).stdout
    lines = text.split("\n")
    cond_col = None
    parsed = []
    for i, line in enumerate(lines):
        if "Conditionality" in line:                   # column positions change from page to page
            cond_col = line.index("Conditionality")
            continue
        m = ROW.match(line)
        if not m:
            continue
        sno, state, district, ftype, name, dept, d1, d2 = (m[k] for k in ("sno", "state", "district", "ftype", "name", "dept", "d1", "d2"))
        name, name_source = name.strip(), "same_line"
        if not name:
            prev = lines[i - 1] if i > 0 else ""
            prev2 = lines[i - 2] if i > 1 else ""
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            m2 = ROW.match(prev2)
            if is_name_line(prev) and m2 and not m2["name"]:
                name, name_source = prev.strip(), "line_above"       # second row of a merged pair
            elif is_name_line(nxt):
                name, name_source = nxt.strip(), "line_below"        # first row of a merged pair
            elif is_name_line(prev):
                name, name_source = prev.strip(), "line_above"
            else:
                name_source = "not_found"
        yes_pos = line.find("Yes", line.find(dept) + len(dept))
        status = "conditional" if (cond_col is not None and yes_pos >= cond_col - 4) else "certified"
        cert, valid = parse_date(d1), parse_date(d2)
        parsed.append(dict(sno=int(sno), state=state.strip(), district=district.strip(), facility_type=ftype,
                           facility_name=name, name_source=name_source, department=dept, status=status,
                           certified_on=cert.isoformat() if cert else d1, valid_until=valid.isoformat() if valid else d2,
                           dates_parsed=bool(cert and valid)))
    cols = list(parsed[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(parsed)
    from collections import Counter
    print(f"rows: {len(parsed)} | max S.NO: {max(r['sno'] for r in parsed)} | departments: {Counter(r['department'] for r in parsed)}")
    print(f"name_source: {Counter(r['name_source'] for r in parsed)} | status: {Counter(r['status'] for r in parsed)}")
    print(f"dates parsed: {sum(r['dates_parsed'] for r in parsed)} | types: {Counter(r['facility_type'] for r in parsed).most_common(8)}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
