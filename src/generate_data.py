"""Deterministic synthetic data generator for the address-verification workflow.

Writes
  data/raw/zip_reference.csv        synthetic reference table of valid ZIP -> city, state
  data/raw/address_records.csv      ~50k patient address records (fictional names) with planted issues
  data/raw/address_ground_truth.csv per record: planted issues, expected status, true clean address, duplicate link

Every record starts from a clean, USPS-style "truth" address drawn from the reference table. Issues are then
planted independently at fixed rates (a record can carry several). The expected status follows the review policy:
  INVALID        cannot be located (no street line, or too little to place the ZIP)
  NEEDS_REVIEW   a human must decide (conflicting city/ZIP, ambiguous fix, missing house number, possible duplicate)
  AUTO_CORRECTED fixable deterministically (formatting, unit parsing, fill from ZIP, small city typo, exact duplicate)
  VERIFIED       already clean

Run:  python src/generate_data.py               (50,000 records, ~6 MB)
      python src/generate_data.py --n 200000     (larger run; keep the CSV under 20 MB for the repo)
"""
import argparse
import os

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import STATES  # noqa: E402

SEED = 7
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")

# leading ZIP digit by region (mirrors the real national layout; the 5-digit codes themselves are synthetic)
ZIP_DIGIT = {**{s: 0 for s in ["CT", "MA", "ME", "NH", "NJ", "RI", "VT"]},
             **{s: 1 for s in ["DE", "NY", "PA"]}, **{s: 2 for s in ["DC", "MD", "NC", "SC", "VA", "WV"]},
             **{s: 3 for s in ["AL", "FL", "GA", "MS", "TN"]}, **{s: 4 for s in ["IN", "KY", "MI", "OH"]},
             **{s: 5 for s in ["IA", "MN", "MT", "ND", "SD", "WI"]}, **{s: 6 for s in ["IL", "KS", "MO", "NE"]},
             **{s: 7 for s in ["AR", "LA", "OK", "TX"]}, **{s: 8 for s in ["AZ", "CO", "ID", "NM", "NV", "UT", "WY"]},
             **{s: 9 for s in ["AK", "CA", "HI", "OR", "WA"]}}
POP_WEIGHT = {"CA": 8, "TX": 6, "FL": 5, "NY": 5, "PA": 3, "IL": 3, "OH": 3, "GA": 3, "NC": 3, "MI": 3}

CITY_PREFIX = ["Ash", "Brook", "Cedar", "Elm", "Fair", "Glen", "Green", "Hill", "Lake", "Maple", "Oak", "Pine",
               "River", "Rock", "Spring", "Stone", "Sun", "West", "North", "East", "Clear", "Silver", "Red", "Fox",
               "Willow", "Bridge", "Mill", "Wood", "Bay", "Haw", "Kings", "Marsh", "Mead", "Shel", "Har", "Lin",
               "Dun", "Brad", "Carl", "Stan"]
CITY_SUFFIX = ["ford", "field", "ville", "ton", "wood", "dale", "port", "view", "haven", "ridge", "burg", "crest",
               "mont", "side", "wick", "bury", "brook", " Falls", " Springs", " Heights", " Grove", " Junction"]
STREET_NAMES = ["Oak", "Maple", "Cedar", "Pine", "Elm", "Main", "Park", "Lake", "Hill", "Church", "Mill", "River",
                "Spring", "Highland", "Sunset", "Ridge", "Meadow", "Forest", "Valley", "Willow", "Chestnut", "Walnut",
                "Birch", "Laurel", "Hickory", "Franklin", "Jackson", "Lincoln", "Madison", "Jefferson", "Adams",
                "Center", "Market", "Union", "Prospect", "Orchard", "Railroad", "School", "Bridge", "Garden", "Cherry",
                "Dogwood", "Magnolia", "Poplar", "Sycamore", "Spruce", "Holly", "Aspen", "Juniper", "Harbor"]
SUFFIXES = {"ST": ["Street", "Str", "St.", "Strt"], "AVE": ["Avenue", "Av", "Ave.", "Aven"],
            "RD": ["Road", "Rd."], "DR": ["Drive", "Dr.", "Drv"], "LN": ["Lane", "Ln."], "CT": ["Court", "Crt"],
            "BLVD": ["Boulevard", "Blvd.", "Boul"], "PL": ["Place", "Pl."], "WAY": ["Wy"], "CIR": ["Circle", "Cir."]}
SUFFIX_P = [.3, .2, .12, .1, .08, .06, .04, .04, .03, .03]
DIRS = {"N": "North", "S": "South", "E": "East", "W": "West"}
FIRST = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth",
         "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
         "Daniel", "Nancy", "Matthew", "Lisa", "Anthony", "Betty", "Mark", "Margaret", "Donald", "Sandra", "Steven",
         "Ashley", "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna", "Kenneth", "Michelle", "Kevin", "Carol",
         "Brian", "Amanda", "George", "Melissa", "Timothy", "Deborah", "Ronald", "Stephanie", "Jose", "Maria",
         "Luis", "Ana", "Wei", "Mei", "Raj", "Priya", "Omar", "Fatima", "Kofi", "Ama", "Hiro", "Yuki", "Liam", "Noah",
         "Olivia", "Emma", "Ava", "Sophia", "Mateo", "Isabella", "Ethan", "Mia", "Lucas", "Harper"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams",
        "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts", "Patel", "Chen", "Kim",
        "Okafor", "Mensah", "Tanaka", "Silva", "Novak", "Kowalski", "Haddad", "Rossi", "Murphy", "Kelly", "Price",
        "Bennett", "Wood", "Barnes", "Ross", "Henderson", "Coleman", "Jenkins", "Perry", "Powell", "Long"]

# planted issue rates (independent draws; incompatible combinations are skipped)
RATES = {
    "suffix_variant": 0.10, "state_spelled_out": 0.02, "unit_in_line1": 0.06, "po_box": 0.03,
    "city_typo_minor": 0.05, "city_typo_major": 0.012, "wrong_city": 0.012,
    "missing_city": 0.015, "missing_state": 0.01, "missing_zip": 0.025, "zip_leading_zero_dropped": 0.012,
    "wrong_state": 0.012, "zip_transposed": 0.015, "missing_line1": 0.006, "no_house_number": 0.006,
    "missing_city_and_zip": 0.005, "exact_duplicate": 0.03, "fuzzy_duplicate": 0.01,
}
SEVERITY = {"VERIFIED": 0, "AUTO_CORRECTED": 1, "NEEDS_REVIEW": 2, "INVALID": 3}


def make_reference(rng):
    rows, used_zip = [], set()
    for st in STATES:
        n_cities = int(rng.integers(18, 30) * POP_WEIGHT.get(st, 1.5) ** 0.5)
        names = set()
        while len(names) < n_cities:
            names.add(rng.choice(CITY_PREFIX) + rng.choice(CITY_SUFFIX))
        prefixes = [ZIP_DIGIT[st] * 100 + p for p in rng.choice(100, 4, replace=False)]
        for city in sorted(names):
            for _ in range(rng.choice([1, 1, 1, 2, 2, 3, 4])):
                while True:
                    z = f"{rng.choice(prefixes):03d}{rng.integers(0, 100):02d}"
                    if z not in used_zip:
                        used_zip.add(z)
                        break
                rows.append((z, city, st))
    return pd.DataFrame(rows, columns=["zip", "city", "state"]).sort_values("zip").reset_index(drop=True)


def typo(word, rng, edits=1):
    """Apply `edits` random character edits; always returns a string different from `word`."""
    for _ in range(20):
        out = _typo_once(word, rng, edits)
        if out.upper() != word.upper():
            return out
    return word + "x"


def _typo_once(word, rng, edits):
    w = list(word)
    for _ in range(edits):
        letters = [i for i, c in enumerate(w) if c.isalpha()]
        if len(letters) < 3:
            break
        i = int(rng.choice(letters[1:]))
        op = rng.integers(0, 4)
        if op == 0:
            w[i] = chr(rng.integers(97, 123)) if w[i].islower() else chr(rng.integers(65, 91))
        elif op == 1:
            del w[i]
        elif op == 2:
            w.insert(i, w[i])
        elif i + 1 < len(w):
            w[i], w[i + 1] = w[i + 1], w[i]
    return "".join(w)


def make_records(rng, ref, n):
    # ---- clean truth ------------------------------------------------------
    st_w = np.array([POP_WEIGHT.get(s, 1.0) for s in ref["state"]])
    ref_idx = rng.choice(len(ref), n, p=st_w / st_w.sum())
    house = rng.integers(1, 9999, n).astype(str)
    sname = rng.choice(STREET_NAMES, n)
    suf = rng.choice(list(SUFFIXES), n, p=SUFFIX_P)
    dirs = np.where(rng.random(n) < 0.12, rng.choice(list(DIRS), n), "")
    has_unit = rng.random(n) < 0.18
    unit_no = np.char.add(rng.integers(1, 40, n).astype(str), rng.choice(["", "", "A", "B", "C"], n))
    unit_type = rng.choice(["APT", "UNIT", "STE"], n, p=[.75, .2, .05])
    first, last = rng.choice(FIRST, n), rng.choice(LAST, n)
    # ~3% of records are household members sharing another record's address and surname (not duplicates)
    hh = rng.choice(n, int(0.03 * n), replace=False)
    src = rng.choice(n, len(hh))
    for arr in (ref_idx, house, sname, suf, dirs, has_unit, unit_no, unit_type, last):
        arr[hh] = arr[src]
    for i in hh[first[hh] == first[src]]:
        first[i] = rng.choice([f for f in FIRST if f != first[i]])
    base = ref.iloc[ref_idx].reset_index(drop=True)
    truth = pd.DataFrame({
        "record_id": [f"R{i:06d}" for i in range(n)],
        "first_name": first, "last_name": last,
        "true_line1": [f"{h} {d + ' ' if d else ''}{s} {x}".upper() for h, d, s, x in zip(house, dirs, sname, suf)],
        "true_line2": np.where(has_unit, np.char.add(np.char.add(unit_type, " "), unit_no), ""),
        "true_city": base["city"].str.upper(), "true_state": base["state"], "true_zip": base["zip"],
    })

    # raw rendering of the clean address in the usual "title case, standard abbreviation" style
    raw = pd.DataFrame({
        "record_id": truth["record_id"], "first_name": first, "last_name": last,
        "address_line1": [f"{h} {d + ' ' if d else ''}{s} {x.title()}" for h, d, s, x in zip(house, dirs, sname, suf)],
        "address_line2": [t.title() for t in truth["true_line2"]],
        "city": base["city"].values, "state": base["state"].values, "zip": base["zip"].values,
    })
    comp = pd.DataFrame({"house": house, "dir": dirs, "street": sname, "suffix": suf})

    # ZIP+4 on some clean records (valid; standardiser should keep the 5-digit ZIP)
    plus4 = rng.random(n) < 0.05
    raw.loc[plus4, "zip"] = raw.loc[plus4, "zip"] + "-" + pd.Series(rng.integers(1000, 9999, plus4.sum())).astype(str).values

    issues = [[] for _ in range(n)]
    expected = np.array(["VERIFIED"] * n, dtype=object)

    def mark(i, issue, status):
        issues[i].append(issue)
        if SEVERITY[status] > SEVERITY[expected[i]]:
            expected[i] = status

    by_city = ref.assign(city_u=ref["city"].str.upper()).groupby(["city_u", "state"])["zip"].apply(list).to_dict()
    cities_by_state = ref.assign(city_u=ref["city"].str.upper()).groupby("state")["city_u"].apply(
        lambda s: sorted(set(s))).to_dict()
    all_zips = set(ref["zip"])
    draws = {k: rng.random(n) < r for k, r in RATES.items()}
    dup_rows = []

    T = {c: truth[c].tolist() for c in ["true_line1", "true_line2", "true_city", "true_state", "true_zip"]}
    R = {c: raw[c].tolist() for c in ["address_line1", "address_line2", "city", "state", "zip"]}
    C = comp.to_dict("list")
    for i in range(n):
        h, d, s, x = C["house"][i], C["dir"][i], C["street"][i], C["suffix"][i]
        line1, line2 = R["address_line1"][i], R["address_line2"][i]
        city, state, zipc = R["city"][i], R["state"][i], R["zip"][i]
        multi_zip = len(by_city[(T["true_city"][i], T["true_state"][i])]) > 1

        if draws["po_box"][i]:
            box = rng.integers(1, 9999)
            T["true_line1"][i], T["true_line2"][i] = f"PO BOX {box}", ""
            fmt = rng.choice(["PO Box {}", "P.O. Box {}", "Post Office Box {}", "POB {}", "P O Box {}"],
                             p=[.35, .25, .15, .15, .1])
            line1, line2 = fmt.format(box), ""
            mark(i, "po_box", "VERIFIED" if fmt == "PO Box {}" else "AUTO_CORRECTED")
        else:
            if draws["suffix_variant"][i]:
                variant = rng.choice(SUFFIXES[x])
                dd = DIRS[d] if d and rng.random() < 0.5 else d
                line1 = f"{h} {dd + ' ' if dd else ''}{s} {variant}"
                mark(i, "suffix_variant", "AUTO_CORRECTED")
            if draws["unit_in_line1"][i]:
                if not T["true_line2"][i]:
                    u = f"{rng.integers(1, 40)}{rng.choice(['', 'A', 'B'])}"
                    T["true_line2"][i] = f"APT {u}"
                else:
                    u = T["true_line2"][i].split(" ", 1)[1]
                ut = T["true_line2"][i].split(" ", 1)[0]
                token = rng.choice([f"Apt {u}", f"Apt. {u}", f"#{u}", f"{ut.title()} {u}"]) if ut == "APT" else f"{ut.title()} {u}"
                if token.startswith("#"):
                    T["true_line2"][i] = f"APT {u}"
                line1, line2 = f"{line1} {token}" if rng.random() < .6 else f"{line1}, {token}", ""
                mark(i, "unit_in_line1", "AUTO_CORRECTED")
            if draws["no_house_number"][i]:
                line1 = line1.split(" ", 1)[1]
                mark(i, "no_house_number", "NEEDS_REVIEW")
        if draws["missing_line1"][i]:
            line1 = ""
            mark(i, "missing_line1", "INVALID")

        # ---- city / state / zip -------------------------------------------------------
        if draws["missing_city_and_zip"][i]:
            city, zipc = "", ""
            mark(i, "missing_city_and_zip", "INVALID")
        else:
            zip_issue = draws["missing_zip"][i] or draws["zip_transposed"][i]
            if zip_issue and (draws["missing_city"][i] or draws["wrong_city"][i]):
                pass  # a ZIP problem is only combined with city problems that leave the city recognisable
            elif draws["missing_city"][i]:
                city = ""
                mark(i, "missing_city", "AUTO_CORRECTED")
            elif draws["city_typo_major"][i]:
                city = typo(city, rng, edits=max(3, len(city) // 3 + 1))
                mark(i, "city_typo_major", "NEEDS_REVIEW")
            elif draws["city_typo_minor"][i]:
                city = typo(city, rng, edits=1)
                mark(i, "city_typo_minor", "AUTO_CORRECTED")
            elif draws["wrong_city"][i]:
                others = [c for c in cities_by_state[state] if c != T["true_city"][i]]
                city = str(rng.choice(others)).title()
                mark(i, "wrong_city", "NEEDS_REVIEW")
            if zip_issue and not draws["state_spelled_out"][i]:
                pass  # state issues and ZIP issues are not combined (a record needs one anchor to be fixable)
            elif draws["missing_state"][i]:
                state = ""
                mark(i, "missing_state", "AUTO_CORRECTED")
            elif draws["wrong_state"][i]:
                state = str(rng.choice([s2 for s2 in STATES if s2 != state]))
                # fixable only when the city still agrees with the ZIP; two conflicting fields -> review
                city_intact = city == R["city"][i] and city != ""
                mark(i, "wrong_state", "AUTO_CORRECTED" if city_intact else "NEEDS_REVIEW")
            elif draws["state_spelled_out"][i]:
                state = STATES[state]
                mark(i, "state_spelled_out", "AUTO_CORRECTED")
            if draws["missing_zip"][i]:
                zipc = ""
                ok = city and state and not draws["city_typo_major"][i] and not draws["wrong_city"][i]
                mark(i, "missing_zip", "AUTO_CORRECTED" if ok and not multi_zip else
                     ("NEEDS_REVIEW" if ok or city else "INVALID"))
            elif draws["zip_transposed"][i]:
                z = T["true_zip"][i]
                j = int(rng.integers(1, 4))
                z2 = z[:j] + z[j + 1] + z[j] + z[j + 2:]
                if z2 == z:
                    z2 = z[:4] + str((int(z[4]) + 1) % 10)
                zipc = z2
                ok = city and state and not draws["city_typo_major"][i] and not draws["wrong_city"][i]
                # a transposition that lands on another real ZIP is a genuine city/ZIP conflict -> human decides
                mark(i, "zip_transposed", "AUTO_CORRECTED" if ok and not multi_zip and z2 not in all_zips
                     else "NEEDS_REVIEW")
            elif draws["zip_leading_zero_dropped"][i] and zipc.startswith("0"):
                zipc = zipc[1:5]
                mark(i, "zip_leading_zero_dropped", "AUTO_CORRECTED")
        for col, val in zip(["address_line1", "address_line2", "city", "state", "zip"], [line1, line2, city, state, zipc]):
            R[col][i] = val
    for col, vals in T.items():
        truth[col] = vals
    for col, vals in R.items():
        raw[col] = vals

    # ---- duplicates: re-submitted copies of earlier records ------------------------------
    gt = truth.assign(planted_issues=["|".join(x) for x in issues], expected_status=expected, duplicate_of="")
    clean_pool = np.where(expected == "VERIFIED")[0]
    n_exact, n_fuzzy = int(RATES["exact_duplicate"] * n), int(RATES["fuzzy_duplicate"] * n)
    picks = rng.choice(clean_pool, n_exact + n_fuzzy, replace=False)
    new_raw, new_gt = [], []
    for k, i in enumerate(picks):
        rec = raw.loc[i].copy()
        g = gt.loc[i].copy()
        rid = f"R{n + k:06d}"
        rec["record_id"], g["record_id"], g["duplicate_of"] = rid, rid, raw.at[i, "record_id"]
        # cosmetic variation that standardisation removes
        rec["address_line1"] = rec["address_line1"].upper() if rng.random() < .5 else rec["address_line1"]
        rec["city"] = rec["city"].upper() if rng.random() < .5 else rec["city"]
        if k < n_exact:
            g["planted_issues"], g["expected_status"] = "exact_duplicate", "AUTO_CORRECTED"
        else:
            which = "first_name" if rng.random() < .6 else "last_name"
            rec[which] = typo(rec[which], rng, 1) if len(rec[which]) > 3 else rec[which] + "e"
            g[which] = rec[which]
            g["planted_issues"], g["expected_status"] = "fuzzy_duplicate", "NEEDS_REVIEW"
        new_raw.append(rec)
        new_gt.append(g)
    raw = pd.concat([raw, pd.DataFrame(new_raw)], ignore_index=True)
    gt = pd.concat([gt, pd.DataFrame(new_gt)], ignore_index=True)

    # cosmetic noise everywhere (case, stray whitespace): not an issue, the standardiser absorbs it
    noisy = rng.random(len(raw)) < 0.08
    raw.loc[noisy, "address_line1"] = "  " + raw.loc[noisy, "address_line1"].str.lower() + " "
    noisy = rng.random(len(raw)) < 0.05
    raw.loc[noisy, "city"] = raw.loc[noisy, "city"].str.upper() + " "

    # shuffle so duplicates are not at the end; add source + timestamp
    order = rng.permutation(len(raw))
    raw, gt = raw.iloc[order].reset_index(drop=True), gt.iloc[order].reset_index(drop=True)
    raw["source_system"] = rng.choice(["EHR_INTAKE", "PATIENT_PORTAL", "CALL_CENTER", "PARTNER_FEED"], len(raw),
                                      p=[.4, .3, .2, .1])
    # base records arrive in id order across 270 days; a re-submitted duplicate arrives 1-60 days after its original
    span = 270 * 24 * 60
    base_min = dict(zip([f"R{i:06d}" for i in range(n)], np.sort(rng.integers(0, span, n))))
    dup_src = dict(zip(gt["record_id"], gt["duplicate_of"]))
    minutes = [base_min[r] if not dup_src[r] else min(base_min[dup_src[r]] + int(rng.integers(1440, 60 * 1440)), span - 1)
               for r in raw["record_id"]]
    raw["created_at"] = (pd.Timestamp("2025-01-01") + pd.to_timedelta(minutes, unit="min")).strftime("%Y-%m-%d %H:%M")
    gt = gt[["record_id", "planted_issues", "expected_status", "duplicate_of", "true_line1", "true_line2",
             "true_city", "true_state", "true_zip"]]
    return raw, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48_000, help="base records before duplicates are appended (~4%)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    os.makedirs(RAW, exist_ok=True)
    ref = make_reference(rng)
    ref.to_csv(os.path.join(RAW, "zip_reference.csv"), index=False)
    raw, gt = make_records(rng, ref, args.n)
    raw.to_csv(os.path.join(RAW, "address_records.csv"), index=False)
    gt.to_csv(os.path.join(RAW, "address_ground_truth.csv"), index=False)
    print(f"reference: {len(ref):,} ZIPs, {ref.groupby(['city', 'state']).ngroups:,} cities")
    print(f"records: {len(raw):,}; expected status mix:")
    print(gt["expected_status"].value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
