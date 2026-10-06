"""Address-verification workflow: standardise -> validate -> correct -> de-duplicate -> flag status.

Status policy
  INVALID        the address cannot be located (no street line, or neither a ZIP nor a city to anchor it)
  NEEDS_REVIEW   a person must decide: conflicting city/ZIP, ambiguous or failed correction, missing house number,
                 possible duplicate with a different name spelling
  AUTO_CORRECTED every problem was fixed deterministically (format, unit parsing, fill/correct from the ZIP
                 reference, close city typo, exact duplicate merged)
  VERIFIED       matched the reference as submitted (only case / whitespace differences)
"""
import difflib
from collections import defaultdict

import pandas as pd

from reference import STATES
from standardize import standardize_record

CITY_CUTOFF = 0.8          # difflib ratio needed to auto-correct a city
AMBIGUITY_MARGIN = 0.05    # best fuzzy candidate must beat the runner-up by this much
NAME_DUP_CUTOFF = 0.85     # name similarity for a possible duplicate at the same address

REVIEW_REASONS = {"NO_HOUSE_NUMBER", "CITY_ZIP_CONFLICT", "STATE_ZIP_CONFLICT", "CITY_NOT_FOUND", "AMBIGUOUS_CITY", "MULTIPLE_ZIPS",
                  "STATE_UNKNOWN", "POSSIBLE_DUPLICATE"}
INVALID_REASONS = {"MISSING_LINE1", "UNLOCATABLE"}
INFO_REASONS = {"PO_BOX"}


class ZipIndex:
    def __init__(self, ref):
        self.by_zip = {z: (c, s) for z, c, s in zip(ref["zip"], ref["city"], ref["state"])}
        self.zips_by_city = defaultdict(list)
        self.cities_by_state = defaultdict(set)
        for z, c, s in zip(ref["zip"], ref["city"], ref["state"]):
            self.zips_by_city[(c, s)].append(z)
            self.cities_by_state[s].add(c)
        self.cities_by_state = {s: sorted(v) for s, v in self.cities_by_state.items()}


def _fuzzy_city(city, candidates):
    """Return (match, score, ambiguous) using difflib against the candidate city list."""
    scored = sorted(((difflib.SequenceMatcher(None, city, c).ratio(), c) for c in candidates), reverse=True)[:2]
    if not scored or scored[0][0] < CITY_CUTOFF:
        return None, scored[0][0] if scored else 0.0, False
    ambiguous = len(scored) > 1 and scored[0][0] - scored[1][0] < AMBIGUITY_MARGIN
    return scored[0][1], scored[0][0], ambiguous


def resolve_location(city, state, zip5, zip_format_ok, idx):
    """Validate and correct city/state/ZIP against the reference.

    Returns (city, state, zip, corrections, issues, rule_hits, suggestion).
    """
    corr, issues, rules, suggestion = [], [], [], ""
    if state and state not in STATES:
        rules.append("STATE_INVALID")
        state_known = False
    else:
        state_known = bool(state)
    if not zip5:
        rules.append("MISSING_ZIP")
    elif not zip_format_ok:
        rules.append("ZIP_FORMAT_INVALID")
    if not city:
        rules.append("MISSING_CITY")
    if not state:
        rules.append("MISSING_STATE")

    if zip5 and zip_format_ok and zip5 in idx.by_zip:
        zc, zs = idx.by_zip[zip5]
        if not city:
            if state_known and state != zs:
                issues.append("STATE_ZIP_CONFLICT")  # only the ZIP anchors the record and the state disagrees
                suggestion = f"ZIP {zip5} is {zc}, {zs}"
                return city, state, zip5, corr, issues, rules + ["ZIP_STATE_MISMATCH"], suggestion
            city = zc
            corr.append("CITY_FILLED_FROM_ZIP")
        if city == zc:
            if state != zs:
                rules.append("ZIP_STATE_MISMATCH" if state else "MISSING_STATE")
                corr.append("STATE_FILLED_FROM_ZIP" if not state else "STATE_CORRECTED_FROM_ZIP")
                state = zs
            return city, state, zip5, corr, issues, rules, suggestion
        rules.append("CITY_ZIP_MISMATCH")
        score = difflib.SequenceMatcher(None, city, zc).ratio()
        if score >= CITY_CUTOFF and (state in ("", zs) or not state_known):
            corr.append("CITY_FUZZY_CORRECTED")
            if state != zs:
                corr.append("STATE_FILLED_FROM_ZIP" if not state else "STATE_CORRECTED_FROM_ZIP")
            return zc, zs, zip5, corr, issues, rules, suggestion
        issues.append("CITY_ZIP_CONFLICT")
        suggestion = f"ZIP {zip5} is {zc}, {zs}"
        return city, state, zip5, corr, issues, rules, suggestion

    if zip5 and zip_format_ok:
        rules.append("ZIP_NOT_FOUND")
    # no usable ZIP: anchor on city + state
    if not city:
        issues.append("UNLOCATABLE")
        return city, state, zip5, corr, issues, rules, suggestion
    if not state_known:
        issues.append("STATE_UNKNOWN")
        return city, state, zip5, corr, issues, rules, suggestion
    cities = idx.cities_by_state.get(state, [])
    if city not in cities:
        rules.append("CITY_NOT_IN_STATE")
        match, score, ambiguous = _fuzzy_city(city, cities)
        if match is None:
            issues.append("CITY_NOT_FOUND")
            return city, state, zip5, corr, issues, rules, suggestion
        if ambiguous:
            issues.append("AMBIGUOUS_CITY")
            suggestion = f"closest city {match} ({score:.2f})"
            return city, state, zip5, corr, issues, rules, suggestion
        city = match
        corr.append("CITY_FUZZY_CORRECTED")
    zips = idx.zips_by_city[(city, state)]
    if len(zips) == 1:
        corr.append("ZIP_FILLED_FROM_CITY" if not zip5 else "ZIP_CORRECTED_FROM_CITY")
        return city, state, zips[0], corr, issues, rules, suggestion
    issues.append("MULTIPLE_ZIPS")
    suggestion = f"{city}, {state} has ZIPs {', '.join(sorted(zips))}"
    return city, state, zip5, corr, issues, rules, suggestion


def find_duplicates(df):
    """Exact duplicates (same person + address after standardisation) and possible duplicates (same address,
    near-identical name). The earliest record in a group is the master."""
    df = df.sort_values("created_at")
    addr_key = df["final_line1"] + "|" + df["final_line2"] + "|" + df["final_zip"]
    dup_of, possible_of = {}, {}
    for _, grp in df[df["final_line1"] != ""].groupby(addr_key):
        if len(grp) < 2:
            continue
        seen = []  # (record_id, full_name)
        for rid, name in zip(grp["record_id"], grp["first_name_std"] + " " + grp["last_name_std"]):
            exact = next((r for r, n in seen if n == name), None)
            if exact:
                dup_of[rid] = exact
                continue
            close = next((r for r, n in seen if difflib.SequenceMatcher(None, n, name).ratio() >= NAME_DUP_CUTOFF), None)
            if close:
                possible_of[rid] = close
            seen.append((rid, name))
    return dup_of, possible_of


def run_workflow(records, ref):
    idx = ZipIndex(ref)
    out = []
    for rec in records.to_dict("records"):
        s = standardize_record(rec)
        corrections, issues, rules = list(s["changes"]), [], []
        line1 = s["std_line1"]
        if not line1:
            rules.append("MISSING_LINE1")
            issues.append("MISSING_LINE1")
        elif not s["is_po_box"] and not line1.split()[0][0].isdigit():
            rules.append("NO_HOUSE_NUMBER")
            issues.append("NO_HOUSE_NUMBER")
        if s["is_po_box"]:
            rules.append("PO_BOX")
        city, state, zip5, corr, iss, rl, suggestion = resolve_location(
            s["std_city"], s["std_state"], s["std_zip"], s["zip_format_ok"], idx)
        corrections += corr
        issues += iss
        rules += rl
        out.append({
            "record_id": rec["record_id"], "created_at": rec["created_at"], "source_system": rec["source_system"],
            "first_name_std": s["first_name_std"], "last_name_std": s["last_name_std"],
            "final_line1": line1, "final_line2": s["std_line2"], "final_city": city, "final_state": state,
            "final_zip": zip5, "is_po_box": s["is_po_box"], "corrections": corrections, "issues": issues,
            "rules_fired": rules, "suggestion": suggestion,
        })
    res = pd.DataFrame(out)

    dup_of, possible_of = find_duplicates(res)
    res["duplicate_of"] = res["record_id"].map(dup_of).fillna("")
    res["possible_duplicate_of"] = res["record_id"].map(possible_of).fillna("")
    for i in res.index[res["duplicate_of"] != ""]:
        res.at[i, "corrections"] = res.at[i, "corrections"] + ["DUPLICATE_MERGED"]
        res.at[i, "rules_fired"] = res.at[i, "rules_fired"] + ["DUPLICATE"]
    for i in res.index[res["possible_duplicate_of"] != ""]:
        res.at[i, "issues"] = res.at[i, "issues"] + ["POSSIBLE_DUPLICATE"]
        res.at[i, "rules_fired"] = res.at[i, "rules_fired"] + ["POSSIBLE_DUPLICATE"]
        res.at[i, "suggestion"] = f"same address as {res.at[i, 'possible_duplicate_of']}, similar name"

    def status(row):
        if set(row["issues"]) & INVALID_REASONS:
            return "INVALID"
        if row["issues"]:
            return "NEEDS_REVIEW"
        if row["corrections"]:
            return "AUTO_CORRECTED"
        return "VERIFIED"

    res["status"] = res.apply(status, axis=1)
    res["reasons"] = (res["issues"] + res["corrections"] + res["is_po_box"].map(lambda p: ["PO_BOX"] if p else [])) \
        .apply(lambda r: "|".join(dict.fromkeys(r)))
    return res


# ---------------------------------------------------------------------------------------------------------
# Legacy (pre-workflow) exception rules: what the old manual process pulled for a person to check.
# ---------------------------------------------------------------------------------------------------------
def legacy_exceptions(records, ref):
    """Flag raw records the old process sent to manual review: any field that does not exactly match the
    reference or the house format (no standardisation, no auto-correction)."""
    by_zip = {z: (c, s) for z, c, s in zip(ref["zip"], ref["city"], ref["state"])}
    std_suffix = {"ST", "AVE", "RD", "DR", "LN", "CT", "BLVD", "PL", "WAY", "CIR"}
    flags = []
    seen = set()
    for rec in records.sort_values("created_at").to_dict("records"):
        f = []
        l1 = str(rec["address_line1"] if isinstance(rec["address_line1"], str) else "").strip().upper()
        city = str(rec["city"] if isinstance(rec["city"], str) else "").strip().upper()
        state = str(rec["state"] if isinstance(rec["state"], str) else "").strip().upper()
        z = str(rec["zip"] if isinstance(rec["zip"], str) else "").strip()[:5]
        if not l1 or not city or not state or not z:
            f.append("missing_field")
        toks = l1.split()
        if l1 and (toks[-1] not in std_suffix):
            f.append("nonstandard_line1")  # spelled-out suffix, trailing unit, PO box, ...
        if any(t in {"NORTH", "SOUTH", "EAST", "WEST"} for t in toks[1:2]):
            f.append("nonstandard_line1")
        if l1 and not l1[0].isdigit():
            f.append("no_house_number")
        if z and (z not in by_zip or by_zip[z] != (city, state)):
            f.append("zip_city_state_mismatch")
        key = (str(rec["first_name"]).upper(), str(rec["last_name"]).upper(), l1, z)
        if key in seen:
            f.append("duplicate")
        seen.add(key)
        flags.append({"record_id": rec["record_id"], "legacy_flags": "|".join(dict.fromkeys(f))})
    out = pd.DataFrame(flags)
    out["legacy_exception"] = out["legacy_flags"] != ""
    return out
