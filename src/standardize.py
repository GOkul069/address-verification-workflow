"""Step 1 of the workflow: standardise raw address fields to one USPS-style format.

Each function returns the standardised value(s) plus a list of change codes. Pure case / whitespace changes are not
reported as corrections: they never needed a human.
"""
import re

from reference import DIRECTIONALS, STATE_BY_NAME, STATES, SUFFIX_MAP, UNIT_TYPES

PO_BOX_RE = re.compile(r"^(?:P\s*O|POST\s+OFFICE|POB|PO)\s*(?:BOX|BX)?\s*#?\s*(\d+)$")
UNIT_RE = re.compile(r"(?:,\s*|\s+)(?:(APT|APARTMENT|UNIT|STE|SUITE)\s+|(#)\s*)([A-Z0-9-]+)$")
ZIP_RE = re.compile(r"^(\d{5})(?:-?(\d{4}))?$")


def clean(text):
    """Upper-case, strip, collapse whitespace, drop periods. None/NaN -> ''."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    return re.sub(r"\s+", " ", str(text).replace(".", " ").upper()).strip()


def standardize_line1(raw_line1, raw_line2):
    line1, line2 = clean(raw_line1), clean(raw_line2)
    changes = []
    m = PO_BOX_RE.match(line1)
    if m:
        std = f"PO BOX {m.group(1)}"
        if std != line1:
            changes.append("PO_BOX_STANDARDISED")
        return std, line2, changes, True
    m = UNIT_RE.search(line1)
    if m and not line2:
        unit_type = UNIT_TYPES[m.group(1) or m.group(2)]
        line2 = f"{unit_type} {m.group(3)}"
        line1 = line1[:m.start()].rstrip(", ")
        changes.append("UNIT_PARSED")
    tokens = line1.replace(",", " ").split()
    if len(tokens) >= 3 and tokens[1] in DIRECTIONALS:
        tokens[1] = DIRECTIONALS[tokens[1]]
        changes.append("DIRECTIONAL_STANDARDISED")
    if len(tokens) >= 2 and tokens[-1] in SUFFIX_MAP and SUFFIX_MAP[tokens[-1]] != tokens[-1]:
        tokens[-1] = SUFFIX_MAP[tokens[-1]]
        changes.append("SUFFIX_STANDARDISED")
    line1 = " ".join(tokens)
    if not changes and line1 != re.sub(r"\s+", " ", str(raw_line1 or "").upper()).strip():
        changes.append("PUNCTUATION_REMOVED")  # e.g. "St." -> "ST": inconsistent abbreviation format
    if line2:
        parts = line2.replace("#", "# ").split()
        if parts[0] in UNIT_TYPES and len(parts) >= 2:
            line2 = f"{UNIT_TYPES[parts[0]]} {' '.join(parts[1:])}"
    return line1, line2, changes, False


def standardize_state(raw_state):
    s = clean(raw_state)
    if s in STATES or not s:
        return s, []
    if s in STATE_BY_NAME:
        return STATE_BY_NAME[s], ["STATE_ABBREVIATED"]
    return s, []  # unknown value: left for validation


def standardize_zip(raw_zip):
    """Return (zip5, plus4, changes, format_ok)."""
    z = clean(raw_zip).replace(" ", "")
    if not z:
        return "", "", [], True
    m = ZIP_RE.match(z)
    if m:
        return m.group(1), m.group(2) or "", [], True
    if re.fullmatch(r"\d{4}", z):  # leading zero lost when the ZIP was stored as a number
        return "0" + z, "", ["ZIP_ZERO_PADDED"], True
    return z, "", [], False


def standardize_record(rec):
    line1, line2, ch1, is_po = standardize_line1(rec["address_line1"], rec["address_line2"])
    state, ch2 = standardize_state(rec["state"])
    zip5, plus4, ch3, zip_ok = standardize_zip(rec["zip"])
    return {
        "std_line1": line1, "std_line2": line2, "std_city": clean(rec["city"]), "std_state": state,
        "std_zip": zip5, "std_zip4": plus4, "is_po_box": is_po, "zip_format_ok": zip_ok,
        "first_name_std": clean(rec["first_name"]), "last_name_std": clean(rec["last_name"]),
        "changes": ch1 + ch2 + ch3,
    }
