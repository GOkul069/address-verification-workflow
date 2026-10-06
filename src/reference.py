"""Static reference vocabularies: state codes and USPS-style street suffix / directional / unit abbreviations."""
import pandas as pd

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
STATE_BY_NAME = {v.upper(): k for k, v in STATES.items()}

# variant -> standard suffix (standard forms map to themselves)
SUFFIX_MAP = {
    "STREET": "ST", "STR": "ST", "STRT": "ST", "ST": "ST",
    "AVENUE": "AVE", "AV": "AVE", "AVEN": "AVE", "AVE": "AVE",
    "ROAD": "RD", "RD": "RD", "DRIVE": "DR", "DRV": "DR", "DR": "DR", "LANE": "LN", "LN": "LN",
    "COURT": "CT", "CRT": "CT", "CT": "CT", "BOULEVARD": "BLVD", "BOUL": "BLVD", "BLVD": "BLVD",
    "PLACE": "PL", "PL": "PL", "WY": "WAY", "WAY": "WAY", "CIRCLE": "CIR", "CIR": "CIR",
}
STANDARD_SUFFIXES = set(SUFFIX_MAP.values())
DIRECTIONALS = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}
UNIT_TYPES = {"APT": "APT", "APARTMENT": "APT", "UNIT": "UNIT", "STE": "STE", "SUITE": "STE", "#": "APT"}


def load_zip_reference(path):
    ref = pd.read_csv(path, dtype=str)
    ref["city"] = ref["city"].str.upper()
    return ref
