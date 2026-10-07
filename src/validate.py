import re
import sys
import pandas as pd
from sympy import false

import config

MAX_EXAMPLES = 5

def build_patterns():
    return {
        "eid": re.compile(config.EID_PATTERN),
        "email": re.compile(config.EMAIL_PATTERN),
        "phone": re.compile(config.PHONE_PATTERN),
    }

def resolve_ids(df):
    ids = None
    for col in config.ID_COLUMN:
        values = df[col].astype("string").str.strip()
        values = values.mask(values == "")
        ids = values if ids is None else ids.combine_first(values)
    return ids.fillna("(no id)").astype(str)

def main():
    config.check_config()

    if not config.DATA_FILE.exists():
        print (f"ERRPR: data file not found: {config.DATA_FILE}")
        return 1

    column = config.TEXT_COLUMN
    df = pd.read_csv(config.DATA_FILE, usecols=[*config.ID_COLUMN, column])
    text = df[column].fillna("").astype(str)
    df[config.TICKET_ID_COLUMN] = resolve_ids(df)

    print(f"Checking '{column}' in {len(df)} tickets")
    no_id = int((df[config.TICKET_ID_COLUMN] == "(no id)").sum())
    print(f"Tickets with no INC or CTC: {no_id}\n")

    empty = int((text.str.strip() == "").sum())
    print(f"Empty short descriptions: {empty}")

    found_pii = False
    for label, pattern in build_patterns().items():
        hits = text.str.contains(pattern, regex=True)
        count = int(hits.sum())
        if count:
            found_pii = True
            for idx in df.index[hits][:MAX_EXAMPLES]:
                match = pattern.search(text[idx]).group(0)
                print(f" {df.loc[idx, config.TICKET_ID_COLUMN]}: {match!r}")

    print()
    if found_pii:
        print("FAILED. POSSIBLE PII FOUND")
        return 1
    print("PASSED")
    return 0