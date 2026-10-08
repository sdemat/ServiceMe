"""Check TEXT_COLUMNS for leftover PII. Run: python src/validate.py (exit 1 on a match)."""

import re
import sys

import pandas as pd

import config
from load import resolve_ids

MAX_EXAMPLES = 5  # matches shown per pattern


def build_patterns():
    """Compiled PII regexes from config."""
    greetings = "|".join(re.escape(p) for p in config.NAME_CONTEXT_PHRASES)
    return {
        "eid": re.compile(config.EID_PATTERN),
        "email": re.compile(config.EMAIL_PATTERN),
        "phone": re.compile(config.PHONE_PATTERN),
        "name_after_greeting": re.compile(rf"\b(?:{greetings})[,\s]+[A-Z][a-z]+"),
    }


def main():
    config.check_config()
    if not config.DATA_FILE.exists():
        print(f"ERROR: data file not found: {config.DATA_FILE}")
        return 1

    columns = list(config.TEXT_COLUMNS)
    df = pd.read_csv(config.DATA_FILE, usecols=[*config.ID_COLUMNS, *columns], dtype=str)
    df[config.TICKET_ID_COLUMN] = resolve_ids(df)
    print(f"{len(df)} tickets | no ID: {int((df[config.TICKET_ID_COLUMN] == '(no id)').sum())}\n")

    patterns = build_patterns()
    found = False
    for column in columns:
        text = df[column].fillna("").astype(str)
        print(f"== {column} == (empty: {int((text.str.strip() == '').sum())})")
        for label, pattern in patterns.items():
            hits = text.str.contains(pattern, regex=True)
            print(f"{label}: {int(hits.sum())}")
            if hits.any():
                found = True
                for idx in df.index[hits][:MAX_EXAMPLES]:  # ID and match only, no full text
                    print(f"    {df.loc[idx, config.TICKET_ID_COLUMN]}: {pattern.search(text[idx]).group(0)!r}")
        print()

    print("FAIL: possible PII (may include false positives)." if found else "PASS: no PII patterns found.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
