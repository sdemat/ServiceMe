"""Load the ticket export into a tidy DataFrame: `df = load_tickets()`."""

import pandas as pd

import config


def resolve_ids(df):
    """One ID per ticket: first non-blank of ID_COLUMNS, else '(no id)'."""
    ids = None
    for col in config.ID_COLUMNS:
        values = df[col].astype("string").str.strip()
        values = values.mask(values == "")
        ids = values if ids is None else ids.combine_first(values)
    return ids.fillna("(no id)").astype(str)


def build_text(df):
    """Join TEXT_COLUMNS (skipping exact repeats), collapse whitespace, truncate."""
    parts = [df[c].fillna("").astype(str).str.strip() for c in config.TEXT_COLUMNS]
    joined = parts[0]
    for p in parts[1:]:
        joined = (joined + " " + p.where(p != joined, "")).str.strip()
    cleaned = joined.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned.str.slice(0, config.MAX_TEXT_CHARS)


def _clean_label(series):
    """Strip whitespace; blank becomes missing."""
    values = series.astype("string").str.strip()
    return values.mask(values == "")


def load_tickets(path=None, verbose=False):
    """Return a new DataFrame from `path` (default DATA_FILE).

    Drops rows with empty text or no start time. Blank labels are kept.
    """
    path = path or config.DATA_FILE
    label_cols = list(config.LABEL_COLUMNS.values())

    header = list(pd.read_csv(path, nrows=0).columns)
    required = [*config.ID_COLUMNS, *config.TEXT_COLUMNS, *label_cols]
    missing = [c for c in required if c not in header]
    start_cols = [c for c in config.START_COLUMNS if c in header]
    if not start_cols:
        missing.append(f"one of {list(config.START_COLUMNS)}")
    if missing:
        raise ValueError(f"Columns not found in {path}: {missing}. Available: {header}")

    closed = [config.CLOSED_COLUMN] if config.CLOSED_COLUMN in header else []
    profile = [c for c in config.PROFILE_COLUMNS if c in header]
    keep = list(dict.fromkeys(required + start_cols + closed + profile))
    raw = pd.read_csv(path, usecols=keep, dtype=str)

    out = pd.DataFrame(index=raw.index)
    out[config.TICKET_ID_COLUMN] = resolve_ids(raw)
    out[config.COMBINED_TEXT_COLUMN] = build_text(raw)

    starts = pd.DataFrame({c: pd.to_datetime(raw[c], errors="coerce") for c in start_cols})
    for c in start_cols:
        out[c] = starts[c]
    out[config.TIME_COLUMN] = starts.bfill(axis=1).iloc[:, 0]
    if closed:
        out[config.CLOSED_COLUMN] = pd.to_datetime(raw[config.CLOSED_COLUMN], errors="coerce")

    for col in [*label_cols, *profile]:
        out[col] = _clean_label(raw[col])

    empty_text = out[config.COMBINED_TEXT_COLUMN] == ""
    bad_time = out[config.TIME_COLUMN].isna()
    keep_rows = ~(empty_text | bad_time)

    if verbose:
        print(f"Loaded {len(out)} rows | dropped: {int(empty_text.sum())} empty text, "
              f"{int(bad_time.sum())} no start time | kept {int(keep_rows.sum())}")

    return out[keep_rows].reset_index(drop=True)
