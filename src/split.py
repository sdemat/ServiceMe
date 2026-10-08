"""Time-based split: oldest train, then validation, newest test. Run: python src/split.py"""

import pandas as pd

import config
from load import add_types, load_tickets

NAMES = ("train", "val", "test")


def split(df):
    """Sorted by start time; returns train, val, test."""
    df = df.sort_values(config.TIME_COLUMN, kind="stable").reset_index(drop=True)
    n = len(df)
    if config.SPLIT_METHOD == "date":
        test_start = int((df[config.TIME_COLUMN] < pd.Timestamp(config.SPLIT_DATE)).sum())
    else:
        test_start = n - int(round(n * config.TEST_FRACTION))
    val_start = test_start - int(round(n * config.VAL_FRACTION))
    return df[:val_start], df[val_start:test_start], df[test_start:]


def split_path(name):
    return config.SPLITS_DIR / f"{name}.pkl"


def load_split(name):
    """A saved split, as written by main()."""
    return pd.read_pickle(split_path(name))


def main():
    config.check_config()
    config.ensure_dirs()
    df = add_types(load_tickets(verbose=True))
    for name, part in zip(NAMES, split(df)):
        part.to_pickle(split_path(name))
        t = part[config.TIME_COLUMN]
        print(f"{name:5} {len(part):5} tickets | {t.min():%Y-%m-%d} to {t.max():%Y-%m-%d} | "
              f"inquiries {int(part['is_inquiry'].sum())}, incidents {int(part['is_incident'].sum())}")


if __name__ == "__main__":
    main()
