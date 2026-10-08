"""Fit and save the model. Run: python src/train.py [--final]

Default fits on train + val (so evaluate.py can score it on test).
--final also fits on test, for the model you ship.
"""

import argparse

import joblib
import pandas as pd

import config
from model import TicketModel
from split import load_split


def bundle_path(final):
    return config.classifier_path(config.DEFAULT_MODEL, "final" if final else "bundle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="also fit on the test split")
    final = ap.parse_args().final

    config.ensure_dirs()
    parts = ["train", "val", "test"] if final else ["train", "val"]
    df = pd.concat([load_split(n) for n in parts], ignore_index=True)
    m = TicketModel.fit(df)
    joblib.dump(m, bundle_path(final))
    fixed = [f for f, c in m.fields.items() if c is None]
    print(f"Fit on {len(df)} tickets ({'+'.join(parts)}); {len(m.ids)} typed, {int(m.is_inc.sum())} incidents.")
    print(f"Fixed defaults: {', '.join(f'{f}={m.majority[f]}' for f in fixed) or 'none'}")
    print(f"Saved {bundle_path(final)}")


if __name__ == "__main__":
    main()
