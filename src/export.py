"""Write the model as JSON for the Chrome extension. Run: python src/export.py

Uses the --final model if train.py made one, else the evaluation model.
The file holds past ticket text (similar tickets, suggestions): keep extension/model/ out of git.
"""

import json

import joblib
import numpy as np

import config
from model import TEXT
from split import load_split
from train import bundle_path

OUT = config.EXTENSION_MODEL_DIR / "model.json"


def r(a, d=5):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def linear(clf):
    """Weights of a fitted LogisticRegression."""
    return {"classes": [c.item() if hasattr(c, "item") else c for c in clf.classes_],
            "coef": r(clf.coef_, 4), "intercept": r(clf.intercept_, 4)}


def build(m):
    v = m.vec
    X = m.X.tocsr()
    return {
        "config": config.extension_constants(),
        "vectorizer": {"vocab": {t: int(i) for t, i in v.vocabulary_.items()}, "idf": r(v.idf_),
                       "ngram_range": list(v.ngram_range), "token_pattern": v.token_pattern,
                       "lowercase": bool(v.lowercase), "sublinear_tf": bool(v.sublinear_tf)},
        "type": linear(m.clf_type),
        "fields": {n: ({"default": m.majority[n]} if c is None else linear(c)) for n, c in m.fields.items()},
        "tickets": {"ids": m.ids, "texts": m.texts, "cats": m.cats, "is_incident": m.is_inc.tolist(),
                    "hours": [None if np.isnan(h) else round(float(h), 3) for h in m.hours]},
        "matrix": {"indptr": X.indptr.tolist(), "indices": X.indices.tolist(), "data": r(X.data)},
    }


def check(m, blob, n=50):
    """Stage-1 probabilities recomputed from the exported weights must match the model."""
    texts = load_split("test")[TEXT].head(n).tolist()
    X = m.vec.transform(texts)
    w, b = np.array(blob["type"]["coef"])[0], blob["type"]["intercept"][0]
    p = 1 / (1 + np.exp(-(X @ w + b)))
    want = [x["p_inquiry"] for x in m.predict(texts)]
    return float(np.abs(p - want).max())


def main():
    final = bundle_path(True)
    path = final if final.exists() else bundle_path(False)
    if path != final:
        print("No --final model; exporting the evaluation model.")
    m = joblib.load(path)
    blob = build(m)
    config.ensure_dirs()
    OUT.write_text(json.dumps(blob, separators=(",", ":")))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) from {path.name}")
    print(f"Max stage-1 probability difference after export: {check(m, blob):.1e}")


if __name__ == "__main__":
    main()
