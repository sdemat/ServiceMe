"""Score the saved model on the test split. Run: python src/evaluate.py (after split.py and train.py).

Prints counts and rates only, never ticket text.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from model import TEXT
from split import load_split
from train import bundle_path

EDGES = list(config.BUCKET_EDGES)
G = dict(  # pass thresholds (same as the readiness notebook)
    incident_recall=0.85, informed=0.30, time_2x=0.05, time_bucket=0.10, best_hold=0.75,
    miss_cost=3, extra_time_cost=0.25, bad_bucket_cost=1, e2e_gain=0.10,
    soft_top3=0.80, soft_below_default=0.02, similar_lift=0.10,
    suggest_cover=0.50, suggest_same=0.80, suggest_meaning=0.90,
)


def within_2x(pred, actual, floor=0.25):
    ratio = np.maximum(pred, floor) / np.maximum(actual, floor)
    return float(((ratio >= 0.5) & (ratio <= 2)).mean())


def near_bucket(pred, actual):
    return np.abs(np.digitize(pred, EDGES) - np.digitize(actual, EDGES)) <= 1


def main():
    m = joblib.load(bundle_path(False))
    test = load_split("test")
    r = m.predict(test[TEXT].tolist())
    typed = test["typed"].to_numpy()
    out = []

    def add(tier, name, ok, detail):
        out.append((tier, name, ok, detail))

    # Stage 1: inquiry vs incident
    t = test[typed].reset_index(drop=True)
    rt = [x for x, f in zip(r, typed) if f]
    y = t["is_inquiry"].to_numpy()
    p = np.array([x["p_inquiry"] for x in rt])
    pred = np.array([x["inquiry"] for x in rt])
    ri, rn = float(pred[y].mean()), float((~pred[~y]).mean())
    inf, auc = ri + rn - 1, float(roc_auc_score(y, p))
    print(f"Test {len(test)} tickets | typed {len(t)} (inquiries {int(y.sum())}, incidents {int((~y).sum())})")
    add("MUST", "stage 1: inquiry vs incident", rn >= G["incident_recall"] and inf >= G["informed"],
        f"incidents caught {rn:.0%} (need {G['incident_recall']:.0%}), informedness {inf:.0%} (need {G['informed']:.0%}), AUC {auc:.2f}")

    # Time estimate on incidents
    inc = ~y
    act = t.loc[inc, "hours"].to_numpy()
    est = np.array([x["time"]["median"] for x in rt])
    best = np.array([x["time"]["best"] for x in rt])
    rough = np.array([x["time"]["rough"] for x in rt])
    hrs = m.hours[m.is_inc]
    cands = np.quantile(hrs, np.linspace(0.05, 0.95, 19))
    best_c = max(cands, key=lambda c: within_2x(np.full(len(hrs), c), hrs))
    bk = np.digitize(hrs, EDGES)
    base = [np.median(hrs), best_c, np.median(hrs[bk == np.bincount(bk, minlength=len(EDGES) + 1).argmax()])]
    b2 = max(within_2x(np.full(len(act), v), act) for v in base)
    b1 = max(float(near_bucket(np.full(len(act), v), act).mean()) for v in base)
    e2, e1 = within_2x(est[inc], act), float(near_bucket(est[inc], act).mean())
    add("MUST", "time estimate: incidents", e2 - b2 >= G["time_2x"] and e1 - b1 >= G["time_bucket"],
        f"within 2x {e2:.0%} vs {b2:.0%} best constant (need +{G['time_2x']:.0%}), within-1-bucket {e1:.0%} vs {b1:.0%} "
        f"(need +{G['time_bucket']:.0%}); {1 - rough[inc].mean():.0%} had close matches, rest nearest {config.SIMILAR_TICKETS_K}")
    hold = float((act >= best[inc]).mean())
    add("NICE", "time estimate: best case", hold >= G["best_hold"],
        f"ticket took at least the best case {hold:.0%} (need {G['best_hold']:.0%})")

    # End to end: weighted cost against two lazy strategies
    ok_b = np.zeros(len(t), dtype=bool)
    ok_b[inc] = near_bucket(est[inc], act)
    missed, extra, bad = pred & inc, ~pred & y, ~pred & inc & ~ok_b
    cost = G["miss_cost"] * missed.sum() + G["extra_time_cost"] * extra.sum() + G["bad_bucket_cost"] * bad.sum()
    lazy = min(G["miss_cost"] * inc.sum(), G["extra_time_cost"] * y.sum() + G["bad_bucket_cost"] * (inc & ~ok_b).sum())
    add("MUST", "end to end: what the technician sees", cost <= (1 - G["e2e_gain"]) * lazy,
        f"weighted cost {cost:.0f} vs {lazy:.0f} cheaper lazy strategy (need {G['e2e_gain']:.0%} lower); "
        f"{int(missed.sum())} incidents shown AT THE WINDOW, {int(extra.sum())} inquiries given a time")

    # Fields
    for name in (*config.ENTRY_FIELDS_STRICT, *config.ENTRY_FIELDS_SOFT):
        col = config.LABEL_COLUMNS[name]
        has = test[col].notna().to_numpy()
        truth = test.loc[has, col].astype(str).to_numpy()
        g = [x["fields"][name] for x, f in zip(r, has) if f]
        top1 = float(np.mean([a[0][0] == v for a, v in zip(g, truth)]))
        top3 = float(np.mean([v in [c for c, _ in a] for a, v in zip(g, truth)]))
        dflt = float((truth == m.majority[name]).mean())
        how = "fixed default" if m.fields[name] is None else "model"
        if name in config.ENTRY_FIELDS_STRICT:
            add("MUST", f"entry: {name}", top1 >= dflt, f"{how}: top-1 {top1:.0%} vs always '{m.majority[name]}' {dflt:.0%}")
        else:
            add("NICE", f"entry: {name}", top3 >= G["soft_top3"] and top1 >= dflt - G["soft_below_default"],
                f"{how}: top-1 {top1:.0%} (default {dflt:.0%}), top-3 {top3:.0%} (need {G['soft_top3']:.0%})")

    # Similar tickets: same-category hit outside the dominant category
    cat = config.TARGET_COLUMN
    known = {i: c for i, c in zip(m.ids, m.cats)}
    tr_cats = pd.Series([c for c in m.cats if c])
    share = tr_cats.value_counts(normalize=True)
    lab = test[cat].notna().to_numpy() & (test[cat].astype(str) != share.index[0]).to_numpy()
    sel = np.where(lab)[0]
    if len(sel) >= 20:
        truth = test[cat].astype(str).to_numpy()
        hit = float(np.mean([any(known.get(s["id"]) == truth[i] for s in r[i]["similar"]) for i in sel]))
        chance = float(np.mean(1 - (1 - share.reindex(truth[sel]).fillna(0).to_numpy()) ** config.SIMILAR_SHOWN))
        add("MUST", "similar tickets", hit >= chance + G["similar_lift"],
            f"top-{config.SIMILAR_SHOWN} same-'{cat}' hit {hit:.0%} vs random {chance:.0%} (need +{G['similar_lift']:.0%}), {len(sel)} non-dominant tickets")

    # Suggested description
    sg = [x["suggestion"] for x in rt]
    has = np.array([s is not None for s in sg])
    cover = float(has.mean())
    if has.sum() >= 20:
        Xq = m.vec.transform(t[TEXT])
        S = (Xq @ Xq.T).toarray()
        np.fill_diagonal(S, 0)
        nn = S.argmax(axis=1)
        pairs = np.where(has & has[nn] & (S.max(axis=1) >= config.SUGGEST_CUTOFF))[0]
        same = float(np.mean([sg[i] == sg[nn[i]] for i in pairs])) if len(pairs) else float("nan")
        r2 = m.predict([s for s in sg if s])
        r1 = [x for x, h in zip(rt, has) if h]
        keep = [a["inquiry"] == b["inquiry"] and all(a["fields"][n][0][0] == b["fields"][n][0][0] for n in config.ENTRY_FIELDS_SOFT)
                for a, b in zip(r1, r2)]
        meaning = float(np.mean(keep))
        add("NICE", "suggested description", cover >= G["suggest_cover"] and same >= G["suggest_same"] and meaning >= G["suggest_meaning"],
            f"{cover:.0%} covered (need {G['suggest_cover']:.0%}), same for near-identical {same:.0%} (need {G['suggest_same']:.0%}), "
            f"meaning unchanged {meaning:.0%} (need {G['suggest_meaning']:.0%})")
    else:
        add("NICE", "suggested description", False, f"only {int(has.sum())} tickets get a suggestion")

    print(f"\n{'RESULT':7} {'TIER':5} {'TEST':38} DETAIL\n" + "-" * 120)
    for tier, name, ok, detail in sorted(out, key=lambda o: (o[0] != "MUST", o[1])):
        print(f"{'PASS' if ok else 'FAIL':7} {tier:5} {name:38} {detail}")


if __name__ == "__main__":
    main()
