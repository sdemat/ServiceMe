"""Trained ticket model: TicketModel.fit(df), then .predict(texts)."""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config
from features import make_vectorizer, top_k

TEXT = config.COMBINED_TEXT_COLUMN
PII = [re.compile(p) for p in (config.EID_PATTERN, config.EMAIL_PATTERN, config.PHONE_PATTERN)]


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def pool(hist, sc, cut, k, qb):
    """Per ticket: median, 25th/75th percentile and best case of matches >= cut (up to SIMILAR_MAX).
    Fewer than SIMILAR_MIN close matches: nearest k, flagged."""
    keep = sc >= cut
    keep[:, config.SIMILAR_MAX:] = False
    few = keep.sum(axis=1) < config.SIMILAR_MIN
    keep[few] = False
    keep[few, :k] = True
    nb = np.where(keep, hist, np.nan)
    return (np.nanmedian(nb, axis=1), np.nanpercentile(nb, 25, axis=1), np.nanpercentile(nb, 75, axis=1),
            np.nanquantile(nb, qb, axis=1), few, keep.sum(axis=1))


def lr(**kw):
    return LogisticRegression(C=config.REGULARIZATION_C, max_iter=config.MAX_ITER, **kw)


class TicketModel:
    @classmethod
    def fit(cls, df):
        """df: tickets with add_types columns."""
        m = cls()
        m.vec = make_vectorizer().fit(df[TEXT])
        typed = df[df["typed"]].reset_index(drop=True)
        m.X = m.vec.transform(typed[TEXT])
        m.ids = typed[config.TICKET_ID_COLUMN].tolist()
        m.texts = typed[TEXT].tolist()
        m.words = [words(t) for t in m.texts]
        m.is_inc = typed["is_incident"].to_numpy()
        m.hours = typed["hours"].to_numpy()
        m.cats = [None if pd.isna(v) else str(v) for v in typed[config.TARGET_COLUMN]]
        m.clf_type = lr(class_weight=config.CLASS_WEIGHT).fit(m.X, typed["is_inquiry"].to_numpy())

        m.fields, m.majority = {}, {}  # field -> classifier, or None for a fixed default
        for name in (*config.ENTRY_FIELDS_STRICT, *config.ENTRY_FIELDS_SOFT):
            col = config.LABEL_COLUMNS[name]
            ok = df[col].notna().to_numpy()
            y = df.loc[ok, col].astype(str)
            share = y.value_counts(normalize=True)
            m.majority[name] = share.index[0]
            fixed = share.iloc[0] >= config.DEFAULT_SHARE or y.nunique() < 2
            m.fields[name] = None if fixed else lr().fit(m.vec.transform(df.loc[ok, TEXT]), y)
        return m

    def suggest(self, idx, sc):
        """Most typical wording among close matches, built only from recurring words; or None."""
        nb = idx[sc >= config.SUGGEST_CUTOFF]
        if len(nb) < config.SUGGEST_MIN:
            return None
        seen = Counter(w for i in nb for w in self.words[i])
        need = max(config.SUGGEST_WORD_MIN_COUNT, int(np.ceil(config.SUGGEST_WORD_SHARE * (len(nb) - 1))))
        S = (self.X[nb] @ self.X[nb].T).toarray()
        typical = (S.sum(axis=1) - 1) / (len(nb) - 1)
        for j in sorted(range(len(nb)), key=lambda j: (-round(typical[j], 4), len(self.texts[nb[j]]))):
            i = nb[j]
            if self.words[i] and all(seen[w] - 1 >= need for w in self.words[i]) \
                    and not any(p.search(self.texts[i]) for p in PII):
                return self.texts[i]
        return None

    def predict(self, texts):
        """One dict per text: inquiry flag, time estimate, similar tickets, field guesses, suggested wording."""
        X = self.vec.transform(texts)
        p = self.clf_type.predict_proba(X)[:, list(self.clf_type.classes_).index(True)]

        idx, sc = top_k(X, self.X, config.SUGGEST_MAX)
        k = config.SIMILAR_TICKETS_K
        Xi, hrs = self.X[self.is_inc], self.hours[self.is_inc]
        idx_i, sc_i = top_k(X, Xi, max(k, config.SIMILAR_MAX))
        kk = min(k, idx_i.shape[1])
        med, lo, hi, best, few, used = pool(hrs[idx_i], sc_i, config.SIMILARITY_CUTOFF, kk, config.BEST_CASE_QUANTILE)

        guesses = {}
        for name, clf in self.fields.items():
            if clf is None:
                guesses[name] = [[(self.majority[name], 1.0)] for _ in texts]
            else:
                P = clf.predict_proba(X)
                top = np.argsort(-P, axis=1)[:, :3]
                guesses[name] = [[(clf.classes_[j], float(P[i, j])) for j in top[i]] for i in range(len(texts))]

        out = []
        for i in range(len(texts)):
            out.append({
                "p_inquiry": float(p[i]),
                "inquiry": bool(p[i] >= config.INQUIRY_CUTOFF),
                "time": {"median": float(med[i]), "low": float(lo[i]), "high": float(hi[i]), "best": float(best[i]),
                         "matches": int(used[i]), "rough": bool(few[i])},
                "similar": [{"id": self.ids[j], "text": self.texts[j], "sim": float(s)}
                            for j, s in zip(idx[i][:config.SIMILAR_SHOWN], sc[i][:config.SIMILAR_SHOWN])],
                "fields": {name: g[i] for name, g in guesses.items()},
                "suggestion": self.suggest(idx[i], sc[i]),
            })
        return out
