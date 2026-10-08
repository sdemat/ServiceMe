"""Text features and nearest-neighbor search."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config


def make_vectorizer():
    """Unfitted TF-IDF vectorizer from config."""
    return TfidfVectorizer(ngram_range=config.TFIDF_WORD_NGRAMS, min_df=config.TFIDF_MIN_DF,
                           max_features=config.TFIDF_MAX_FEATURES, sublinear_tf=True)


def top_k(Xq, Xdb, k, chunk=500):
    """Indices and scores of the k most similar Xdb rows per Xq row (rows are L2-normalized)."""
    k = min(k, Xdb.shape[0])
    idx_out, sc_out = [], []
    for start in range(0, Xq.shape[0], chunk):
        S = (Xq[start:start + chunk] @ Xdb.T).toarray()
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        part_sc = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-part_sc, axis=1)
        idx_out.append(np.take_along_axis(part, order, axis=1))
        sc_out.append(np.take_along_axis(part_sc, order, axis=1))
    return np.vstack(idx_out), np.vstack(sc_out)
