"""Two-stage retrieval: bi-encoder retrieves top-N, a cross-encoder reranks.

Cross-encoders read (query, passage) jointly, so they rank more accurately than
a bi-encoder — but a reranker that hasn't seen the language can REGRESS on
cross-lingual pairs. So `evaluate_rerank` returns a paired before/after on the
SAME candidate set; only ship reranking if the delta is positive (mark3 gate).

`_rank_metrics` is pure (no ML deps) and unit-tested.
"""
from __future__ import annotations

import math

from .config import DEFAULT, Config


def load_reranker(cfg: Config = DEFAULT):
    """Load the cross-encoder reranker (SentenceTransformers CrossEncoder)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(cfg.rerank_model, max_length=512)


def rerank(reranker, query: str, passages: list[str]) -> list[dict]:
    """Score (query, passage) pairs with the cross-encoder; return passages
    reordered best-first. Uses RAW text (no e5 prefixes — different model)."""
    scores = reranker.predict([(query, p) for p in passages])
    order = sorted(range(len(passages)), key=lambda i: -float(scores[i]))
    return [{"rank": r + 1, "score": float(scores[i]), "text": passages[i]} for r, i in enumerate(order)]


def _rank_metrics(ranks: list, ks=(1, 5, 10)) -> dict:
    """ranks[i] = 1-based position of query i's gold (or None if absent).
    Single gold per query, so IDCG=1 → nDCG@k = 1/log2(rank+1) when rank<=k."""
    n = len(ranks) or 1
    out = {}
    for k in ks:
        out[f"recall@{k}"] = sum(1 for r in ranks if r and r <= k) / n
    out["mrr@10"] = sum(1.0 / r for r in ranks if r and r <= 10) / n
    out["ndcg@10"] = sum(1.0 / math.log2(r + 1) for r in ranks if r and r <= 10) / n
    return {k: round(v, 4) for k, v in out.items()}


def evaluate_rerank(bi_model, reranker, queries: list[str], corpus: list[str], cfg: Config = DEFAULT):
    """Paired before/after: retrieve top-N per query with the bi-encoder, then
    rerank that SAME candidate set. gold for query i is corpus[i] (1:1).
    Returns {'before': metrics, 'after': metrics, 'top_n', 'n_queries'}."""
    import numpy as np
    from .model import encode_queries, encode_passages

    n = min(len(queries), len(corpus), cfg.rerank_eval_queries)
    q, c = queries[:n], corpus[:n]
    P = encode_passages(bi_model, c, cfg).astype("float32")
    Q = encode_queries(bi_model, q, cfg).astype("float32")
    topn = cfg.rerank_top_n

    before, after = [], []
    for i in range(n):
        sims = Q[i] @ P.T
        cand = list(np.argsort(-sims)[:topn])
        before.append(cand.index(i) + 1 if i in cand else None)
        scores = reranker.predict([(q[i], c[j]) for j in cand])
        reordered = [cand[j] for j in np.argsort(-np.asarray(scores))]
        after.append(reordered.index(i) + 1 if i in reordered else None)

    return {
        "before": _rank_metrics(before),
        "after": _rank_metrics(after),
        "top_n": topn,
        "n_queries": n,
    }
