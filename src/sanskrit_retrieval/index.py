"""FAISS index for the retrieval/RAG demo.

IndexFlatIP (exact inner product) is deliberate: on ~90-186K vectors it is fast,
and being exact it gives ground-truth recall so ANN approximation can't confound
the eval (03 §4 / 05 §2). Vectors must be L2-normalized so inner product ==
cosine — the model layer already normalizes.
"""
from __future__ import annotations

import numpy as np

from .config import DEFAULT, Config
from .model import encode_passages, encode_queries


class VerseIndex:
    """Thin wrapper over a flat FAISS index + the passage texts it holds."""

    def __init__(self, model, passages: list[str], cfg: Config = DEFAULT):
        import faiss

        self.model = model
        self.cfg = cfg
        self.passages = list(passages)  # keep our own copy (no external mutation)
        emb = encode_passages(model, self.passages, cfg).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = encode_queries(self.model, [query], self.cfg).astype("float32")
        scores, ids = self.index.search(q, k)
        return [
            {"rank": r + 1, "score": float(scores[0][r]), "text": self.passages[i]}
            for r, i in enumerate(ids[0])
            if i != -1
        ]
