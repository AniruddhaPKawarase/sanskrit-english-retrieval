"""Standalone mini-RAG demo — clone, install, run with a query, get ranked verses.

    python mini_rag.py "what does the Gita say about karma?"
    python mini_rag.py --k 5 "the duty of a warrior in battle"
    python mini_rag.py "धर्मक्षेत्रे कुरुक्षेत्रे"           # Sanskrit (Devanagari) query
    python mini_rag.py --model artifacts/e5-small-sa-en-mnrl "karma"   # explicit checkpoint

Corpus = Bhagavad Gita (701 verses, downloaded from Hugging Face on first run).
Model = the fine-tuned checkpoint in artifacts/ if present, else base
`intfloat/multilingual-e5-small` (downloaded from HF). Retrieval only; pass an LLM
to `sanskrit_retrieval.rag.answer(...)` to add grounded generation.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sanskrit_retrieval.config import DEFAULT
from sanskrit_retrieval import data, model as M, index as IX


def resolve_model(cfg, explicit: str | None) -> str:
    """Explicit path/HF-id wins; else the fine-tuned checkpoint if it exists on
    disk; else the base model (downloaded from HF, works out of the box)."""
    if explicit:
        return explicit
    ckpt = os.path.join(cfg.output_dir, cfg.run_name)
    return ckpt if os.path.isdir(ckpt) else cfg.base_model


def main():
    ap = argparse.ArgumentParser(description="Sanskrit/English mini-RAG retrieval over the Bhagavad Gita.")
    ap.add_argument("query", nargs="*", help="query text (English or Sanskrit); prompts if omitted")
    ap.add_argument("--k", type=int, default=5, help="number of passages to retrieve (default 5)")
    ap.add_argument("--model", default=None, help="HF id or local path (default: artifacts/ checkpoint, else base)")
    args = ap.parse_args()

    query = " ".join(args.query).strip() or input("Query: ").strip()
    if not query:
        sys.exit("empty query")

    cfg = DEFAULT
    model_id = resolve_model(cfg, args.model)
    kind = "fine-tuned" if model_id != cfg.base_model else "BASE (not fine-tuned — run the notebook for best results)"
    print(f"[model] {model_id}  ({kind})")
    print("[corpus] Bhagavad Gita — downloading + indexing 701 verses ...")

    m = M.load_model(cfg, checkpoint=model_id)
    _, gita_en = data.load_gita()
    idx = IX.VerseIndex(m, gita_en, cfg)

    hits = idx.search(query, k=args.k)
    print(f"\nTop {args.k} passages for: {query!r}\n" + "-" * 60)
    for h in hits:
        print(f"[{h['rank']}] score={h['score']:.3f}\n    {h['text']}\n")


if __name__ == "__main__":
    main()
