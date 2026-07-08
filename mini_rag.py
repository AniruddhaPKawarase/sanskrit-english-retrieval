"""Standalone mini-RAG demo — clone, install, run with a query, get ranked verses.

    python mini_rag.py "what does the Gita say about karma?"
    python mini_rag.py --k 5 "the duty of a warrior in battle"
    python mini_rag.py "धर्मक्षेत्रे कुरुक्षेत्रे"           # Sanskrit (Devanagari) query
    python mini_rag.py --model artifacts/e5-small-sa-en-mnrl "karma"   # explicit local checkpoint

Corpus = Bhagavad Gita (701 verses, downloaded from Hugging Face on first run).
Model resolution order: --model > local artifacts/ checkpoint > the fine-tuned checkpoint
published on Hugging Face (FINETUNED_HF_ID) > base multilingual-e5-small. Retrieval only; pass an
LLM to sanskrit_retrieval.rag.answer(...) to add grounded generation.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sanskrit_retrieval.config import DEFAULT
from sanskrit_retrieval import data, model as M, index as IX

# Fine-tuned weights are hosted on the HF Hub (too large for GitHub). Published with:
#   SentenceTransformer("<drive>/artifacts/e5-small-sa-en-mnrl").push_to_hub("sanskrit-e5-small-retrieval")
FINETUNED_HF_ID = "AniruddhaAI/sanskrit-e5-small-retrieval"


def resolve_model(cfg, explicit: str | None):
    if explicit:
        return explicit, "explicit"
    ckpt = os.path.join(cfg.output_dir, cfg.run_name)
    if os.path.isdir(ckpt):
        return ckpt, "local fine-tuned checkpoint"
    return FINETUNED_HF_ID, "fine-tuned (Hugging Face Hub)"


def main():
    ap = argparse.ArgumentParser(description="Sanskrit/English mini-RAG retrieval over the Bhagavad Gita.")
    ap.add_argument("query", nargs="*", help="query text (English or Sanskrit); prompts if omitted")
    ap.add_argument("--k", type=int, default=5, help="number of passages to retrieve (default 5)")
    ap.add_argument("--model", default=None, help="HF id or local path (default: local checkpoint, else HF, else base)")
    args = ap.parse_args()

    query = " ".join(args.query).strip() or input("Query: ").strip()
    if not query:
        sys.exit("empty query")

    cfg = DEFAULT
    model_id, source = resolve_model(cfg, args.model)
    print(f"[model] {model_id}  ({source})")
    try:
        m = M.load_model(cfg, checkpoint=model_id)
    except Exception as e:
        print(f"[warn] could not load '{model_id}' ({type(e).__name__}); falling back to base {cfg.base_model}")
        print("       (publish the fine-tuned weights to HF, or run the notebook to create artifacts/, for best results)")
        m = M.load_model(cfg, checkpoint=cfg.base_model)

    print("[corpus] Bhagavad Gita — downloading + indexing 701 verses ...")
    _, gita_en = data.load_gita()
    idx = IX.VerseIndex(m, gita_en, cfg)

    hits = idx.search(query, k=args.k)
    print(f"\nTop {args.k} passages for: {query!r}\n" + "-" * 60)
    for h in hits:
        print(f"[{h['rank']}] score={h['score']:.3f}\n    {h['text']}\n")


if __name__ == "__main__":
    main()
