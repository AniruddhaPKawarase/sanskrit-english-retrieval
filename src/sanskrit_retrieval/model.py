"""Load the embedding model and encode with correct e5 conventions.

e5 is asymmetric: queries get "query: " and passages get "passage: ", and
embeddings are L2-normalized. Centralizing this here means no caller can forget
the prefix (03 section 5 landmine) — everyone goes through encode_queries/passages.
"""
from __future__ import annotations

from .config import DEFAULT, Config


def load_model(cfg: Config = DEFAULT, checkpoint: str | None = None):
    """Load a SentenceTransformer (base model or a saved fine-tuned checkpoint)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(checkpoint or cfg.base_model)
    model.max_seq_length = cfg.max_seq_len
    return model


def _encode(model, texts: list[str], prefix: str, cfg: Config):
    prefixed = [prefix + t for t in texts]
    return model.encode(
        prefixed,
        normalize_embeddings=cfg.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=cfg.batch_size,
    )


def encode_queries(model, texts: list[str], cfg: Config = DEFAULT):
    """Encode search queries (applies 'query: ' + L2-norm)."""
    return _encode(model, texts, cfg.query_prefix, cfg)


def encode_passages(model, texts: list[str], cfg: Config = DEFAULT):
    """Encode corpus passages (applies 'passage: ' + L2-norm)."""
    return _encode(model, texts, cfg.passage_prefix, cfg)


def devanagari_fertility(model, words: list[str]) -> float:
    """Mean subword tokens per word for the model's tokenizer — the tokenizer
    diagnostic the assignment explicitly asks about (03 section 1). >1 means Sanskrit
    words fragment into multiple subwords, burning context and signal.
    """
    tok = model.tokenizer
    counts = [len(tok.tokenize(w)) for w in words if w.strip()]
    return sum(counts) / len(counts) if counts else 0.0
