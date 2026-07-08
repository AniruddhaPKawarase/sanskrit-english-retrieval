"""Contrastive fine-tune of e5-small with MultipleNegativesRankingLoss.

MNRL uses in-batch negatives: every other positive in the batch is a negative
for the current anchor. So batch size IS the negative count — bigger batch =
better embeddings (03 §3). Gradient accumulation does NOT add in-batch
negatives, so we prioritize raw batch size; CachedMNRL fakes a large effective
batch when VRAM is the limit.
"""
from __future__ import annotations

import os

from .config import DEFAULT, Config


def _to_input_examples(pairs: list[dict]):
    """pairs carry pre-prefixed anchor/positive (see data.build_pairs).
    If a pair has a 'negative' (from bonus hard-negative mining) it becomes an
    (anchor, positive, negative) triplet — MNRL then uses the explicit hard
    negative on top of the in-batch ones. Backward compatible with plain pairs.
    """
    from sentence_transformers import InputExample

    out = []
    for p in pairs:
        texts = [p["anchor"], p["positive"]]
        if p.get("negative"):
            texts.append(p["negative"])
        out.append(InputExample(texts=texts))
    return out


def _build_loss(model, cfg: Config, loss_type: str):
    """Pick the contrastive loss. `gist`/`cached_gist` load a frozen guide model
    that masks near-duplicate in-batch negatives (false-negative fix, mark3).
    Optionally wrapped in MatryoshkaLoss. Falls back to Cached-MNRL if the
    installed sentence-transformers lacks GIST."""
    from sentence_transformers import losses, SentenceTransformer

    lt = loss_type
    try:
        if lt in ("gist", "cached_gist"):
            guide = SentenceTransformer(cfg.guide_model)
            if lt == "cached_gist":
                base = losses.CachedGISTEmbedLoss(model, guide, mini_batch_size=cfg.mini_batch_size)
            else:
                base = losses.GISTEmbedLoss(model, guide)
        elif lt == "cached_mnrl":
            base = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg.mini_batch_size)
        else:  # mnrl
            base = losses.MultipleNegativesRankingLoss(model)
    except AttributeError:  # GIST/Cached not in this ST version
        base = losses.MultipleNegativesRankingLoss(model)

    if cfg.matryoshka:
        base = losses.MatryoshkaLoss(model, base, matryoshka_dims=list(cfg.matryoshka_dims))
    return base


def train(
    model,
    pairs: list[dict],
    cfg: Config = DEFAULT,
    cached: bool = False,
    loss_type: str | None = None,
):
    """Fine-tune `model` in place on `pairs`; returns the saved output path.

    loss_type overrides `cached`/cfg: "mnrl" | "cached_mnrl" | "gist" | "cached_gist".
    Default preserves prior behaviour (MNRL, or Cached-MNRL if cached=True).
    """
    from torch.utils.data import DataLoader

    loss_type = loss_type or cfg.loss_type or ("cached_mnrl" if cached else "mnrl")
    examples = _to_input_examples(pairs)
    if not examples:
        raise ValueError("no training pairs")

    loader = DataLoader(examples, shuffle=True, batch_size=cfg.batch_size)
    loss = _build_loss(model, cfg, loss_type)

    steps_per_epoch = max(1, len(loader))
    warmup = int(steps_per_epoch * cfg.epochs * cfg.warmup_ratio)
    out_path = os.path.join(cfg.output_dir, cfg.run_name)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=cfg.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": cfg.learning_rate},
        use_amp=cfg.use_fp16,
        output_path=out_path,
        show_progress_bar=True,
    )
    return out_path
