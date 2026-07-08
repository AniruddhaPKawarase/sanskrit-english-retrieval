"""Optional bonus experiments (Colab-only, need ML deps).

Hard-negative mining: turn (anchor, positive) pairs into (anchor, positive,
negative) triplets so a run-B fine-tune trains against genuinely confusable
passages, not just in-batch negatives. Thin wrapper over SentenceTransformers'
`mine_hard_negatives`; column names and kwargs vary across versions, so both are
handled defensively.

Lesson from the pilot run: on the epic corpus the mined "negatives" were on
average MORE similar to the anchor than the positives — i.e. near-duplicate
shlokas that are actually relevant (false negatives), which degraded run B. The
fixes: skip more top neighbours (range_min) and require a similarity margin
below the positive, so semantically-equivalent verses are not mined as negatives.
"""
from __future__ import annotations

from .config import DEFAULT, Config


def mine_triplets(
    pairs: list[dict],
    model,
    cfg: Config = DEFAULT,
    num_negatives: int = 1,
    range_min: int | None = None,
    range_max: int = 100,
    margin: float | None = None,
) -> list[dict]:
    """Mine hard negatives for anchor/positive pairs -> triplet dicts.

    range_min skips the nearest neighbours (the true positive + near-duplicates);
    margin rejects any negative within `margin` cosine of the positive (false-
    negative guard). Both default to the config values. Returns dicts shaped like
    data.build_pairs output plus a 'negative' key, so they flow straight into
    train.train().
    """
    from datasets import Dataset
    from sentence_transformers.util import mine_hard_negatives

    range_min = cfg.hard_neg_range_min if range_min is None else range_min
    margin = cfg.hard_neg_margin if margin is None else margin

    ds = Dataset.from_dict(
        {"anchor": [p["anchor"] for p in pairs], "positive": [p["positive"] for p in pairs]}
    )
    kwargs = dict(
        num_negatives=num_negatives,
        range_min=range_min,
        range_max=range_max,
        batch_size=cfg.batch_size,
        use_faiss=True,
    )
    # `margin` guards against false negatives; retry without it if the installed
    # sentence-transformers version does not accept the kwarg.
    try:
        mined = mine_hard_negatives(ds, model, margin=margin, **kwargs)
    except TypeError:
        mined = mine_hard_negatives(ds, model, **kwargs)

    out: list[dict] = []
    for row in mined:
        neg = row.get("negative") or row.get("negative_1")
        if not neg:
            continue
        out.append(
            {"anchor": row["anchor"], "positive": row["positive"], "negative": neg, "direction": "hardneg"}
        )
    return out
