"""Dataset loading + pair construction for cross-lingual retrieval.

Two layers, kept separate on purpose:
  * PURE logic (build_pairs / augment_iast / dedup_pairs / split) — no ML deps,
    fully unit-tested here without network or GPU.
  * THIN loaders (load_itihasa / load_gita / load_in22 / load_flores_sanskrit)
    — call HuggingFace `datasets`; only run on Colab. Kept tiny so the
    untestable surface is small.

A training example is a plain dict {"anchor", "positive", "direction"} with e5
prefixes ALREADY applied — train.py just wraps these into InputExample.
"""
from __future__ import annotations

import random
from typing import Callable, Iterable

from .config import DEFAULT, Config
from .normalize import normalize_unicode


# --------------------------------------------------------------------------
# PURE logic (unit-tested, dependency-free)
# --------------------------------------------------------------------------
def build_pairs(
    sanskrit: list[str],
    english: list[str],
    cfg: Config = DEFAULT,
) -> list[dict]:
    """Aligned (sanskrit[i], english[i]) -> directional prefixed examples.

    Sa->En: anchor=Sanskrit(query), positive=English(passage).
    En->Sa (if cfg.both_directions): anchor=English(query), positive=Sanskrit(passage).
    Empty/blank rows are dropped (fail-fast on bad alignment).
    """
    if len(sanskrit) != len(english):
        raise ValueError(f"misaligned: {len(sanskrit)} sa vs {len(english)} en")

    qp, pp = cfg.query_prefix, cfg.passage_prefix
    pairs: list[dict] = []
    for sa_raw, en_raw in zip(sanskrit, english):
        sa, en = normalize_unicode(sa_raw), normalize_unicode(en_raw)
        if not sa or not en:
            continue
        pairs.append({"anchor": qp + sa, "positive": pp + en, "direction": "sa2en"})
        if cfg.both_directions:
            pairs.append({"anchor": qp + en, "positive": pp + sa, "direction": "en2sa"})
    return pairs


def augment_iast(
    sanskrit: list[str],
    english: list[str],
    cfg: Config = DEFAULT,
    transliterate_fn: Callable[[str], str] | None = None,
) -> list[dict]:
    """Add IAST (Roman) variants of a fraction of Sanskrit-side pairs so the
    model is robust to transliterated queries (03 §2). Deterministic via seed.

    transliterate_fn is injectable so this is testable without the
    indic-transliteration dependency; defaults to normalize.to_iast on Colab.
    """
    if cfg.iast_augment_ratio <= 0 or not sanskrit:
        return []
    if transliterate_fn is None:
        from .normalize import to_iast
        transliterate_fn = to_iast

    rng = random.Random(cfg.seed)
    n = max(1, int(len(sanskrit) * cfg.iast_augment_ratio))
    idx = rng.sample(range(len(sanskrit)), min(n, len(sanskrit)))

    qp, pp = cfg.query_prefix, cfg.passage_prefix
    out: list[dict] = []
    for i in idx:
        sa, en = normalize_unicode(sanskrit[i]), normalize_unicode(english[i])
        if not sa or not en:
            continue
        iast = normalize_unicode(transliterate_fn(sa))
        if not iast:
            continue
        out.append({"anchor": qp + iast, "positive": pp + en, "direction": "iast2en"})
    return out


def dedup_pairs(pairs: Iterable[dict]) -> list[dict]:
    """Drop exact (anchor, positive) duplicates; preserve first-seen order.
    Near-duplicate verses can create false negatives in contrastive training."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for p in pairs:
        key = (p["anchor"], p["positive"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def split(
    items: list,
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42,
) -> tuple[list, list, list]:
    """Deterministic train/val/test split with disjoint members."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


# --------------------------------------------------------------------------
# THIN loaders (Colab only — need `datasets`; not unit-tested)
# --------------------------------------------------------------------------
def load_itihasa(split_name: str = "train") -> tuple[list[str], list[str]]:
    """Return (sanskrit, english) aligned lists from rahular/itihasa.

    Read the repo's plain-text files directly via hf_hub_download instead of
    load_dataset: the dataset ships a *script* loader, which `datasets>=3.0`
    (what Colab installs) refuses to run. Direct file read is version-independent.
    Files are one sentence per line, index-aligned across the .sn / .en files.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 (Colab-only import)

    file_split = {"train": "train", "validation": "dev", "test": "test"}[split_name]
    sa_path = hf_hub_download(DEFAULT.itihasa_id, f"{file_split}.sn.csv", repo_type="dataset")
    en_path = hf_hub_download(DEFAULT.itihasa_id, f"{file_split}.en.csv", repo_type="dataset")
    with open(sa_path, encoding="utf-8") as f:
        sa = f.read().splitlines()
    with open(en_path, encoding="utf-8") as f:
        en = f.read().splitlines()
    n = min(len(sa), len(en))  # guard against a trailing-newline length mismatch
    return sa[:n], en[:n]


def load_gita() -> tuple[list[str], list[str]]:
    """Bhagavad Gita (701 verses): Sanskrit (Devanagari) + English translation.
    Demo / qualitative-eval corpus (assignment-suggested). Not gated.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(DEFAULT.gita_id, split="train")
    return [r["sanskrit"] for r in ds], [r["english"] for r in ds]


def load_in22() -> tuple[list[str], list[str]]:
    """AI4Bharat IN22 (~1,024 n-way sentences incl. Sanskrit): OOD eval.

    GATED -> needs `notebook_login()` + accepted terms. n-way parallel, so we
    pick the Sanskrit (san_Deva) and English (eng_Latn) columns explicitly —
    several columns are Devanagari (Hindi/Marathi/Sanskrit), so column choice
    matters.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(DEFAULT.in22_id, DEFAULT.in22_config, split=DEFAULT.in22_split)
    return [r[DEFAULT.in22_sa_col] for r in ds], [r[DEFAULT.in22_en_col] for r in ds]


def load_flores_sanskrit(split_name: str = "devtest") -> tuple[list[str], list[str]]:
    """Return (sanskrit, english) held-out OOD eval pairs from FLORES+ (OOD
    fallback if IN22 is unavailable). n-way parallel: row i is the same sentence
    across languages, so we load each language and zip by index.
    """
    from datasets import load_dataset  # noqa: PLC0415

    sa = load_dataset(DEFAULT.flores_id, DEFAULT.flores_sanskrit, split=split_name)
    en = load_dataset(DEFAULT.flores_id, "eng_Latn", split=split_name)
    return [r["text"] for r in sa], [r["text"] for r in en]
