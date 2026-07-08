"""Central typed config. No hardcoded values scattered across modules —
everything tunable lives here so experiments are reproducible (rubric: SDLC,
resource management). Immutable: build a new Config to change a run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Config:
    # --- Model (see 03-technical-design.md §1) ---
    base_model: str = "intfloat/multilingual-e5-small"
    # e5 REQUIRES asymmetric prefixes; forgetting them silently tanks recall.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    max_seq_len: int = 128  # verses are short; keeps VRAM low -> bigger batch
    normalize_embeddings: bool = True  # L2-norm; must match at index + query time

    # --- Data (see 03 §2) — all from the assignment's suggested Option-2 list ---
    itihasa_id: str = "rahular/itihasa"          # Sanskrit-English aligned corpora: train + in-domain eval
    gita_id: str = "JDhruv14/Bhagavad-Gita_Dataset"  # Bhagavad Gita: demo + qualitative eval (not gated)
    # AI4Bharat IN22: out-of-domain eval (GATED -> needs HF auth). n-way parallel.
    in22_id: str = "ai4bharat/IN22-Gen"
    in22_config: str = "all"
    in22_split: str = "gen"
    in22_sa_col: str = "sentence_san_Deva"
    in22_en_col: str = "sentence_eng_Latn"
    # FLORES+ : OOD eval fallback if IN22 auth is unavailable (also gated)
    flores_id: str = "openlanguagedata/flores_plus"
    flores_sanskrit: str = "san_Deva"
    iast_augment_ratio: float = 0.25  # fraction of Sanskrit side duplicated as IAST
    dedup: bool = True

    # --- Training (see 03 §3) ---
    batch_size: int = 64  # in-batch negatives -> larger is better (T4 fits ~64-128)
    epochs: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    use_fp16: bool = True
    both_directions: bool = True  # train Sa->En AND En->Sa
    # bonus run-B: reject mined negatives within this cosine margin of the positive — a
    # false-negative guard, since epic corpora contain many near-duplicate shlokas that are
    # semantically equivalent to the gold (naive mining picks them and degrades training).
    hard_neg_margin: float = 0.05
    hard_neg_range_min: int = 25  # also skip more top neighbours before sampling negatives

    # --- advanced recipe (research-backed) ---
    # CachedGISTEmbedLoss: a frozen guide model masks near-duplicate in-batch negatives
    # (the false-negative fix that naive/margin hard-neg mining could not achieve); Cached
    # gives a large effective batch on a T4; Matryoshka yields truncatable embeddings.
    loss_type: str = "mnrl"            # "mnrl" | "cached_mnrl" | "gist" | "cached_gist"
    guide_model: str = "intfloat/multilingual-e5-base"  # frozen GIST guide (Indic-aware, same e5 prefixes)
    mini_batch_size: int = 32          # GradCache chunk for cached losses — VRAM tracks THIS, not batch_size
    matryoshka: bool = False
    matryoshka_dims: tuple = (384, 256, 128, 64)
    # two-stage reranking
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # XLM-R-large family (same as e5), Apache-2.0
    rerank_top_n: int = 50             # retrieve top-N with bi-encoder, then rerank
    rerank_eval_queries: int = 500     # cap rerank-eval query count for speed

    # --- Eval (see 03 §4) ---
    k_values: tuple[int, ...] = (1, 5, 10)
    eval_max_corpus: int = 2000  # cap eval corpus for speed in-notebook

    # --- Repro / paths ---
    seed: int = 42
    output_dir: str = "artifacts"
    run_name: str = "e5-small-sa-en-mnrl"

    def with_(self, **changes) -> "Config":
        """Return a new Config with fields overridden (immutability)."""
        return replace(self, **changes)


DEFAULT = Config()
