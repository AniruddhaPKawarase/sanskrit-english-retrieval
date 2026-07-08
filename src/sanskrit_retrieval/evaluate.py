"""Retrieval evaluation: Recall@K / MRR / nDCG, base vs fine-tuned.

Because parallel verses give 1:1 relevance labels for free, we build the eval
set with zero manual annotation (02 decision memo's differentiator). Uses
SentenceTransformers' InformationRetrievalEvaluator so metrics are standard and
not hand-rolled.
"""
from __future__ import annotations

from .config import DEFAULT, Config


def build_ir_eval(queries_sa: list[str], passages_en: list[str], cfg: Config = DEFAULT):
    """Build an InformationRetrievalEvaluator for Sa-query -> En-passage.

    queries_sa[i] is the aligned translation of passages_en[i] -> that is the
    single relevant doc for query i (1:1 gold). Prefixes are applied here so the
    evaluator scores under the same convention training used.
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    n = min(len(queries_sa), len(passages_en), cfg.eval_max_corpus)
    corpus = {f"d{i}": cfg.passage_prefix + passages_en[i] for i in range(n)}
    queries = {f"q{i}": cfg.query_prefix + queries_sa[i] for i in range(n)}
    relevant = {f"q{i}": {f"d{i}"} for i in range(n)}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        precision_recall_at_k=list(cfg.k_values),
        mrr_at_k=list(cfg.k_values),
        ndcg_at_k=list(cfg.k_values),
        show_progress_bar=False,
        score_functions={"cos": _cos},
        name="sa2en",
    )


def _cos(a, b):  # cosine == dot on L2-normalized vectors
    from sentence_transformers.util import cos_sim

    return cos_sim(a, b)


def evaluate_model(model, evaluator) -> dict:
    """Run an evaluator, return its metric dict (Recall/MRR/nDCG @ k)."""
    return evaluator(model, output_path=None)


def before_after(base_model, tuned_model, evaluator) -> dict:
    """Return {'before': {...}, 'after': {...}, 'delta': {...}} — the headline
    result of the whole assignment."""
    before = evaluate_model(base_model, evaluator)
    after = evaluate_model(tuned_model, evaluator)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in before}
    return {"before": before, "after": after, "delta": delta}


def recall_at_k(retrieved: list[list], gold: list, k: int) -> float:
    """Fraction of queries whose single gold id is in the top-k retrieved ids.

    Pure function (no ML deps) so it is unit-tested and reusable by the bonus
    quantization cell, which ranks manually instead of via the IR evaluator.
    """
    if not gold:
        return 0.0
    hits = sum(1 for ret, g in zip(retrieved, gold) if g in ret[:k])
    return hits / len(gold)
