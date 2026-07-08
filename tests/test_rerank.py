"""Pure-logic tests for rerank rank-metrics (no ML deps)."""
import math

from sanskrit_retrieval.rerank import _rank_metrics


def test_all_rank1():
    m = _rank_metrics([1, 1, 1])
    assert m["recall@1"] == 1.0 and m["mrr@10"] == 1.0 and m["ndcg@10"] == 1.0


def test_missing_gold_counts_zero():
    m = _rank_metrics([None, None])
    assert m["recall@10"] == 0.0 and m["mrr@10"] == 0.0


def test_recall_cutoff_and_mrr():
    # ranks 1, 3, 11 -> recall@1=1/3, recall@5=2/3, recall@10=2/3 (11 excluded)
    m = _rank_metrics([1, 3, 11])
    assert m["recall@1"] == round(1 / 3, 4)
    assert m["recall@5"] == round(2 / 3, 4)
    assert m["recall@10"] == round(2 / 3, 4)
    assert m["mrr@10"] == round((1 / 1 + 1 / 3) / 3, 4)


def test_ndcg_matches_formula():
    m = _rank_metrics([2])  # single query, gold at rank 2
    assert m["ndcg@10"] == round(1.0 / math.log2(3), 4)
