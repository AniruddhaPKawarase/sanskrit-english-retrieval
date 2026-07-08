"""Pure-logic test for the recall_at_k helper (no ML deps)."""
from sanskrit_retrieval.evaluate import recall_at_k


def test_recall_hit_within_k():
    retrieved = [[3, 1, 2], [0, 5, 9]]
    gold = [1, 0]
    assert recall_at_k(retrieved, gold, k=3) == 1.0


def test_recall_miss_outside_k():
    retrieved = [[3, 1, 2], [7, 5, 9]]
    gold = [1, 0]  # q0 gold=1 at rank2 (in k=1? no); q1 gold=0 absent
    assert recall_at_k(retrieved, gold, k=1) == 0.0  # neither gold at rank 1


def test_recall_partial():
    retrieved = [[1, 8, 8], [8, 8, 8]]
    gold = [1, 0]  # q0 hit, q1 miss
    assert recall_at_k(retrieved, gold, k=3) == 0.5


def test_recall_empty():
    assert recall_at_k([], [], k=10) == 0.0


def test_recall_respects_k_cutoff():
    retrieved = [[9, 9, 1]]  # gold at rank 3
    assert recall_at_k(retrieved, [1], k=2) == 0.0
    assert recall_at_k(retrieved, [1], k=3) == 1.0
