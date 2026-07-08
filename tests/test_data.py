"""Pure-logic tests for pair construction / dedup / split / IAST augment.
No network, no ML deps, no GPU — guards the silent-failure landmines from 03 §5.
"""
import pytest

from sanskrit_retrieval import data as D
from sanskrit_retrieval.config import DEFAULT

SA = ["धर्मक्षेत्रे", "योगः कर्मसु कौशलम्"]
EN = ["On the field of dharma", "Yoga is skill in action"]


def test_build_pairs_applies_prefixes_both_directions():
    pairs = D.build_pairs(SA, EN, DEFAULT)
    # 2 aligned rows x 2 directions = 4 examples
    assert len(pairs) == 4
    for p in pairs:
        assert p["anchor"].startswith(DEFAULT.query_prefix)      # LANDMINE: query prefix
        assert p["positive"].startswith(DEFAULT.passage_prefix)  # LANDMINE: passage prefix
    directions = {p["direction"] for p in pairs}
    assert directions == {"sa2en", "en2sa"}


def test_build_pairs_single_direction():
    cfg = DEFAULT.with_(both_directions=False)
    pairs = D.build_pairs(SA, EN, cfg)
    assert len(pairs) == 2
    assert all(p["direction"] == "sa2en" for p in pairs)


def test_build_pairs_drops_blank_rows():
    pairs = D.build_pairs(["धर्म", "  "], ["dharma", "blank"], DEFAULT.with_(both_directions=False))
    assert len(pairs) == 1


def test_build_pairs_misaligned_raises():
    with pytest.raises(ValueError):
        D.build_pairs(["a"], ["b", "c"], DEFAULT)


def test_dedup_removes_exact_duplicates():
    pairs = D.build_pairs(SA, EN, DEFAULT) + D.build_pairs(SA, EN, DEFAULT)
    deduped = D.dedup_pairs(pairs)
    assert len(deduped) == 4  # duplicates collapsed


def test_split_disjoint_and_complete():
    items = list(range(100))
    tr, va, te = D.split(items, (0.8, 0.1, 0.1), seed=1)
    assert len(tr) + len(va) + len(te) == 100
    assert set(tr).isdisjoint(va)
    assert set(tr).isdisjoint(te)
    assert set(va).isdisjoint(te)


def test_split_deterministic():
    items = list(range(50))
    assert D.split(items, seed=7) == D.split(items, seed=7)


def test_split_bad_ratios_raise():
    with pytest.raises(ValueError):
        D.split([1, 2, 3], (0.5, 0.4, 0.4))


def test_augment_iast_with_injected_fn():
    # inject a fake transliterator so the test needs no external dep
    fake = lambda s: "IAST_" + s
    cfg = DEFAULT.with_(iast_augment_ratio=1.0)
    out = D.augment_iast(SA, EN, cfg, transliterate_fn=fake)
    assert len(out) == 2
    for p in out:
        assert "IAST_" in p["anchor"]
        assert p["direction"] == "iast2en"
        assert p["positive"].startswith(DEFAULT.passage_prefix)


def test_augment_iast_disabled():
    cfg = DEFAULT.with_(iast_augment_ratio=0.0)
    assert D.augment_iast(SA, EN, cfg, transliterate_fn=lambda s: s) == []
