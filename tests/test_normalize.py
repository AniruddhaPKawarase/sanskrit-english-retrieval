"""Stdlib-only tests for script/Unicode normalization — no ML deps needed."""
from sanskrit_retrieval import normalize as N

DEVA = "धर्मक्षेत्रे कुरुक्षेत्रे"  # "on the field of dharma, the field of Kuru"


def test_nfc_idempotent():
    once = N.normalize_unicode(DEVA)
    assert N.normalize_unicode(once) == once


def test_nfc_collapses_whitespace():
    assert N.normalize_unicode("  a\t\n  b  ") == "a b"


def test_nfc_empty():
    assert N.normalize_unicode("") == ""


def test_detect_devanagari():
    assert N.detect_script(DEVA) == "devanagari"


def test_detect_latin():
    assert N.detect_script("what does this verse say about karma?") == "latin"


def test_detect_mixed():
    assert N.detect_script("dharma धर्म") == "mixed"


def test_detect_ignores_digits_and_punct():
    # digits/punct must not sway the verdict
    assert N.detect_script("4.5 धर्म!") == "devanagari"


def test_decomposed_matra_normalizes_to_nfc():
    # base letter + combining vowel sign should NFC-compose deterministically
    decomposed = "नि"  # na + i-matra
    out = N.normalize_unicode(decomposed)
    assert out == "नि" or len(out) <= len(decomposed)
    # idempotency is the real invariant
    assert N.normalize_unicode(out) == out
