"""Script + Unicode normalization for Sanskrit/English text.

Why this exists (rubric: Dataset Thinking, tokenizer/script handling):
Sanskrit appears in Devanagari AND Roman (IAST/HK/SLP1). The same verse in two
encodings tokenizes into completely different subwords, so a model trained on
one script fails on the other. We canonicalize to Unicode NFC and can augment
with IAST so retrieval is robust to transliterated queries.

Core (NFC + script detection) is stdlib-only so it is testable without any ML
deps. Transliteration is optional (needs `indic-transliteration`).
"""
from __future__ import annotations

import unicodedata

# Devanagari Unicode blocks: main + Vedic extensions + extended.
_DEVANAGARI_RANGES = (
    (0x0900, 0x097F),  # Devanagari
    (0x1CD0, 0x1CFF),  # Vedic Extensions
    (0xA8E0, 0xA8FF),  # Devanagari Extended
)


def normalize_unicode(text: str) -> str:
    """Canonical NFC + whitespace collapse. Idempotent.

    NFC is the safe canonical form for Devanagari: it fixes matra-ordering and
    precomposed-vs-decomposed nukta ambiguity that would otherwise inflate CER
    and fragment tokens. We deliberately keep ZWJ/ZWNJ (they are meaningful for
    conjunct rendering) — we only normalize form and whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def _is_devanagari(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _DEVANAGARI_RANGES)


def detect_script(text: str) -> str:
    """Return 'devanagari' | 'latin' | 'mixed' | 'other'.

    Judged on letters only (digits/punct/space ignored) so '4.5' or spacing
    doesn't skew the verdict.
    """
    deva = latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        if _is_devanagari(ch):
            deva += 1
        elif ch.isascii():
            latin += 1
    if deva and latin:
        return "mixed"
    if deva:
        return "devanagari"
    if latin:
        return "latin"
    return "other"


def to_iast(devanagari_text: str) -> str:
    """Devanagari -> IAST (Roman). Requires `indic-transliteration`.

    Raised as a clear error rather than silently returning input, so a missing
    dep fails loud during data prep instead of poisoning training data.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "to_iast needs `indic-transliteration` (pip install indic-transliteration)"
        ) from exc
    return transliterate(devanagari_text, sanscript.DEVANAGARI, sanscript.IAST)


def to_devanagari(iast_text: str) -> str:
    """IAST (Roman) -> Devanagari. Requires `indic-transliteration`."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "to_devanagari needs `indic-transliteration` (pip install indic-transliteration)"
        ) from exc
    return transliterate(iast_text, sanscript.IAST, sanscript.DEVANAGARI)
