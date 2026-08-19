"""Deterministic, audit-logged Kurdish Sorani Unicode orthographic normalization."""

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class NormalizationAudit:
    """Records transformations applied during normalization for transparency."""

    original: str
    normalized: str
    operations_applied: list[str] = field(default_factory=list)


# Unicode mapping table for Sorani character variants
SORANI_CHAR_MAP: dict[str, str] = {
    # Persian / Arabic Yeh variants -> Kurdish Yeh (ی / U+06CC)
    "\u064a": "\u06cc",  # Arabic Yeh (ي) -> Persian/Kurdish Yeh (ی)
    "\u0649": "\u06cc",  # Alef Maksura (ى) -> Kurdish Yeh (ی)
    "\u06d0": "\u06cc",  # Yeh Barree
    # Kaf variants -> Kurdish Kaf (ک / U+06A9)
    "\u0643": "\u06a9",  # Arabic Kaf (ك) -> Kurdish Kaf (ک)
    # Heh / Teh Marbuta variants -> Kurdish Heh (ه / U+0647)
    "\u0629": "\u0647",  # Teh Marbuta (ة) -> Heh (ه)
    "\u06c1": "\u0647",  # Heh Goal (ہ) -> Heh (ه)
    "\u06be": "\u0647",  # Heh Doachashmee (ھ) -> Heh (ه)
    # Waw variants
    "\u0624": "\u0648",  # Waw with Hamza above (ؤ) -> Waw (و)
    # Zero-width joiners / non-joiners normalization
    "\u200c": " ",  # ZWNJ -> space
    "\u200d": "",  # ZWJ -> remove
    "\ufeff": "",  # Byte order mark
}

# Diacritics (Harakat) removal pattern (optional vowels/fatha/damma/kasra/shadda)
DIACRITICS_PATTERN = re.compile(r"[\u064B-\u0652\u0670\u0640]")

# Punctuation and multiple whitespace pattern
PUNCTUATION_PATTERN = re.compile(r"[،؛؟,\.;:!\?\"\(\)\[\]«»\-—\t]+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def normalize_sorani_text(
    text: str,
    strip_diacritics: bool = True,
    normalize_punctuation: bool = True,
) -> NormalizationAudit:
    """Normalize Sorani Kurdish text deterministically without altering lexical distinctions."""
    if not text:
        return NormalizationAudit(original="", normalized="")

    audit = NormalizationAudit(original=text, normalized=text)

    # 1. Unicode NFKC normalization
    nfkc = unicodedata.normalize("NFKC", text)
    if nfkc != text:
        audit.operations_applied.append("nfkc_normalization")
    current = nfkc

    # 2. Character mapping (Yeh, Kaf, Heh variants)
    mapped_chars: list[str] = []
    char_mapped = False
    for char in current:
        if char in SORANI_CHAR_MAP:
            mapped_chars.append(SORANI_CHAR_MAP[char])
            char_mapped = True
        else:
            mapped_chars.append(char)

    if char_mapped:
        audit.operations_applied.append("sorani_char_unification")
    current = "".join(mapped_chars)

    # 3. Optional diacritics removal
    if strip_diacritics:
        stripped = DIACRITICS_PATTERN.sub("", current)
        if stripped != current:
            audit.operations_applied.append("strip_arabic_diacritics")
        current = stripped

    # 4. Punctuation and whitespace normalization
    if normalize_punctuation:
        no_punct = PUNCTUATION_PATTERN.sub(" ", current)
        clean_space = MULTI_SPACE_PATTERN.sub(" ", no_punct).strip()
        if clean_space != current:
            audit.operations_applied.append("normalize_whitespace_and_punctuation")
        current = clean_space

    audit.normalized = current
    return audit
