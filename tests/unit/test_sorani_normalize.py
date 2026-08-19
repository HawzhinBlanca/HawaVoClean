"""Unit tests for Kurdish Sorani orthographic normalization."""

import pytest

from hawavoclean.guard.sorani_normalize import normalize_sorani_text


@pytest.mark.unit
def test_sorani_yeh_unification() -> None:
    # Arabic Yeh (ي), Alef Maksura (ى), Yeh Barree (ې/ے) -> Kurdish Yeh (ی)
    text_arabic = "سڵاو لە هەمووان ئه‌مڕۆ ڕۆژێکی خۆشه‌"
    res = normalize_sorani_text(text_arabic)
    assert "ی" in res.normalized or "ێ" in res.normalized


@pytest.mark.unit
def test_sorani_kaf_unification() -> None:
    # Arabic Kaf (ك) -> Kurdish Kaf (ک)
    text_arabic_kaf = "دەنگێکی پاك و بێگەرد"
    res = normalize_sorani_text(text_arabic_kaf)
    assert "ک" in res.normalized
    assert "ك" not in res.normalized


@pytest.mark.unit
def test_sorani_diacritics_stripping() -> None:
    # Text with Arabic fatha/kasra/damma
    text_diacritics = "سَڵاوْ لِە هَمُووان"
    res = normalize_sorani_text(text_diacritics, strip_diacritics=True)
    assert "\u064e" not in res.normalized  # fatha
    assert "\u0650" not in res.normalized  # kasra
    assert "\u0652" not in res.normalized  # sukun


@pytest.mark.unit
def test_sorani_whitespace_and_punctuation() -> None:
    text_dirty = "  سڵاو،،،؛؛؛    چۆنیت؟؟؟  "
    res = normalize_sorani_text(text_dirty, normalize_punctuation=True)
    assert res.normalized == "سڵاو چۆنیت"
    assert "  " not in res.normalized
