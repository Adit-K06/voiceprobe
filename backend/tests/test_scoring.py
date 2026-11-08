from app.core.scoring import normalize, wer


def test_identical_is_zero():
    assert wer("do lakh rupaye", "do lakh rupaye") == 0.0


def test_one_substitution():
    assert wer("do lakh pachaas hazaar", "do lac pachaas hazaar") == 0.25


def test_deletion():
    assert wer("do lakh", "do") == 0.5


def test_insertion():
    assert wer("do lakh", "do lakh rupaye") == 0.5


def test_punctuation_and_case_ignored():
    assert wer("Do lakh, rupaye!", "do lakh rupaye") == 0.0


def test_hindi_nukta_variants_equal():
    # U+095B (precomposed ZA) vs U+091C + U+093C (JA + nukta)
    assert normalize("\u095b") == normalize("\u091c\u093c")


def test_empty_reference():
    assert wer("", "") == 0.0