import pandas as pd

from src.preprocessing import clean_dataframe, is_valid_text, normalize_for_model, normalize_key, text_hash


def test_normalize_replaces_urls_and_mentions():
    assert normalize_for_model("Hi @bob see https://t.co/abc  now") == "Hi @user see http now"


def test_normalize_handles_none_nan_and_numbers():
    assert normalize_for_model(None) == ""
    assert normalize_for_model(float("nan")) == ""
    assert normalize_for_model(123) == "123"


def test_normalize_unescapes_html_and_collapses_whitespace():
    assert normalize_for_model("Fish &amp; chips\n\n  today") == "Fish & chips today"


def test_is_valid_text():
    assert is_valid_text("Stay safe")
    assert not is_valid_text("")
    assert not is_valid_text("   ")
    assert not is_valid_text("12345 !!!")      # no letters
    assert not is_valid_text("ab")             # too short
    assert not is_valid_text(None)


def test_dedupe_key_ignores_case_punctuation_urls():
    assert normalize_key("Stay SAFE! http://x.co/1") == normalize_key("stay safe https://y.co/2")


def test_hash_is_stable_and_text_specific():
    assert text_hash("hello world") == text_hash("hello world")
    assert text_hash("hello world") != text_hash("hello there")


def test_clean_dataframe_stats():
    df = pd.DataFrame({"text": ["Good news", "good NEWS!", "", "   ", "1234", "Another post"]})
    out, st = clean_dataframe(df)
    assert list(out["text"]) == ["Good news", "Another post"]
    assert st == {"input_rows": 6, "empty_or_invalid": 3, "duplicates": 1, "valid": 2}


def test_clean_dataframe_empty():
    out, st = clean_dataframe(pd.DataFrame({"text": []}))
    assert out.empty and st["valid"] == 0
