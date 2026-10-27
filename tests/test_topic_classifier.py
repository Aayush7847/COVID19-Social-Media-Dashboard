import pytest

from src.keyword_extractor import extract_keywords, top_keywords
from src.misinformation import screen_text
from src.topic_classifier import classify_topic


@pytest.mark.parametrize("text,topic", [
    ("Hospitals are crowded and doctors are overwhelmed with patients", "Healthcare"),
    ("Got my second dose of the vaccine today #vaccinated", "Vaccination"),
    ("Lockdown extended, stay at home and keep social distancing", "Lockdown"),
    ("So anxious and stressed, my mental health is suffering", "Mental Health"),
    ("Unemployment is rising and the stock market crashed", "Economy"),
    ("Just had a lovely walk in the park", "General"),
])
def test_topics(text, topic):
    assert classify_topic(text).topic == topic


def test_empty_and_non_string_is_general():
    for bad in ["", "   ", None, 5]:
        assert classify_topic(bad).topic == "General"


def test_tie_goes_to_more_specific_topic():
    # one Healthcare keyword (hospital) and one Vaccination keyword (vaccine)
    r = classify_topic("hospital vaccine")
    assert r.scores == {"Healthcare": 1, "Vaccination": 1}
    assert r.topic == "Vaccination"


def test_highest_score_wins_over_priority():
    r = classify_topic("hospital doctors patients and one vaccine")
    assert r.topic == "Healthcare"


def test_stemmed_forms_still_match():       # the global dataset file is stemmed
    assert classify_topic("vaccin trial result").topic == "Vaccination"


def test_no_partial_word_false_positive():
    assert classify_topic("The jobsite was cleaned").topic == "General"   # 'jobs' != 'jobsite'


def test_misinformation_screen_flags_and_clears():
    assert screen_text("5G towers spread covid, total hoax").flagged
    r = screen_text("Wash your hands and wear a mask")
    assert not r.flagged and r.label == "Not Detected"
    assert screen_text("").flagged is False


def test_keywords_remove_stopwords_and_dedupe():
    kws = extract_keywords("The cases are rising and the cases are scary. https://t.co/x @bob")
    assert kws[0] == "cases" and kws.count("cases") == 1
    assert "the" not in kws and "are" not in kws and "bob" not in kws and "http" not in kws
    assert extract_keywords("") == [] and extract_keywords(None) == []


def test_bulk_keywords_exclude_covid_terms():
    out = dict(top_keywords(["covid vaccine rollout", "coronavirus vaccine news"], 5))
    assert out["vaccine"] == 2 and "covid" not in out and "coronavirus" not in out
