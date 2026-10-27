import numpy as np
import pytest

from src import emotion, sentiment
from src.model_utils import ModelOutputError, id2label_to_list, softmax, validate_probs


def test_softmax_sums_to_one_and_is_stable():
    p = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isclose(p.sum(), 1.0) and np.all(np.isfinite(p))
    assert p.argmax() == 1


def test_validate_probs_normalises_and_rejects_bad_input():
    assert np.isclose(validate_probs([2, 1, 1]).sum(), 1.0)
    for bad in ([], [np.nan, 1], [-0.1, 1.1], [0, 0, 0]):
        with pytest.raises(ModelOutputError):
            validate_probs(bad)


def test_id2label_ordering():
    assert id2label_to_list({"2": "c", "0": "a", "1": "b"}) == ["a", "b", "c"]


def test_sentiment_label_mapping_named_and_generic():
    assert sentiment.map_labels(["negative", "neutral", "positive"]) == ["negative", "neutral", "positive"]
    assert sentiment.map_labels(["Positive", "Negative", "Neutral"]) == ["positive", "negative", "neutral"]
    assert sentiment.map_labels(["LABEL_0", "LABEL_1", "LABEL_2"]) == ["negative", "neutral", "positive"]
    with pytest.raises(ModelOutputError):
        sentiment.map_labels(["happy", "sad", "meh"])


def test_sentiment_result_confidence_is_top_probability():
    labels = sentiment.map_labels(["negative", "neutral", "positive"])
    r = sentiment.probs_to_result(np.array([0.91, 0.07, 0.02]), labels)
    assert r.label == "negative" and r.confidence == pytest.approx(0.91)
    assert set(r.probs) == {"negative", "neutral", "positive"}
    assert sum(r.probs.values()) == pytest.approx(1.0)


def test_sentiment_probs_follow_model_label_order_not_fixed_order():
    labels = sentiment.map_labels(["positive", "negative", "neutral"])      # shuffled model
    r = sentiment.probs_to_result(np.array([0.1, 0.8, 0.1]), labels)
    assert r.label == "negative" and r.probs["positive"] == pytest.approx(0.1)


def test_emotion_result():
    r = emotion.probs_to_result(np.array([0.1, 0.72, 0.18]), ["Anger", "Fear", "Joy"])
    assert r.label == "fear" and r.score == pytest.approx(0.72) and r.scores["joy"] == pytest.approx(0.18)


def test_length_mismatch_raises():
    with pytest.raises(ModelOutputError):
        emotion.probs_to_result(np.array([0.5, 0.5]), ["a", "b", "c"])
