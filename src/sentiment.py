"""Sentiment analysis with a pretrained Twitter model.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa-base trained on
~124M tweets, fine-tuned for 3-class sentiment). Output labels are read from the
model's own config - not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from src.model_utils import HFTextClassifier, ModelOutputError, validate_probs

_SYNONYMS = {
    "negative": "negative", "neg": "negative",
    "neutral": "neutral", "neu": "neutral",
    "positive": "positive", "pos": "positive",
}
# Older Cardiff checkpoints expose generic names; their documented order is neg, neu, pos.
_GENERIC_ORDER = ["negative", "neutral", "positive"]


@dataclass
class SentimentResult:
    label: str                       # 'positive' | 'neutral' | 'negative'
    confidence: float                # probability of the predicted label
    probs: dict[str, float]          # always has all three keys


def map_labels(raw_labels: list[str]) -> list[str]:
    """Map the model's label names onto positive/neutral/negative."""
    mapped = [_SYNONYMS.get(str(l).strip().lower()) for l in raw_labels]
    if all(mapped) and len(set(mapped)) == 3:
        return mapped
    generic = all(str(l).strip().lower().startswith("label_") for l in raw_labels)
    if generic and len(raw_labels) == 3:
        return list(_GENERIC_ORDER)
    raise ModelOutputError(f"Cannot map sentiment labels {raw_labels!r} to positive/neutral/negative.")


def probs_to_result(probs: np.ndarray, labels: list[str]) -> SentimentResult:
    """Convert one probability vector + mapped labels into a SentimentResult."""
    p = validate_probs(probs)
    if len(p) != len(labels):
        raise ModelOutputError("Probability vector and label list differ in length.")
    d = {lab: float(v) for lab, v in zip(labels, p)}
    top = max(d, key=d.get)
    return SentimentResult(label=top, confidence=d[top],
                           probs={k: d[k] for k in config.SENTIMENT_ORDER})


class SentimentModel:
    def __init__(self, model_name: str = config.SENTIMENT_MODEL):
        self.clf = HFTextClassifier(model_name)

    @property
    def name(self) -> str:
        return self.clf.model_name

    def predict(self, texts: list[str], batch_size: int = config.BATCH_SIZE) -> list[SentimentResult]:
        probs = self.clf.predict_proba(texts, batch_size)
        labels = map_labels(self.clf.labels)
        return [probs_to_result(row, labels) for row in probs]
