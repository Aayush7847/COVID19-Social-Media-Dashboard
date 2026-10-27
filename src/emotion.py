"""Emotion detection with a pretrained model.

Model: j-hartmann/emotion-english-distilroberta-base (DistilRoBERTa fine-tuned on six
emotion datasets). Labels: anger, disgust, fear, joy, neutral, sadness, surprise.
Labels are read from the model config, so a different emotion model can be used by
changing EMOTION_MODEL in config.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from src.model_utils import HFTextClassifier, ModelOutputError, validate_probs


@dataclass
class EmotionResult:
    label: str
    score: float
    scores: dict[str, float]        # every emotion the model supports


def probs_to_result(probs: np.ndarray, labels: list[str]) -> EmotionResult:
    p = validate_probs(probs)
    if len(p) != len(labels):
        raise ModelOutputError("Probability vector and label list differ in length.")
    names = [str(l).strip().lower() for l in labels]
    d = {n: float(v) for n, v in zip(names, p)}
    top = max(d, key=d.get)
    return EmotionResult(label=top, score=d[top], scores=d)


class EmotionModel:
    def __init__(self, model_name: str = config.EMOTION_MODEL):
        self.clf = HFTextClassifier(model_name)

    @property
    def name(self) -> str:
        return self.clf.model_name

    @property
    def labels(self) -> list[str]:
        return [str(l).strip().lower() for l in self.clf.labels]

    def predict(self, texts: list[str], batch_size: int = config.BATCH_SIZE) -> list[EmotionResult]:
        probs = self.clf.predict_proba(texts, batch_size)
        return [probs_to_result(row, self.clf.labels) for row in probs]
