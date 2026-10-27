"""Shared plumbing for Hugging Face text-classification models.

Everything that does NOT need PyTorch (softmax, label mapping, probability
validation) is a pure function so it can be unit-tested without downloading anything.
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

import config

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded. The message is safe to show to users."""


class ModelOutputError(RuntimeError):
    """Raised when a model returns something we cannot interpret."""


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def validate_probs(probs: Sequence[float]) -> np.ndarray:
    """Check a probability vector is finite and non-negative; renormalise to sum 1."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ModelOutputError(f"Invalid probability vector: {probs!r}")
    total = p.sum()
    if total <= 0 or not math.isfinite(total):
        raise ModelOutputError("Probabilities sum to zero.")
    return p / total


def id2label_to_list(id2label: dict) -> list[str]:
    """Turn a model config's id2label ({0:'negative',...}) into an ordered list."""
    try:
        items = sorted(((int(k), str(v)) for k, v in id2label.items()))
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(f"Unreadable label map: {id2label!r}") from exc
    return [v for _, v in items]


def _torch():
    try:
        import torch  # noqa: WPS433 (lazy import keeps the rest of the app light)
        return torch
    except ImportError as exc:
        raise ModelLoadError(
            "PyTorch is not installed. Close the app and start it with run.bat "
            "(it installs all requirements), or run: pip install -r requirements.txt"
        ) from exc


class HFTextClassifier:
    """Lazy-loading wrapper: tokenizer + sequence-classification model on CPU/GPU."""

    def __init__(self, model_name: str, max_length: int = config.MAX_MODEL_TOKENS):
        self.model_name = model_name
        self.max_length = max_length
        self._tok = None
        self._model = None
        self._device = None
        self.labels: list[str] = []

    # -- loading -----------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        torch = _torch()
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError("The 'transformers' package is not installed. Run run.bat again.") from exc
        try:
            log.info("Loading model %s ...", self.model_name)
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as exc:  # network, disk, auth, bad name ...
            raise ModelLoadError(
                f"Could not load the model '{self.model_name}'. The first run needs an internet "
                f"connection to download it (about 100-500 MB, cached afterwards). "
                f"Check your connection / firewall and try again. Details: {type(exc).__name__}: {exc}"
            ) from exc
        use_cuda = torch.cuda.is_available() and not config.FORCE_CPU
        self._device = torch.device("cuda" if use_cuda else "cpu")
        model.to(self._device)
        model.eval()
        self._tok, self._model = tok, model
        self.labels = id2label_to_list(model.config.id2label)
        log.info("Model %s ready on %s with labels %s", self.model_name, self._device, self.labels)

    # -- inference ---------------------------------------------------------
    def predict_proba(self, texts: Sequence[str], batch_size: int = config.BATCH_SIZE) -> np.ndarray:
        """Return an (n_texts, n_labels) float array of softmax probabilities."""
        self.load()
        torch = _torch()
        n = len(texts)
        if n == 0:
            return np.zeros((0, len(self.labels)))
        # sort by length so each mini-batch needs little padding, then restore order
        order = sorted(range(n), key=lambda i: len(texts[i]))
        out = np.zeros((n, len(self.labels)), dtype=np.float64)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch = [texts[i] if texts[i] else "." for i in idx]
            enc = self._tok(batch, padding=True, truncation=True, max_length=self.max_length,
                            return_tensors="pt").to(self._device)
            with torch.inference_mode():
                logits = self._model(**enc).logits.detach().cpu().numpy()
            if logits.shape[-1] != len(self.labels):
                raise ModelOutputError("Model output size does not match its label list.")
            out[idx] = softmax(logits)
        return out
