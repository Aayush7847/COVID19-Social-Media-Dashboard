"""One analysis engine used by BOTH the single-post pages and bulk processing."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import config
from src.emotion import EmotionModel, EmotionResult
from src.keyword_extractor import extract_keywords
from src.misinformation import MisinfoResult, screen_text
from src.model_utils import ModelLoadError
from src.preprocessing import is_valid_text, normalize_for_model
from src.sentiment import SentimentModel, SentimentResult
from src.topic_classifier import TopicResult, classify_topic

log = logging.getLogger(__name__)


@dataclass
class PostAnalysis:
    text: str
    sentiment: SentimentResult | None
    emotion: EmotionResult | None
    topic: TopicResult
    keywords: list[str]
    misinfo: MisinfoResult
    errors: dict[str, str] = field(default_factory=dict)   # model name -> message


class PostAnalyzer:
    """Holds the two pretrained models (loaded once, on first use)."""

    def __init__(self, sentiment_model: SentimentModel | None = None,
                 emotion_model: EmotionModel | None = None):
        self.sentiment_model = sentiment_model or SentimentModel()
        self.emotion_model = emotion_model or EmotionModel()

    @property
    def signature_parts(self) -> tuple[str, str]:
        return self.sentiment_model.name, self.emotion_model.name

    # -- model predictions for a batch; failures become None + message ---------------
    def predict_batch(self, texts: list[str], batch_size: int = config.BATCH_SIZE):
        """Return (sentiments, emotions, errors) lists aligned with `texts`.

        A failing mini-batch is retried one post at a time so one bad post cannot
        discard its neighbours. ModelLoadError (model missing / offline) is raised
        to the caller because nothing can be analysed without the model.
        """
        n = len(texts)
        sents: list[SentimentResult | None] = [None] * n
        emos: list[EmotionResult | None] = [None] * n
        errs: list[str] = [""] * n
        for kind, model, out in (("sentiment", self.sentiment_model, sents),
                                 ("emotion", self.emotion_model, emos)):
            try:
                res = model.predict(texts, batch_size)
                for i, r in enumerate(res):
                    out[i] = r
            except ModelLoadError:
                raise
            except Exception as batch_exc:
                log.warning("%s batch failed (%s); retrying per post", kind, batch_exc)
                for i, t in enumerate(texts):
                    try:
                        out[i] = model.predict([t], 1)[0]
                    except ModelLoadError:
                        raise
                    except Exception as exc:
                        errs[i] = (errs[i] + f" {kind}: {type(exc).__name__}: {exc}").strip()
        return sents, emos, errs

    # -- single post -------------------------------------------------------------------
    def analyze(self, text: str) -> PostAnalysis:
        """Analyse one post. Missing model -> that part is None with an error message;
        the rule-based parts still work. Nothing is ever invented."""
        clean = normalize_for_model(text)
        if not is_valid_text(clean):
            raise ValueError("Please enter some text (at least a few letters) to analyse.")
        errors: dict[str, str] = {}
        sent = emo = None
        try:
            sent = self.sentiment_model.predict([clean], 1)[0]
        except Exception as exc:
            errors["Sentiment model"] = str(exc)
        try:
            emo = self.emotion_model.predict([clean], 1)[0]
        except Exception as exc:
            errors["Emotion model"] = str(exc)
        return PostAnalysis(text=clean, sentiment=sent, emotion=emo, topic=classify_topic(clean),
                            keywords=extract_keywords(clean), misinfo=screen_text(clean), errors=errors)
