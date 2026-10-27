import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emotion import EmotionResult  # noqa: E402
from src.sentiment import SentimentResult  # noqa: E402


class FakeSentiment:
    """Deterministic stand-in for the real model: lets us test everything AROUND the model.
    Its outputs are NOT real predictions and are used only inside tests."""
    name = "fake-sentiment"
    fail_on = ()

    def predict(self, texts, batch_size=32):
        out = []
        for t in texts:
            if any(f in t for f in self.fail_on):
                raise RuntimeError("boom")
            if "good" in t or "happy" in t:
                p = {"negative": 0.05, "neutral": 0.15, "positive": 0.80}
            elif "bad" in t or "worry" in t:
                p = {"negative": 0.85, "neutral": 0.10, "positive": 0.05}
            else:
                p = {"negative": 0.2, "neutral": 0.6, "positive": 0.2}
            top = max(p, key=p.get)
            out.append(SentimentResult(top, p[top], p))
        return out


class FakeEmotion:
    name = "fake-emotion"
    labels = ["anger", "fear", "joy", "neutral"]
    fail_on = ()

    def predict(self, texts, batch_size=32):
        out = []
        for t in texts:
            if any(f in t for f in self.fail_on):
                raise RuntimeError("boom")
            s = {"anger": 0.05, "fear": 0.7 if "worry" in t else 0.1, "joy": 0.8 if "happy" in t else 0.05, "neutral": 0.2}
            tot = sum(s.values())
            s = {k: v / tot for k, v in s.items()}
            top = max(s, key=s.get)
            out.append(EmotionResult(top, s[top], s))
        return out


@pytest.fixture
def fake_analyzer():
    from src.analyzer import PostAnalyzer
    return PostAnalyzer(FakeSentiment(), FakeEmotion())
