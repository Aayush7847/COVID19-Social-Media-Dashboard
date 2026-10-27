"""Exercises the REAL HFTextClassifier / SentimentModel / EmotionModel code (tokenizer, torch
forward pass, softmax, label mapping, batching) with a tiny randomly-initialised local model.

The weights are random, so the PREDICTIONS are meaningless - these tests only prove the plumbing.
Skipped automatically when torch/transformers are not installed.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.emotion import EmotionModel  # noqa: E402
from src.model_utils import ModelLoadError  # noqa: E402
from src.sentiment import SentimentModel  # noqa: E402

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "covid", "cases", "are", "rising", "good", "news", "hospital", "the", "."]


def make_model(path, id2label):
    (path).mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(VOCAB))
    tok = transformers.BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
    cfg = transformers.BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, num_labels=len(id2label), id2label=id2label,
                                  label2id={v: k for k, v in id2label.items()})
    torch.manual_seed(0)
    transformers.BertForSequenceClassification(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_sentiment_model_end_to_end(tmp_path):
    m = SentimentModel(make_model(tmp_path / "s", {0: "negative", 1: "neutral", 2: "positive"}))
    res = m.predict(["covid cases are rising", "good news", "the hospital ."], batch_size=2)
    assert len(res) == 3
    for r in res:
        assert r.label in {"negative", "neutral", "positive"}
        assert sum(r.probs.values()) == pytest.approx(1.0)
        assert r.confidence == pytest.approx(max(r.probs.values()))


def test_batch_results_equal_single_results_and_keep_order(tmp_path):
    m = SentimentModel(make_model(tmp_path / "s", {0: "negative", 1: "neutral", 2: "positive"}))
    texts = ["good news", "covid cases are rising rising rising", "the .", "hospital"]
    batched = m.predict(texts, batch_size=3)
    singles = [m.predict([t], 1)[0] for t in texts]
    for b, s in zip(batched, singles):
        assert b.label == s.label
        for k in b.probs:
            assert b.probs[k] == pytest.approx(s.probs[k], abs=1e-5)


def test_emotion_model_reads_labels_from_config(tmp_path):
    labels = {0: "anger", 1: "fear", 2: "joy", 3: "sadness"}
    m = EmotionModel(make_model(tmp_path / "e", labels))
    r = m.predict(["covid cases are rising"])[0]
    assert set(r.scores) == set(labels.values()) and r.label in labels.values()
    assert m.labels == ["anger", "fear", "joy", "sadness"]


def test_empty_and_long_text_do_not_crash(tmp_path):
    m = SentimentModel(make_model(tmp_path / "s", {0: "negative", 1: "neutral", 2: "positive"}))
    assert len(m.predict(["", "good " * 500])) == 2
    assert m.predict([]) == []


def test_missing_model_gives_friendly_error():
    with pytest.raises(ModelLoadError) as e:
        SentimentModel("definitely/not-a-real-model-xyz").predict(["hello"])
    assert "internet" in str(e.value).lower()
