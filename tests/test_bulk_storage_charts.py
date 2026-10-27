import io
import urllib.error

import pandas as pd
import pytest
from PIL import Image

import config
from src import charts, ocr, storage, url_fetcher
from src.bulk_processor import run_bulk
from src.storage import PredictionCache, model_signature, to_csv_bytes

POSTS = pd.DataFrame({
    "text": ["I am happy and good", "bad news, I worry about hospitals", "a vaccine rollout update", "lockdown rules changed"],
    "date": pd.to_datetime(["2020-08-20", "2020-08-20", "2020-08-21", "2020-08-22"]),
    "location": ["India", "India", "USA", "USA"],
})


def test_bulk_run_columns_counts_and_rules(tmp_path, fake_analyzer):
    cache = PredictionCache(tmp_path / "c.csv.gz")
    run = run_bulk(POSTS, fake_analyzer, cache, batch_size=2)
    r = run.results
    assert (run.requested, run.newly_analyzed, run.from_cache, run.failed, run.not_processed) == (4, 4, 0, 0, 0)
    assert {"sentiment", "sentiment_confidence", "p_negative", "p_neutral", "p_positive", "emotion", "emotion_score",
            "emo_fear", "topic", "misinfo_flag", "status", "date", "location"} <= set(r.columns)
    assert r.loc[0, "sentiment"] == "positive" and r.loc[1, "sentiment"] == "negative"
    assert r.loc[2, "topic"] == "Vaccination" and r.loc[3, "topic"] == "Lockdown"
    assert (r["status"] == "ok").all()
    assert (r[["p_negative", "p_neutral", "p_positive"]].sum(axis=1) - 1).abs().max() < 1e-9


def test_second_run_uses_persistent_cache(tmp_path, fake_analyzer):
    path = tmp_path / "c.csv.gz"
    run_bulk(POSTS, fake_analyzer, PredictionCache(path), batch_size=2)
    calls = []
    orig = fake_analyzer.sentiment_model.predict
    fake_analyzer.sentiment_model.predict = lambda t, b=32: (calls.append(len(t)), orig(t, b))[1]
    run2 = run_bulk(POSTS, fake_analyzer, PredictionCache(path), batch_size=2)       # fresh object, reads the file
    assert calls == [] and run2.from_cache == 4 and run2.newly_analyzed == 0 and len(run2.results) == 4


def test_cache_is_keyed_by_model_signature(tmp_path, fake_analyzer):
    assert model_signature("a", "b") != model_signature("a", "c")


def test_failed_posts_are_reported_not_invented(tmp_path, fake_analyzer):
    fake_analyzer.sentiment_model.fail_on = ("worry",)
    run = run_bulk(POSTS, fake_analyzer, PredictionCache(tmp_path / "c.csv.gz"), batch_size=4)
    bad = run.results[run.results["status"] == "failed"]
    assert run.failed == 1 and len(bad) == 1 and "worry" in bad.iloc[0]["text"]
    assert pd.isna(bad.iloc[0]["sentiment"]) and "sentiment" in bad.iloc[0]["error"]
    assert (run.results["status"] == "ok").sum() == 3
    # failed posts are not cached, so a later run retries them
    fake_analyzer.sentiment_model.fail_on = ()
    run2 = run_bulk(POSTS, fake_analyzer, PredictionCache(tmp_path / "c.csv.gz"), batch_size=4)
    assert run2.failed == 0 and run2.newly_analyzed == 1 and run2.from_cache == 3


def test_stop_returns_partial_results_and_saves_progress(tmp_path, fake_analyzer):
    stop_after = {"n": 0}

    def should_stop():
        stop_after["n"] += 1
        return stop_after["n"] > 1            # allow exactly one batch
    path = tmp_path / "c.csv.gz"
    run = run_bulk(POSTS, fake_analyzer, PredictionCache(path), batch_size=2, should_stop=should_stop)
    assert run.stopped and len(run.results) == 2 and run.not_processed == 2
    assert len(PredictionCache(path)) == 2


def test_progress_callback_reports_totals(tmp_path, fake_analyzer):
    seen = []
    run_bulk(POSTS, fake_analyzer, PredictionCache(tmp_path / "c.csv.gz"), batch_size=2,
             progress=lambda d, t, r, e, m: seen.append((d, t)))
    assert seen[0] == (0, 4) and seen[-1] == (4, 4)


def test_duplicate_texts_in_input_are_analysed_once(tmp_path, fake_analyzer):
    posts = pd.DataFrame({"text": ["same text good"] * 3})
    run = run_bulk(posts, fake_analyzer, PredictionCache(tmp_path / "c.csv.gz"))
    assert run.newly_analyzed == 1 and len(run.results) == 3


def test_empty_input(tmp_path, fake_analyzer):
    run = run_bulk(pd.DataFrame({"text": []}), fake_analyzer, PredictionCache(tmp_path / "c.csv.gz"))
    assert run.requested == 0 and run.results.empty


def test_single_post_partial_failure_keeps_rule_based_parts(fake_analyzer):
    fake_analyzer.emotion_model.fail_on = ("worry",)
    r = fake_analyzer.analyze("I worry about the hospital")
    assert r.sentiment is not None and r.emotion is None and "Emotion model" in r.errors
    assert r.topic.topic == "Healthcare" and r.keywords


def test_single_post_rejects_empty(fake_analyzer):
    for bad in ["", "  ", "123"]:
        with pytest.raises(ValueError):
            fake_analyzer.analyze(bad)


# ---------------------------------------------------------------- storage / export
def test_results_save_load_and_csv_export_roundtrip(tmp_path, fake_analyzer):
    run = run_bulk(POSTS, fake_analyzer, PredictionCache(tmp_path / "c.csv.gz"))
    path = tmp_path / "latest.csv"
    storage.save_results(run.results, path)
    loaded = storage.load_results(path)
    assert len(loaded) == 4 and loaded["date"].notna().all()
    csv = pd.read_csv(io.BytesIO(to_csv_bytes(loaded)), encoding="utf-8-sig")
    assert list(csv["sentiment"]) == list(run.results["sentiment"])
    assert csv["text"].tolist() == POSTS["text"].tolist()


def test_load_results_missing_or_corrupt(tmp_path):
    assert storage.load_results(tmp_path / "none.csv").empty
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\x00\x01\x02")
    storage.load_results(bad)       # must not raise


def test_corrupt_cache_starts_fresh(tmp_path):
    p = tmp_path / "c.csv.gz"
    p.write_bytes(b"not gzip")
    assert len(PredictionCache(p)) == 0


# ---------------------------------------------------------------- charts
CHARTS = [charts.sentiment_donut, charts.topic_sentiment_stacked, charts.topic_counts,
          charts.emotion_distribution, charts.emotions_by_topic]


@pytest.mark.parametrize("fn", CHARTS)
@pytest.mark.parametrize("data", [None, pd.DataFrame(), pd.DataFrame({"unrelated": [1]}),
                                  pd.DataFrame({"status": ["failed"], "sentiment": [None], "topic": ["General"], "emotion": [None]})])
def test_charts_survive_empty_or_unusable_data(fn, data):
    assert fn(data) is not None


def test_charts_use_real_counts(tmp_path, fake_analyzer):
    r = run_bulk(POSTS, fake_analyzer, PredictionCache(tmp_path / "c.csv.gz")).results
    donut = charts.sentiment_donut(r)
    assert sum(donut.data[0].values) == 4
    assert "4" in donut.layout.annotations[0].text
    tc = charts.topic_counts(r)
    assert dict(zip(tc.data[0].y, tc.data[0].x)) == r["topic"].value_counts().to_dict()


def test_charts_exclude_failed_rows(tmp_path, fake_analyzer):
    fake_analyzer.sentiment_model.fail_on = ("worry",)
    r = run_bulk(POSTS, fake_analyzer, PredictionCache(tmp_path / "c.csv.gz")).results
    assert sum(charts.sentiment_donut(r).data[0].values) == 3


def test_trend_hidden_without_dates_and_present_with_dates(tmp_path, fake_analyzer):
    r = run_bulk(POSTS, fake_analyzer, PredictionCache(tmp_path / "c.csv.gz")).results
    assert charts.sentiment_trend(r) is not None
    assert charts.sentiment_trend(r.drop(columns=["date"])) is None
    assert charts.sentiment_trend(r.assign(date=pd.NaT)) is None


def test_single_post_charts_handle_none():
    assert charts.sentiment_probabilities(None) is not None and charts.emotion_bars({}) is not None
    fig = charts.sentiment_probabilities({"negative": 0.91, "neutral": 0.07, "positive": 0.02})
    assert list(fig.data[0].y) == [0.91, 0.07, 0.02]


# ---------------------------------------------------------------- OCR helpers (no OCR engine needed)
def _img_bytes(fmt="PNG", size=(100, 60)):
    b = io.BytesIO()
    Image.new("RGB", size, "white").save(b, fmt)
    return b.getvalue()


def test_image_validation_accepts_png_jpg():
    assert ocr.validate_image(_img_bytes("PNG")).size == (100, 60)
    assert ocr.validate_image(_img_bytes("JPEG")).size == (100, 60)


@pytest.mark.parametrize("data", [b"", b"this is not an image", _img_bytes("GIF"), _img_bytes("PNG", (5, 5))])
def test_invalid_uploads_rejected(data):
    with pytest.raises(ocr.InvalidImageError):
        ocr.validate_image(data)


def test_oversized_upload_rejected():
    with pytest.raises(ocr.InvalidImageError):
        ocr.validate_image(b"0" * (config.MAX_UPLOAD_MB * 1024 * 1024 + 1))


def test_large_images_are_downscaled():
    img = ocr.validate_image(_img_bytes("PNG", (config.MAX_IMAGE_SIDE * 2, 400)))
    assert max(img.size) == config.MAX_IMAGE_SIDE


def box(x, y, w=80, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_group_into_lines_reading_order():
    boxes = [(box(200, 10), "world", 0.9), (box(10, 12), "hello", 0.9), (box(10, 60), "second", 0.9), (box(120, 62), "line", 0.9)]
    assert ocr.group_into_lines(boxes) == ["hello world", "second line"]
    assert ocr.group_into_lines([]) == []


def test_strip_tweet_chrome():
    lines = ["John Doe @johndoe · Apr 12", "The rise in cases is worrying", "120", "450", "2.1K", "..."]
    assert ocr.strip_tweet_chrome(lines) == ["The rise in cases is worrying"]
    assert ocr.strip_tweet_chrome(["Thanks @bob for the update"]) == ["Thanks @bob for the update"]
    # OCR often drops the middle dot (real case seen on the reference screenshot)
    assert ocr.strip_tweet_chrome(["John Doe @johndoe Apr 12", "Real text here"]) == ["Real text here"]
    assert ocr.strip_tweet_chrome(["Jane @jane_d 5h", "Real text"]) == ["Real text"]


# ---------------------------------------------------------------- URL fetcher (no network used)
def test_url_validation():
    ok = ["https://x.com/user/status/123", "https://twitter.com/u_1/status/99?s=20", "http://mobile.twitter.com/a/status/5"]
    bad = ["", "https://example.com/a/status/1", "https://x.com/user", "not a url"]
    assert all(url_fetcher.is_post_url(u) for u in ok) and not any(url_fetcher.is_post_url(u) for u in bad)


def test_invalid_url_gives_friendly_message_without_network(monkeypatch):
    monkeypatch.setattr(url_fetcher.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network must not be used"))
    r = url_fetcher.fetch_public_post("https://example.com/x")
    assert not r.ok and "paste" in r.message.lower()


def test_network_errors_are_handled(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(url_fetcher.urllib.request, "urlopen", boom)
    r = url_fetcher.fetch_public_post("https://x.com/u/status/1")
    assert not r.ok and "paste" in r.message.lower()

    def nf(*a, **k):
        raise urllib.error.HTTPError("u", 404, "nf", {}, None)
    monkeypatch.setattr(url_fetcher.urllib.request, "urlopen", nf)
    assert "private" in url_fetcher.fetch_public_post("https://x.com/u/status/1").message


def test_parse_oembed_html():
    html_doc = '<blockquote class="twitter-tweet"><p lang="en">Cases are rising &amp; hospitals are full<br>Stay safe <a href="x">pic.twitter.com/abc</a></p>&mdash; A (@a)</blockquote>'
    assert url_fetcher.parse_oembed_html(html_doc) == "Cases are rising & hospitals are full\nStay safe"
    assert url_fetcher.parse_oembed_html("") == ""
