"""Executes every Streamlit page headlessly (no browser) and checks nothing raises."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config

PAGES = ["pages/1_Dashboard.py", "pages/2_Analyze_Post.py", "pages/3_Bulk_Analysis.py",
         "pages/4_Topic_Insights.py", "pages/5_All_Results.py", "pages/6_About_Project.py"]


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    """Never touch the user's real outputs/ folder while testing (defaults are bound at import time)."""
    import src.storage as storage
    latest = tmp_path / "latest.csv"
    for fn in (storage.save_results, storage.load_results, storage.results_mtime):
        monkeypatch.setattr(fn, "__defaults__", (latest,))
    monkeypatch.setattr(storage.PredictionCache.__init__, "__defaults__", (tmp_path / "cache.csv.gz",))
    yield


def run(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_results(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_analyze_empty_text_shows_warning_not_crash():
    at = run(PAGES[1])
    at.button(key="an_analyze").click().run()
    assert not at.exception and any("enter some text" in w.value.lower() for w in at.warning)


def test_analyze_without_models_shows_helpful_error(monkeypatch):
    """In an offline/no-torch environment the user must get a message, not a traceback."""
    at = run(PAGES[1])
    at.session_state["an_text"] = "The rise in COVID-19 cases is really worrying."
    at.session_state["_post_text"] = at.session_state["an_text"]
    at.button(key="an_analyze").click().run()
    assert not at.exception
    # Either models work (user machine) or we get a friendly warning about the unavailable models
    assert at.session_state["_post_result"] is not None
    res = at.session_state["_post_result"]
    assert res.topic.topic == "Healthcare"
    if res.sentiment is None:
        assert any("unavailable" in w.value.lower() for w in at.warning)


def test_pages_render_with_results(tmp_path, fake_analyzer):
    from src.bulk_processor import run_bulk
    from src.storage import PredictionCache, save_results
    posts = pd.DataFrame({"text": ["good vaccine news", "bad lockdown worry", "market jobs recession", "a lovely day"] * 3,
                          "date": pd.to_datetime(["2020-08-20", "2020-08-21", "2020-08-22", "2020-08-23"] * 3),
                          "location": ["USA"] * 12})
    posts["text"] = [f"{t} {i}" for i, t in enumerate(posts["text"])]
    res = run_bulk(posts, fake_analyzer, PredictionCache(tmp_path / "x.csv.gz")).results
    save_results(res, tmp_path / "latest.csv")
    for page in PAGES:
        at = run(page)
        assert not at.exception, (page, [e.value for e in at.exception])


@pytest.mark.parametrize("page", [PAGES[0], PAGES[1]])
def test_result_cards_render_with_full_result(page, fake_analyzer):
    """Success path: sentiment + emotion + topic + keywords all present."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.session_state["_post_result"] = fake_analyzer.analyze("I worry about the hospital, bad news")
    at.run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Negative" in html and "Fear" in html and "Healthcare" in html and "Confidence: 85%" in html
    assert "Not Detected" in html and "not a fact-checker" in html
