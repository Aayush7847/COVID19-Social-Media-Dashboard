import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import config
from src import data_loader as dl

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = (
    "created_at,tweet,near,clean_tweet,vader_score\n"
    '2020-08-20,"Hello, world! Stay safe",India,hello world stay safe,0.5\n'
    '2020-08-21,"Multi-line\ntweet with a comma, inside",India,multi line tweet,0.0\n'
    "2020-08-21,hello WORLD stay safe!,India,hello world stay safe,0.5\n"      # duplicate
    "2020-08-22,,India,,0.0\n"                                                   # empty text
    "not-a-date,Vaccines are rolling out today,USA,vaccin roll today,0.1\n"
)


def write(tmp_path, name, content, encoding="utf-8-sig"):
    p = tmp_path / name
    p.write_text(content, encoding=encoding)
    return p


def source_for(path):
    return dl.DataSource(path.name, path, (), path.stat().st_size)


def test_loads_quoted_commas_multiline_bom_and_cleans(tmp_path):
    ld = dl.load_dataset(source_for(write(tmp_path, "d.csv", RAW_CSV)))
    assert ld.detection.text == "tweet" and not ld.detection.text_is_processed
    assert ld.stats == {"rows_in_file": 5, "empty_or_invalid": 1, "duplicates": 1, "valid": 3}
    assert "Hello, world! Stay safe" in ld.df["text"].tolist()
    assert any("Multi-line tweet with a comma, inside" == t for t in ld.df["text"])   # newline collapsed
    assert ld.detection.date == "created_at" and ld.detection.location == "near" and ld.detection.platform is None


def test_invalid_dates_become_nat_and_date_usability(tmp_path):
    ld = dl.load_dataset(source_for(write(tmp_path, "d.csv", RAW_CSV)))
    assert ld.df["date"].isna().sum() == 1 and ld.date_usable          # 2 of 3 valid >= 50%
    bad = "created_at,tweet\nx,Some tweet text here\ny,Another tweet text here\n"
    assert not dl.load_dataset(source_for(write(tmp_path, "b.csv", bad))).date_usable


def test_missing_optional_columns(tmp_path):
    ld = dl.load_dataset(source_for(write(tmp_path, "t.csv", "text\nHello there friend\nAnother good post\n")))
    assert list(ld.df.columns) == ["text", "source_row"] and not ld.date_usable


def test_prefers_raw_over_processed_but_falls_back(tmp_path):
    only_clean = "created_at,clean_tweet\n2020-01-01,fan request arang bed posit\n"
    ld = dl.load_dataset(source_for(write(tmp_path, "c.csv", only_clean)))
    assert ld.detection.text == "clean_tweet" and ld.detection.text_is_processed


def test_ids_only_dataset_is_rejected_with_explanation(tmp_path):
    p = write(tmp_path, "ids.csv", "tweet_id,created_at\n1234567890123,2020-01-01\n1234567890124,2020-01-02\n")
    with pytest.raises(dl.NoTextColumnError) as e:
        dl.load_dataset(source_for(p))
    assert "IDs" in str(e.value) or "ids" in str(e.value).lower()


def test_unknown_column_name_detected_by_heuristic(tmp_path):
    p = write(tmp_path, "u.csv", "id,whatever\n1,This is a fairly long free text post about vaccines\n2,Another reasonably long post about lockdown rules\n")
    assert dl.load_dataset(source_for(p)).detection.text == "whatever"


def test_ambiguous_columns_raise_with_candidates(tmp_path):
    row = "This is a fairly long free text post about vaccines today"
    p = write(tmp_path, "a.csv", f"col_a,col_b\n{row},{row}x\n{row},{row}y\n")
    with pytest.raises(dl.AmbiguousTextColumnError) as e:
        dl.load_dataset(source_for(p))
    assert set(e.value.candidates) == {"col_a", "col_b"}


def test_manual_override(tmp_path):
    p = write(tmp_path, "o.csv", "a,b\nlong enough text number one,other text value here\nlong enough text number two,other text value 2\n")
    ld = dl.load_dataset(source_for(p), text_column="b")
    assert ld.detection.text == "b" and ld.df["text"].iloc[0].startswith("other text")
    with pytest.raises(dl.DatasetError):
        dl.load_dataset(source_for(p), text_column="nope")


def test_latin1_bytes_do_not_crash(tmp_path):
    p = tmp_path / "l.csv"
    p.write_bytes("tweet\nCaf\xe9 is open again\n".encode("latin-1"))
    ld = dl.load_dataset(source_for(p))
    assert len(ld.df) == 1


def test_zip_and_nested_zip_discovery_and_loading(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("deep/tweets.csv", "tweet\nNested zip post one here\nNested zip post two here\n")
    with zipfile.ZipFile(tmp_path / "outer.zip", "w") as z:
        z.writestr("readme.txt", "ignore me")        # .txt is a supported suffix, so it is listed too
        z.writestr("inner.zip", inner.getvalue())
        z.writestr("top.csv", "text\nTop level post one\n")
    srcs = {s.label.split("  ")[0]: s for s in dl.discover_sources(tmp_path)}
    assert "tweets.csv" in srcs and "top.csv" in srcs
    assert len(dl.load_dataset(srcs["tweets.csv"]).df) == 2
    assert len(dl.load_dataset(srcs["top.csv"]).df) == 1


def test_json_and_jsonl(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"text": "First json post here", "date": "2021-01-01"}\n{"text": "Second json post here", "date": "2021-01-02"}\n')
    ld = dl.load_dataset(source_for(tmp_path / "a.jsonl"))
    assert len(ld.df) == 2 and ld.date_usable
    (tmp_path / "b.json").write_text('[{"full_text": "A json array post"}, {"full_text": "Second array post"}]')
    assert len(dl.load_dataset(source_for(tmp_path / "b.json")).df) == 2


def test_empty_file(tmp_path):
    p = write(tmp_path, "e.csv", "tweet\n")
    with pytest.raises(dl.DatasetError):
        dl.load_dataset(source_for(p))


def test_sampling_is_reproducible_and_exact():
    df = pd.DataFrame({"text": [f"post number {i}" for i in range(100)]})
    a, b = dl.sample_dataset(df, 10, "random", 1), dl.sample_dataset(df, 10, "random", 1)
    assert a.equals(b) and len(a) == 10
    assert dl.sample_dataset(df, 5, "first")["text"].tolist() == df["text"].head(5).tolist()
    assert len(dl.sample_dataset(df, 0)) == 100 and len(dl.sample_dataset(df, 1000)) == 100


BUNDLED = config.DATA_DIR / "Extended_Covid_Twitter_Datasets.zip"


@pytest.mark.skipif(not BUNDLED.exists(), reason="bundled dataset not present")
def test_bundled_usa_file():
    src = next(s for s in dl.discover_sources() if "only_USA" in s.label)
    ld = dl.load_dataset(src)
    assert ld.detection.text == "tweet" and not ld.detection.text_is_processed
    assert ld.stats["rows_in_file"] == 3000 and ld.stats["valid"] == len(ld.df) > 2900
    assert ld.date_usable and ld.detection.location == "near"


@pytest.mark.skipif(not BUNDLED.exists(), reason="bundled dataset not present")
def test_bundled_global_file_is_flagged_as_processed():
    src = next(s for s in dl.discover_sources() if "global" in s.label)
    _, det = dl.inspect_source(src)
    assert det.text == "clean_tweet" and det.text_is_processed
