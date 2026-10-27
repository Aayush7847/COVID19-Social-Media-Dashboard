import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import config
from src import charts, data_loader, ui
from src.bulk_processor import run_bulk
from src.keyword_extractor import top_keywords
from src.model_utils import ModelLoadError
from src.preprocessing import text_hash
from src.storage import PredictionCache, model_signature, save_results, to_csv_bytes

ui.render_shell(2)
ui.section_title("Bulk Analysis")
st.caption("Analyse many posts from your dataset with the same models used for single posts. "
           "Results are saved locally in the outputs/ folder.")


@st.cache_resource(show_spinner=False)
def get_cache() -> PredictionCache:
    return PredictionCache()


@st.cache_resource(show_spinner="Reading and cleaning the dataset (large files can take a few seconds)...")
def load_cached(_source: data_loader.DataSource, source_key: str, text_col: str | None):
    return data_loader.load_dataset(_source, text_col)


@st.cache_data(show_spinner=False)
def inspect_cached(source_key: str, _source: data_loader.DataSource, text_col: str | None):
    try:
        sample, det = data_loader.inspect_source(_source, text_col)
        return {"ok": True, "det": det, "cols": list(sample.columns)}
    except data_loader.AmbiguousTextColumnError as exc:
        return {"ok": False, "error": str(exc), "candidates": exc.candidates, "ambiguous": True}
    except data_loader.DatasetError as exc:
        return {"ok": False, "error": str(exc), "ambiguous": False}
    except Exception as exc:
        return {"ok": False, "error": f"Could not read this file: {type(exc).__name__}: {exc}", "ambiguous": False}


# ------------------------------------------------------------------ 1. choose dataset
sources = data_loader.discover_sources()
with st.expander("Add or replace a dataset file", expanded=not sources):
    st.write(f"Put a `.csv`, `.json`, `.jsonl` or `.zip` file into the project's **data/** folder, or upload it here. "
             f"Folder in use: `{config.DATA_DIR}`")
    up = st.file_uploader("Upload dataset", type=["csv", "tsv", "json", "jsonl", "zip"], key="ds_upload")
    if up is not None and st.button("Save to data/ folder"):
        safe = Path(up.name).name.replace(" ", "_")
        (config.DATA_DIR / safe).write_bytes(up.getvalue())
        st.success(f"Saved {safe}. Reloading...")
        st.rerun()
if not sources:
    st.warning("No dataset found in the data/ folder yet - add one above.")
    st.stop()

# Prefer a source with a raw text column as the default (checked cheaply on 500 rows).
infos = {s.key: inspect_cached(s.key, s, None) for s in sources}
order = sorted(sources, key=lambda s: (not (infos[s.key]["ok"] and not infos[s.key]["det"].text_is_processed), s.size_bytes))
label_map = {s.label: s for s in order}
choice = st.selectbox("Dataset file", list(label_map), help="Files found in the data/ folder (including inside ZIP files).")
source = label_map[choice]
info = infos[source.key]

text_col = None
if not info["ok"]:
    st.error(info["error"])
    if info.get("ambiguous"):
        text_col = st.selectbox("Text column", info["candidates"])
        info = inspect_cached(source.key, source, text_col)
        if not info["ok"]:
            st.stop()
    else:
        st.stop()
det = info["det"]
all_cols = info["cols"]
with st.expander("Column detection", expanded=False):
    st.write(det.reason)
    st.write(f"**Text:** `{det.text}` · **Date:** `{det.date}` · **Location:** `{det.location}` · **Platform:** `{det.platform}`")
    pick = st.selectbox("Override the text column", ["(auto)"] + [str(c) for c in all_cols])
    if pick != "(auto)":
        text_col = pick
        info = inspect_cached(source.key, source, text_col)
        det = info["det"]

if det.text_is_processed:
    st.warning("**This file has no original post text.** It only contains the pre-processed column "
               f"`{det.text}` (lower-cased, stop-words removed, words stemmed - e.g. “posit”, “comun”). "
               "The language models were trained on normal tweets, so predictions on this text are "
               "**less reliable**, and the topic keywords match less often. For the best results use a file "
               "with a raw text column (like sample_data_only_USA.csv).")

loaded = load_cached(source, source.key, text_col)
st.session_state["_loaded_meta"] = loaded.stats
s = loaded.stats
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows in source file", f"{s['rows_in_file']:,}")
m2.metric("Empty / invalid removed", f"{s['empty_or_invalid']:,}")
m3.metric("Duplicates removed", f"{s['duplicates']:,}")
m4.metric("Valid unique posts available", f"{s['valid']:,}")
if loaded.df.empty:
    st.error("No usable posts were found in this file.")
    st.stop()

st.markdown("**Preview (first 15 valid posts)**")
st.dataframe(loaded.df.drop(columns=["source_row"]).head(15), hide_index=True, width="stretch")
if not loaded.date_usable:
    st.info("No usable date column was found, so the time-trend chart will be hidden.")

# ------------------------------------------------------------------ 2. settings
ui.section_title("Run settings")
valid_total = len(loaded.df)
default_n = valid_total if config.DEFAULT_PROCESS_LIMIT <= 0 else min(config.DEFAULT_PROCESS_LIMIT, valid_total)
c1, c2, c3 = st.columns(3)
limit = c1.number_input("Posts to analyse", min_value=1, max_value=valid_total, value=int(default_n), step=100,
                        help="Start small. The default can be changed in config.py (DEFAULT_PROCESS_LIMIT).")
mode = c2.radio("Which posts?", ["random", "first"], horizontal=True,
                help="'random' is a reproducible random sample; 'first' takes the first posts in the file.")
batch = c3.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=config.BATCH_SIZE if config.BATCH_SIZE in (8, 16, 32, 64, 128) else 32)

sample = data_loader.sample_dataset(loaded.df, int(limit), mode)
analyzer = ui.get_analyzer()
cache = get_cache()
sig = model_signature(*analyzer.signature_parts)
keys = [f"{sig}:{text_hash(t)}" for t in sample["text"]]
n_cached = int(sum(cache.has(keys)))
st.caption(f"{len(sample):,} posts selected · {n_cached:,} already analysed earlier (will be reused from the saved cache) · "
           f"{len(sample) - n_cached:,} to analyse. Speed depends on your CPU - the live speed and time left are shown while "
           "it runs. If you need to cancel, press **Stop** at the top right; finished batches are already saved.")
if len(sample) - n_cached > 5000:
    st.warning("That is a lot of new posts for a laptop CPU and may take a long time. Consider a smaller sample first.")

b1, b2 = st.columns([1, 4])
start = b1.button("▶ Start analysis", type="primary", width="stretch")
if b2.button("Clear saved prediction cache"):
    cache.clear()
    st.success("Prediction cache cleared.")
    st.rerun()

# ------------------------------------------------------------------ 3. run
if start:
    bar = st.progress(0.0)
    status = st.empty()

    def progress(done, total, rate, eta, msg):
        bar.progress(min(done / total, 1.0) if total else 1.0)
        extra = f" · {rate:.1f} posts/s · about {int(eta // 60)} min {int(eta % 60)} s left" if rate else ""
        status.write(msg + extra)

    try:
        run = run_bulk(sample, analyzer, cache, batch_size=int(batch), progress=progress)
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()
    bar.progress(1.0)
    if len(run.results):
        save_results(run.results)
        st.cache_data.clear()
    st.session_state["_last_run"] = {k: getattr(run, k) for k in
                                     ("requested", "from_cache", "newly_analyzed", "failed", "not_processed", "seconds", "stopped")}
    status.empty()

# ------------------------------------------------------------------ 4. results
results = ui.get_results()
run_info = st.session_state.get("_last_run")
ui.section_title("Results")
if results.empty:
    ui.no_results_notice()
    st.stop()

ok = results[results["status"] == "ok"] if "status" in results else results
failed = int((results["status"] == "failed").sum()) if "status" in results else 0
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Available in source", f"{valid_total:,}")
k2.metric("Selected for the run", f"{run_info['requested']:,}" if run_info else f"{len(results):,}")
k3.metric("Analysed successfully", f"{len(ok):,}")
k4.metric("Failed", f"{failed:,}")
k5.metric("Shown in charts", f"{len(ok):,}")
if run_info:
    note = f"Last run took {run_info['seconds']:.1f} s ({run_info['newly_analyzed']:,} new, {run_info['from_cache']:,} from cache)."
    if run_info["not_processed"]:
        note += f" {run_info['not_processed']:,} selected posts were not processed."
    st.caption(note)
if failed:
    with st.expander(f"{failed} posts failed"):
        st.dataframe(results[results["status"] == "failed"][["text", "error"]], hide_index=True, width="stretch")

a, b, c = st.columns(3)
with a:
    st.markdown('<div class="panel-title">Sentiment distribution</div>', unsafe_allow_html=True)
    ui.plot(charts.sentiment_donut(results), width="stretch", key="b_donut")
with b:
    st.markdown('<div class="panel-title">Emotion distribution</div>', unsafe_allow_html=True)
    ui.plot(charts.emotion_distribution(results), width="stretch", key="b_emo")
with c:
    st.markdown('<div class="panel-title">Topic distribution (rule-based)</div>', unsafe_allow_html=True)
    ui.plot(charts.topic_counts(results), width="stretch", key="b_topic")
d, e = st.columns(2)
with d:
    st.markdown('<div class="panel-title">Topic-wise sentiment</div>', unsafe_allow_html=True)
    ui.plot(charts.topic_sentiment_stacked(results), width="stretch", key="b_topsent")
with e:
    st.markdown('<div class="panel-title">Frequent keywords</div>', unsafe_allow_html=True)
    ui.plot(charts.keywords_bar(top_keywords(ok["text"], 15)), width="stretch", key="b_kw")
trend = charts.sentiment_trend(results)
if trend is not None:
    st.markdown('<div class="panel-title">Sentiment over time</div>', unsafe_allow_html=True)
    ui.plot(trend, width="stretch", key="b_trend")
else:
    st.info("Sentiment-over-time chart hidden: the analysed posts have no usable date values.")

st.download_button("⬇ Download processed results (CSV)", to_csv_bytes(results), "covid_social_media_results.csv",
                   "text/csv", type="primary")
