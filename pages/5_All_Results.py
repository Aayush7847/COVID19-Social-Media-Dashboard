import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import config
from src import ui
from src.storage import to_csv_bytes

ui.render_shell(4)
ui.section_title("All Results")

results = ui.get_results()
if results.empty:
    ui.no_results_notice()
    st.stop()

f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
query = f1.text_input("Search text", placeholder="e.g. vaccine")
sent = f2.multiselect("Sentiment", config.SENTIMENT_ORDER, format_func=str.title)
topic = f3.multiselect("Topic", config.TOPICS)
emo_options = sorted(results["emotion"].dropna().unique()) if "emotion" in results else []
emo = f4.multiselect("Emotion", emo_options, format_func=str.title)

df = results
if query.strip():
    df = df[df["text"].str.contains(query.strip(), case=False, regex=False, na=False)]
if sent:
    df = df[df["sentiment"].isin(sent)]
if topic:
    df = df[df["topic"].isin(topic)]
if emo:
    df = df[df["emotion"].isin(emo)]

p1, p2, p3 = st.columns([1, 1, 3])
page_size = p1.selectbox("Rows per page", [25, 50, 100, 250], index=1)
n_pages = max(1, math.ceil(len(df) / page_size))
page = p2.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
p3.markdown(f"<div style='padding-top:1.9rem'>{len(df):,} of {len(results):,} posts match · page {page} of {n_pages}</div>",
            unsafe_allow_html=True)

cols = [c for c in ["text", "sentiment", "sentiment_confidence", "p_negative", "p_neutral", "p_positive", "emotion",
                    "emotion_score", "topic", "misinfo_flag", "date", "location", "platform", "status", "error"] if c in df]
view = df.iloc[(page - 1) * page_size: page * page_size][cols]
st.dataframe(view, hide_index=True, width="stretch", height=560, column_config={
    "text": st.column_config.TextColumn("Original text", width="large"),
    "sentiment": "Sentiment",
    "sentiment_confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
    "p_negative": st.column_config.NumberColumn("P(neg)", format="%.3f"),
    "p_neutral": st.column_config.NumberColumn("P(neu)", format="%.3f"),
    "p_positive": st.column_config.NumberColumn("P(pos)", format="%.3f"),
    "emotion": "Emotion",
    "emotion_score": st.column_config.NumberColumn("Emotion score", format="%.2f"),
    "topic": "Topic (rule-based)",
    "misinfo_flag": st.column_config.CheckboxColumn("Review flag (preliminary)"),
    "date": st.column_config.DateColumn("Date"),
})
st.download_button(f"⬇ Download the {len(df):,} filtered rows (CSV)", to_csv_bytes(df), "covid_filtered_results.csv",
                   "text/csv", type="primary")
st.caption("The 'Review flag' is a preliminary keyword screen, not fact-checking. Topics are keyword-rule labels, not ground truth.")
