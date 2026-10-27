import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src import charts, ui

ui.render_shell(0)
ui.render_analyzer("dash")

results = ui.get_results()
n_ok = int((results["status"] == "ok").sum()) if len(results) and "status" in results else 0
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
if n_ok == 0:
    ui.no_results_notice()
else:
    st.caption(f"Dataset view: {n_ok:,} analysed posts from the last Bulk Analysis run.")

c1, c2, c3 = st.columns([0.9, 1.25, 1.0])
with c1:
    st.markdown('<div class="panel-title">Sentiment Distribution (Dataset View)</div>', unsafe_allow_html=True)
    ui.plot(charts.sentiment_donut(results), width="stretch", key="d_donut")
with c2:
    st.markdown('<div class="panel-title">Topic-wise Sentiment</div>', unsafe_allow_html=True)
    ui.plot(charts.topic_sentiment_stacked(results), width="stretch", key="d_topsent")
with c3:
    st.markdown('<div class="panel-title">Top Discussed Topics (All Posts)</div>', unsafe_allow_html=True)
    ui.plot(charts.topic_counts(results), width="stretch", key="d_topcount")
