import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import config
from src import charts, ui
from src.keyword_extractor import top_keywords

ui.render_shell(3)
ui.section_title("Topic Insights")
st.caption("Sentiment and emotion come from pretrained models. **Topics are assigned by keyword rules** - "
           "they are rule-based labels, not ground truth.")

results = ui.get_results()
if results.empty or "status" not in results or not (results["status"] == "ok").any():
    ui.no_results_notice()
    st.stop()
ok = results[results["status"] == "ok"]

f1, f2 = st.columns(2)
topics = f1.multiselect("Topics", config.TOPICS, default=[t for t in config.TOPICS if t in set(ok["topic"])])
sents = f2.multiselect("Sentiment", config.SENTIMENT_ORDER, default=config.SENTIMENT_ORDER, format_func=str.title)
view = ok[ok["topic"].isin(topics) & ok["sentiment"].isin(sents)]
st.caption(f"Showing {len(view):,} of {len(ok):,} analysed posts.")
if view.empty:
    st.warning("No posts match these filters.")
    st.stop()

a, b = st.columns(2)
with a:
    st.markdown('<div class="panel-title">Most discussed topics</div>', unsafe_allow_html=True)
    ui.plot(charts.topic_counts(view), width="stretch", key="t_counts")
with b:
    st.markdown('<div class="panel-title">Sentiment across topics</div>', unsafe_allow_html=True)
    ui.plot(charts.topic_sentiment_stacked(view), width="stretch", key="t_sent")

c, d = st.columns(2)
with c:
    st.markdown('<div class="panel-title">Emotions by topic (share of each topic)</div>', unsafe_allow_html=True)
    ui.plot(charts.emotions_by_topic(view), width="stretch", key="t_emo")
with d:
    st.markdown('<div class="panel-title">Frequent keywords</div>', unsafe_allow_html=True)
    kw_topic = st.selectbox("Keywords for topic", ["All selected topics"] + sorted(view["topic"].unique()))
    kview = view if kw_topic == "All selected topics" else view[view["topic"] == kw_topic]
    ui.plot(charts.keywords_bar(top_keywords(kview["text"], 15)), width="stretch", key="t_kw")

trend = charts.sentiment_trend(view)
st.markdown('<div class="panel-title">Trend by date</div>', unsafe_allow_html=True)
if trend is None:
    st.info("No usable date values in the analysed posts, so the trend chart is hidden.")
else:
    ui.plot(trend, width="stretch", key="t_trend")
