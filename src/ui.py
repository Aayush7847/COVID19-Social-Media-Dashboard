"""Streamlit presentation layer: theme, header, sidebar, cards and the analyzer panel.

No model or data logic lives here - it only calls the modules in src/.
"""
from __future__ import annotations

import hashlib
import html
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from src import charts, ocr, storage, url_fetcher
from src.analyzer import PostAnalysis, PostAnalyzer
from src.misinformation import DISCLAIMER
from src.model_utils import ModelLoadError
from src.topic_classifier import TOPIC_DESCRIPTIONS

C = config.COLORS

NAV = [  # (file, label, material icon)
    ("pages/1_Dashboard.py", "Dashboard", ":material/home:"),
    ("pages/2_Analyze_Post.py", "Analyze Post", ":material/image_search:"),
    ("pages/3_Bulk_Analysis.py", "Bulk Analysis", ":material/bar_chart:"),
    ("pages/4_Topic_Insights.py", "Topic Insights", ":material/stacked_line_chart:"),
    ("pages/5_All_Results.py", "All Results", ":material/table_chart:"),
    ("pages/6_About_Project.py", "About Project", ":material/info:"),
]

# --------------------------------------------------------------------------------------
# Icons (inline SVG so the app needs no image files and works offline)
# --------------------------------------------------------------------------------------
def virus_svg(size: int = 44, color: str = "#2d8cff", spikes: int = 12) -> str:
    import math
    parts = []
    for i in range(spikes):
        a = 2 * math.pi * i / spikes
        x1, y1 = 50 + 30 * math.cos(a), 50 + 30 * math.sin(a)
        x2, y2 = 50 + 41 * math.cos(a), 50 + 41 * math.sin(a)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>'
                     f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="5" fill="{color}"/>')
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}'
            f'<circle cx="50" cy="50" r="31" fill="{color}"/><circle cx="42" cy="42" r="6" fill="#bfe0ff" opacity=".8"/>'
            f'<circle cx="58" cy="55" r="4" fill="#bfe0ff" opacity=".7"/><circle cx="46" cy="62" r="3" fill="#bfe0ff" opacity=".6"/></svg>')


def _face_svg(kind: str) -> str:
    col = {"negative": C["negative"], "neutral": C["neutral"], "positive": C["positive"]}.get(kind, "#94a3b8")
    mouth = {"negative": "M30 66 Q50 48 70 66", "neutral": "M32 62 L68 62", "positive": "M30 58 Q50 78 70 58"}.get(kind, "M32 62 L68 62")
    return (f'<svg width="64" height="64" viewBox="0 0 100 100"><circle cx="50" cy="50" r="46" fill="{col}"/>'
            f'<circle cx="35" cy="40" r="5" fill="#fff"/><circle cx="65" cy="40" r="5" fill="#fff"/>'
            f'<path d="{mouth}" stroke="#fff" stroke-width="7" fill="none" stroke-linecap="round"/></svg>')


WARN_SVG = ('<svg width="64" height="64" viewBox="0 0 100 100"><path d="M50 10 L92 86 H8 Z" fill="#fff3e0" stroke="#f97316" '
            'stroke-width="7" stroke-linejoin="round"/><rect x="46" y="38" width="8" height="26" rx="3" fill="#f97316"/>'
            '<circle cx="50" cy="73" r="5" fill="#f97316"/></svg>')
EMOTION_EMOJI = {"fear": "😨", "sadness": "😢", "anger": "😠", "joy": "😄", "surprise": "😲", "disgust": "🤢",
                 "neutral": "😐", "optimism": "🙂", "love": "😍"}
TOPIC_EMOJI = {"Healthcare": "🏥", "Vaccination": "💉", "Lockdown": "🏠", "Mental Health": "🧠",
               "Economy": "💼", "General": "💬"}

# --------------------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------------------
CSS = f"""
<style>
.stApp {{ background:#f4f7fb; }}
header[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {{ visibility:hidden; height:0; }}
.block-container {{ padding:0 1.6rem 2.5rem 1.6rem !important; max-width:none !important; }}

/* ---- header ---- */
.st-key-app_header {{ background:linear-gradient(90deg,{C['navy_dark']},{C['navy']}); margin:-1rem -1.6rem 1.2rem -1.6rem !important; width:calc(100% + 3.2rem) !important; max-width:none !important;
  height:auto !important; min-height:96px; overflow:visible !important; box-sizing:border-box;
  padding:1rem 1.8rem 1rem 1.8rem !important; color:#fff; border-bottom:1px solid #123a6b; }}
.st-key-app_header * {{ color:#fff; }}
.app-title {{ font-size:1.5rem; font-weight:700; line-height:1.15; margin:0; }}
.app-sub {{ font-size:1rem; color:#c7d7ee !important; margin-top:.15rem; }}
.app-desc {{ font-size:.95rem; color:#dbe7f7 !important; text-align:right; padding-top:.6rem; }}
.app-clock {{ border-left:1px solid #2c4a75; padding-left:1rem; font-size:.9rem; line-height:1.5; padding-top:.2rem; }}
.title-wrap {{ display:flex; align-items:center; gap:.9rem; }}

/* ---- sidebar ---- */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"], [data-testid="stSidebarHeader"] {{ background:linear-gradient(180deg,{C['navy_dark']},{C['navy']}) !important; border-right:0; }}
[data-testid="stSidebarHeader"] {{ height:0; min-height:0; padding:0; }}
section[data-testid="stSidebar"] * {{ color:#e6eefb; }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{ border-radius:10px; padding:.62rem .8rem; margin-bottom:.2rem; }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{ background:rgba(255,255,255,.08); }}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p {{ font-size:1rem; font-weight:500; }}
.side-card {{ margin-top:1.4rem; border:1px solid #1d3f6f; border-radius:12px; padding:.9rem; background:rgba(255,255,255,.04); }}
.side-card h4 {{ margin:.5rem 0 0 0; font-size:1.05rem; }}
.side-card .s1 {{ font-weight:600; font-size:.9rem; }}
.side-card p {{ font-size:.8rem; color:#b9c9e3 !important; margin:.35rem 0 0 0; }}

/* ---- section headings & cards ---- */
.sec-title {{ font-size:1.2rem; font-weight:700; color:#0f172a; margin:.2rem 0 .5rem 0; }}
.panel {{ background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:1rem 1.1rem; }}
.panel-title {{ font-weight:700; color:#0f172a; margin-bottom:.4rem; }}
.rcard {{ display:flex; align-items:center; gap:1rem; border-radius:14px; padding:1rem 1.2rem; min-height:132px; border:1px solid; }}
.rcard .lbl {{ font-size:.95rem; font-weight:600; }}
.rcard .val {{ font-size:2rem; font-weight:800; line-height:1.1; }}
.rcard .sub {{ font-size:.85rem; color:#475569; }}
.rc-neg {{ background:#fdecec; border-color:#f8cfcf; }} .rc-neg .lbl,.rc-neg .val {{ color:{C['negative']}; }}
.rc-neu {{ background:#eaf2fe; border-color:#cfe0fb; }} .rc-neu .lbl,.rc-neu .val {{ color:{C['neutral']}; }}
.rc-pos {{ background:#e8f7ee; border-color:#c3ebd2; }} .rc-pos .lbl,.rc-pos .val {{ color:{C['positive']}; }}
.rc-emo {{ background:#f1edff; border-color:#ddd3ff; }} .rc-emo .lbl {{ color:#334155; }} .rc-emo .val {{ color:{C['emotion']}; }}
.rc-top {{ background:#e9f7f4; border-color:#c4e9e2; }} .rc-top .lbl {{ color:{C['topic']}; }} .rc-top .val {{ color:{C['topic']}; }}
.rc-mis {{ background:#fff3e8; border-color:#fbd9bb; }} .rc-mis .lbl {{ color:{C['warning']}; }} .rc-mis .val {{ color:{C['warning']}; }}
.rc-idle {{ background:#f8fafc; border-color:#e2e8f0; }} .rc-idle .val {{ color:#94a3b8; }} .rc-idle .lbl {{ color:#64748b; }}
.emoji-circle {{ width:64px; height:64px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2rem; flex:none; }}
.chip {{ display:inline-block; background:#fde8e8; color:#c62828; border-radius:999px; padding:.3rem .8rem; margin:0 .35rem .45rem 0; font-size:.9rem; }}
.shot-ph {{ border:2px dashed #cbd5e1; border-radius:12px; height:210px; display:flex; align-items:center; justify-content:center; color:#94a3b8; text-align:center; }}
.note {{ font-size:.8rem; color:#64748b; }}

/* ---- widgets ---- */
textarea:disabled {{ color:#0f172a !important; -webkit-text-fill-color:#0f172a !important; opacity:1 !important; }}
.st-key-dash_upload [data-testid="stFileUploaderDropzoneInstructions"] span, .st-key-an_upload [data-testid="stFileUploaderDropzoneInstructions"] span {{ display:none; }}
.st-key-dash_upload [data-testid="stFileUploaderDropzoneInstructions"]::after, .st-key-an_upload [data-testid="stFileUploaderDropzoneInstructions"]::after {{ content:"PNG, JPG or JPEG - max {config.MAX_UPLOAD_MB} MB"; font-size:.85rem; color:#64748b; }}
.stAppDeployButton {{ display:none; }}
.stButton button[kind="primary"] {{ background:{C['blue']}; border:0; font-weight:700; font-size:1.05rem; padding:.7rem 1rem; border-radius:10px; }}
.stButton button[kind="primary"]:hover {{ background:#1558c4; }}
[data-testid="stFileUploaderDropzone"] {{ border:2px dashed #b9c8de; background:#fbfdff; border-radius:12px; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:.7rem 1rem; }}
div[data-testid="stPlotlyChart"] {{ background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:.4rem .6rem; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Shell: header + sidebar
# --------------------------------------------------------------------------------------
@st.fragment(run_every=30)
def _clock() -> None:
    now = datetime.now()
    st.markdown(f'<div class="app-clock">{now.strftime("%b %d, %Y")}<br>{now.strftime("%I:%M %p")}</div>',
                unsafe_allow_html=True)


def render_shell(active: int = 0) -> None:
    """Header + sidebar. Call once at the top of every page (`active` = index in NAV)."""
    apply_theme()
    st.markdown(f'<style>.st-key-nav_{active} a[data-testid="stPageLink-NavLink"] {{ background:{C["blue"]} !important; }}</style>', unsafe_allow_html=True)
    with st.container(key="app_header"):
        c1, c2, c3 = st.columns([0.60, 0.27, 0.13], vertical_alignment="center")
        c1.markdown(
            f'<div class="title-wrap">{virus_svg(52)}<div><div class="app-title">COVID-19 Social Media Intelligence Dashboard</div>'
            f'<div class="app-sub">Sentiment Analysis · Emotion Detection · Topic Classification</div></div></div>',
            unsafe_allow_html=True)
        c2.markdown('<div class="app-desc">Analyze and understand public discussions about COVID-19</div>',
                    unsafe_allow_html=True)
        with c3:
            _clock()
    with st.sidebar:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        for i, (path, label, icon) in enumerate(NAV):
            with st.container(key=f"nav_{i}"):
                st.page_link(path, label=label, icon=icon, width="stretch")
        st.markdown(
            f'<div class="side-card"><div style="text-align:center">{virus_svg(96, "#d6405a")}</div>'
            f'<h4>COVID-19</h4><div class="s1">Social Media Intelligence</div>'
            f'<p>Understand public sentiment and key discussion topics during the pandemic using NLP.</p></div>',
            unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="sec-title">{html.escape(text)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_analyzer() -> PostAnalyzer:
    """One analyzer (and therefore one copy of each model) for the whole server."""
    return PostAnalyzer()


@st.cache_data(show_spinner=False)
def _results_cached(mtime: float) -> pd.DataFrame:
    return storage.load_results()


def get_results() -> pd.DataFrame:
    """Latest saved analysis results (empty DataFrame if Bulk Analysis hasn't run)."""
    return _results_cached(storage.results_mtime())


# --------------------------------------------------------------------------------------
# Result cards
# --------------------------------------------------------------------------------------
def _card(css: str, icon_html: str, label: str, value: str, sub: str) -> str:
    return (f'<div class="rcard {css}">{icon_html}<div><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val">{html.escape(value)}</div><div class="sub">{sub}</div></div></div>')


def render_result_cards(res: PostAnalysis | None) -> None:
    cols = st.columns(4)
    if res is None:
        idle = ('<div class="emoji-circle" style="background:#e2e8f0">·</div>')
        for col, lbl in zip(cols, ["Sentiment", "Detected Emotion", "Detected Topic", "Potential Misinformation"]):
            col.markdown(_card("rc-idle", idle, lbl, "—", "Click “Analyze Post” to see the result"), unsafe_allow_html=True)
        return
    # Sentiment
    if res.sentiment:
        s = res.sentiment
        css = {"negative": "rc-neg", "neutral": "rc-neu", "positive": "rc-pos"}[s.label]
        cols[0].markdown(_card(css, _face_svg(s.label), "Sentiment", s.label.title(),
                               f"Confidence: {s.confidence * 100:.0f}%"), unsafe_allow_html=True)
    else:
        cols[0].markdown(_card("rc-idle", _face_svg("none"), "Sentiment", "Unavailable", "Model could not run"), unsafe_allow_html=True)
    # Emotion
    if res.emotion:
        e = res.emotion
        cols[1].markdown(_card("rc-emo", f'<div class="emoji-circle" style="background:#e3dcff">{EMOTION_EMOJI.get(e.label, "🙂")}</div>',
                               "Detected Emotion", e.label.title(), f"Probability: {e.score * 100:.0f}%"), unsafe_allow_html=True)
    else:
        cols[1].markdown(_card("rc-idle", '<div class="emoji-circle" style="background:#e2e8f0">?</div>',
                               "Detected Emotion", "Unavailable", "Model could not run"), unsafe_allow_html=True)
    # Topic
    t = res.topic.topic
    cols[2].markdown(_card("rc-top", f'<div class="emoji-circle" style="background:#cdeee8">{TOPIC_EMOJI.get(t, "💬")}</div>',
                           "Detected Topic", t, html.escape(TOPIC_DESCRIPTIONS.get(t, ""))), unsafe_allow_html=True)
    # Misinformation (preliminary)
    m = res.misinfo
    sub = (f"Keywords found: {html.escape(', '.join(m.matched))}. " if m.flagged else "No misleading-claim keywords found. ")
    cols[3].markdown(_card("rc-mis", WARN_SVG, "Potential Misinformation", m.label,
                           sub + '<br><span class="note">Preliminary keyword screen only - not a fact-checker.</span>'),
                     unsafe_allow_html=True)


def plot(fig, width="stretch", key=None) -> None:
    """Plotly chart without the hover toolbar (cleaner, matches the reference)."""
    st.plotly_chart(fig, width=width, key=key, config={"displayModeBar": False})


def keyword_chips(words: list[str]) -> str:
    if not words:
        return '<span class="note">No keywords to show yet.</span>'
    return "".join(f'<span class="chip">{html.escape(w)}</span>' for w in words)


# --------------------------------------------------------------------------------------
# Analyzer panel (input -> extracted text -> screenshot -> results)
# --------------------------------------------------------------------------------------
MODES = ["Upload Screenshot", "Paste Text", "Twitter/X Link"]


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("_post_text", "")
    ss.setdefault("_post_image", None)
    ss.setdefault("_post_result", None)
    ss.setdefault("_edit_mode", False)
    ss.setdefault("_ocr_info", "")
    ss.setdefault("_input_msg", None)        # (level, message)
    ss.setdefault("_last_upload_id", None)


def _on_paste(key: str) -> None:
    st.session_state["_post_text"] = st.session_state[key]
    st.session_state["_force_text"] = st.session_state[key]


def _on_text_edit(key: str) -> None:
    st.session_state["_post_text"] = st.session_state[key]


def _set_text(text: str) -> None:
    st.session_state["_post_text"] = text
    st.session_state["_force_text"] = text


def _handle_upload(upload) -> None:
    ss = st.session_state
    data = upload.getvalue()
    uid = hashlib.sha1(data).hexdigest()
    if ss["_last_upload_id"] == uid:
        return
    ss["_last_upload_id"] = uid
    try:
        ocr.validate_image(data, upload.name)
    except ocr.OCRError as exc:
        ss["_post_image"] = None
        ss["_input_msg"] = ("error", str(exc))
        return
    ss["_post_image"] = data
    with st.spinner("Reading text from the screenshot (the first run downloads OCR files)..."):
        try:
            result = ocr.extract_text(data, upload.name)
        except ocr.OCRError as exc:
            ss["_input_msg"] = ("error", f"{exc}  You can still type or paste the text in the box on the right.")
            return
    if result.text:
        _set_text(result.text)
        ss["_ocr_info"] = f"OCR read {result.n_lines} line(s), average confidence {result.mean_confidence * 100:.0f}%. Please check the text."
        ss["_input_msg"] = ("success", ss["_ocr_info"])
    else:
        ss["_input_msg"] = ("warning", "No text was found in this image. Try a clearer screenshot or paste the text instead.")


def render_analyzer(page: str, detailed: bool = False) -> None:
    """The three input panels, the Analyze button and the result cards/charts."""
    _init_state()
    ss = st.session_state
    tx_key, paste_key = f"{page}_text", f"{page}_paste"
    if "_force_text" in ss:                      # a callback / OCR / URL fetch set new text
        ss[tx_key] = ss[paste_key] = ss.pop("_force_text")
    ss.setdefault(tx_key, ss["_post_text"])
    ss.setdefault(paste_key, ss["_post_text"])

    c1, c2, c3 = st.columns([1.15, 1.0, 1.0], gap="medium")

    # ---------------- 1. input ----------------
    with c1:
        section_title("1. Input a Twitter/X Post")
        mode = st.segmented_control("Input mode", MODES, default=MODES[0], key=f"{page}_mode",
                                    label_visibility="collapsed") or MODES[0]
        if mode == MODES[0]:
            up = st.file_uploader("Drag and drop a screenshot here (PNG, JPG, JPEG)", type=config.ALLOWED_IMAGE_TYPES,
                                  key=f"{page}_upload")
            if up is not None:
                _handle_upload(up)
        elif mode == MODES[1]:
            st.text_area("Paste the post text here", key=paste_key, height=170, on_change=_on_paste, args=(paste_key,),
                         placeholder="e.g. The rise in COVID-19 cases is really worrying...")
        else:
            url = st.text_input("Public Twitter/X post link", key=f"{page}_url", placeholder="https://x.com/user/status/123...")
            if st.button("Fetch post text", key=f"{page}_fetch"):
                with st.spinner("Contacting X's public oEmbed service..."):
                    fr = url_fetcher.fetch_public_post(url)
                if fr.ok:
                    _set_text(fr.text)
                    ss["_input_msg"] = ("success", fr.message)
                else:
                    ss["_input_msg"] = ("warning", fr.message)
                st.rerun()
            st.caption("Works only for public posts. If it fails, paste the text or upload a screenshot.")
        msg = ss.get("_input_msg")
        if msg:
            {"error": st.error, "warning": st.warning, "success": st.success}[msg[0]](msg[1])

    # ---------------- 2. extracted text ----------------
    with c2:
        section_title("2. Extracted Text")
        editable = mode == MODES[1] or ss["_edit_mode"]
        st.text_area("Extracted text", key=tx_key, height=170, disabled=not editable, label_visibility="collapsed",
                     on_change=_on_text_edit, args=(tx_key,), placeholder="Text from your screenshot, pasted text or link appears here.")
        b1, b2 = st.columns([1, 2])
        if b1.button("✏️ Edit Text", key=f"{page}_edit", width="stretch"):
            ss["_edit_mode"] = not ss["_edit_mode"]
            st.rerun()
        if ss["_edit_mode"]:
            b2.caption("Editing is on - change the text, then analyze.")
        analyze = st.button("🔍  Analyze Post", type="primary", width="stretch", key=f"{page}_analyze")

    # ---------------- 3. screenshot ----------------
    with c3:
        section_title("3. Original Screenshot")
        if ss["_post_image"]:
            st.image(ss["_post_image"], width="stretch")
        else:
            st.markdown('<div class="shot-ph">No screenshot uploaded<br>(your preview appears here)</div>', unsafe_allow_html=True)

    # ---------------- analyze ----------------
    if analyze:
        text = ss.get(tx_key, "")
        try:
            with st.spinner("Loading models and analysing (the first run downloads them)..."):
                ss["_post_result"] = get_analyzer().analyze(text)
        except ValueError as exc:
            ss["_post_result"] = None
            st.warning(str(exc))
        except ModelLoadError as exc:
            st.error(str(exc))
        except Exception as exc:  # never crash the page
            st.error(f"Something went wrong while analysing: {type(exc).__name__}: {exc}")

    res: PostAnalysis | None = ss["_post_result"]
    section_title("4. Analysis Results")
    if res is not None and res.errors:
        short = {n: e.split(" Details:")[0] for n, e in res.errors.items()}
        names = " and ".join(n.lower() for n in short)
        st.warning(f"The {names} could not run, so those results are shown as unavailable "
                   f"(topic, keywords and the review screen still work). {next(iter(short.values()))}")
        with st.expander("Technical details"):
            for n, e in res.errors.items():
                st.code(f"{n}: {e}")
    render_result_cards(res)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    g1, g2, g3 = st.columns([1.1, 1.1, 0.8])
    with g1:
        st.markdown('<div class="panel-title">Sentiment Probabilities</div>', unsafe_allow_html=True)
        plot(charts.sentiment_probabilities(res.sentiment.probs if res and res.sentiment else None)
             if not (res and not res.sentiment) else charts.empty_figure("Sentiment model unavailable", 250),
                        width="stretch", key=f"{page}_c_prob")
    with g2:
        st.markdown('<div class="panel-title">Emotion Analysis</div>', unsafe_allow_html=True)
        plot(charts.emotion_bars(res.emotion.scores if res and res.emotion else None)
             if not (res and not res.emotion) else charts.empty_figure("Emotion model unavailable", 250),
                        width="stretch", key=f"{page}_c_emo")
    with g3:
        st.markdown('<div class="panel-title">Keywords from Post</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel" style="min-height:230px">{keyword_chips(res.keywords if res else [])}</div>',
                    unsafe_allow_html=True)

    if detailed and res is not None:
        with st.expander("Details: how this result was produced", expanded=True):
            d1, d2 = st.columns(2)
            if res.sentiment:
                d1.markdown("**Sentiment probabilities**")
                d1.dataframe(pd.DataFrame({"Class": [k.title() for k in res.sentiment.probs],
                                           "Probability": [f"{v:.4f}" for v in res.sentiment.probs.values()]}),
                             hide_index=True, width="stretch")
            if res.emotion:
                d2.markdown("**All emotion scores**")
                d2.dataframe(pd.DataFrame({"Emotion": [k.title() for k in res.emotion.scores],
                                           "Score": [f"{v:.4f}" for v in res.emotion.scores.values()]})
                             .sort_values("Score", ascending=False), hide_index=True, width="stretch")
            matched = ", ".join(res.topic.matched) or "none"
            st.markdown(f"**Topic (rule-based):** {res.topic.topic} - matched keywords: {matched}. "
                        f"Topic scores: {res.topic.scores or 'no matches -> General'}.")
            st.markdown(f"**Misinformation screen:** {res.misinfo.label}"
                        + (f" - matched: {', '.join(res.misinfo.matched)}" if res.misinfo.flagged else "") + f"  \n*{DISCLAIMER}*")
            an = get_analyzer()
            st.caption(f"Models: {an.sentiment_model.name} (sentiment) · {an.emotion_model.name} (emotion). "
                       f"Analysed text: “{res.text}”")


# --------------------------------------------------------------------------------------
# Small helpers used by several pages
# --------------------------------------------------------------------------------------
def no_results_notice() -> None:
    st.info("No analysed dataset yet. Open **Bulk Analysis**, choose how many posts to process "
            "(a few hundred is a good first run) and click **Start analysis**. Results are saved and "
            "appear on every page.")
