"""Plotly figures. Every function accepts empty / partial data and returns a figure
with a friendly message instead of raising. All numbers come from the data passed in."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import config

C = config.COLORS
SENT_COLORS = config.SENTIMENT_COLORS
EMOTION_COLORS = {"fear": "#f87171", "sadness": "#a78bfa", "anger": "#fb923c", "joy": "#4ade80",
                  "disgust": "#a3a312", "surprise": "#38bdf8", "neutral": "#94a3b8",
                  "optimism": "#22c55e", "love": "#f472b6"}
FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#1e293b")


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    margin = kw.pop("margin", dict(l=10, r=10, t=10, b=10))
    fig.update_layout(height=height, margin=margin, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=FONT, **kw)
    return fig


def empty_figure(message: str = "No data yet", height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=C["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, height)


def ok_rows(df: pd.DataFrame | None, required: tuple[str, ...] = ()) -> pd.DataFrame:
    """Only successfully analysed rows that contain the columns we need."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "status" in df:
        df = df[df["status"] == "ok"]
    if any(c not in df for c in required):
        return pd.DataFrame()
    return df.dropna(subset=list(required)) if required else df


def pretty(label: str) -> str:
    return str(label).replace("_", " ").title()


# --- single post -------------------------------------------------------------------------
def sentiment_probabilities(probs: dict[str, float] | None) -> go.Figure:
    if not probs:
        return empty_figure("Analyse a post to see probabilities")
    order = ["negative", "neutral", "positive"]
    vals = [probs.get(k, 0.0) for k in order]
    fig = go.Figure(go.Bar(x=[k.title() for k in order], y=vals, marker_color=[SENT_COLORS[k] for k in order],
                           text=[f"{v:.2f}" for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.12], gridcolor="#e2e8f0", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return _layout(fig, 250)


def emotion_bars(scores: dict[str, float] | None, top_n: int = 5) -> go.Figure:
    if not scores:
        return empty_figure("Analyse a post to see emotions")
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n][::-1]
    fig = go.Figure(go.Bar(y=[pretty(k) for k, _ in items], x=[v * 100 for _, v in items], orientation="h",
                           marker_color=[EMOTION_COLORS.get(k, C["emotion"]) for k, _ in items],
                           text=[f"{v * 100:.0f}%" for _, v in items], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(range=[0, 115], visible=False)
    fig.update_yaxes(showgrid=False)
    return _layout(fig, 250)


# --- dataset views -----------------------------------------------------------------------
def sentiment_donut(df: pd.DataFrame | None) -> go.Figure:
    d = ok_rows(df, ("sentiment",))
    if d.empty:
        return empty_figure("Run Bulk Analysis to see the dataset view", 270)
    counts = d["sentiment"].value_counts().reindex(config.SENTIMENT_ORDER, fill_value=0)
    fig = go.Figure(go.Pie(labels=[k.title() for k in counts.index], values=counts.values, hole=0.62,
                           marker=dict(colors=[SENT_COLORS[k] for k in counts.index], line=dict(color="white", width=2)),
                           sort=False, textinfo="percent", hovertemplate="%{label}: %{value:,} posts (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(d):,}</b><br>Posts", x=0.5, y=0.5, showarrow=False, font=dict(size=18))
    return _layout(fig, 270, legend=dict(orientation="v", x=1.02, y=0.5))


def topic_sentiment_stacked(df: pd.DataFrame | None) -> go.Figure:
    d = ok_rows(df, ("sentiment", "topic"))
    if d.empty:
        return empty_figure("Run Bulk Analysis to see topic-wise sentiment", 270)
    ct = pd.crosstab(d["topic"], d["sentiment"]).reindex(columns=config.SENTIMENT_ORDER, fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]            # most discussed at the top
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for s in config.SENTIMENT_ORDER:
        fig.add_bar(y=ct.index, x=pct[s], name=s.title(), orientation="h", marker_color=SENT_COLORS[s],
                    text=[f"{v:.0f}%" if v >= 6 else "" for v in pct[s]], textposition="inside",
                    textfont=dict(color="white"), customdata=ct[s],
                    hovertemplate="%{y} - " + s.title() + ": %{x:.1f}% (%{customdata:,} posts)<extra></extra>")
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
    fig.update_xaxes(range=[0, 100], visible=False)
    return _layout(fig, 290, margin=dict(l=10, r=28, t=10, b=10))


def topic_counts(df: pd.DataFrame | None) -> go.Figure:
    d = ok_rows(df, ("topic",))
    if d.empty:
        return empty_figure("Run Bulk Analysis to see topic counts", 270)
    vc = d["topic"].value_counts().sort_values()
    fig = go.Figure(go.Bar(y=vc.index, x=vc.values, orientation="h", marker_color=C["blue"],
                           text=[f"{v:,}" for v in vc.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,} posts<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, vc.max() * 1.18])
    return _layout(fig, 290)


def emotion_distribution(df: pd.DataFrame | None) -> go.Figure:
    d = ok_rows(df, ("emotion",))
    if d.empty:
        return empty_figure("Run Bulk Analysis to see emotions", 270)
    vc = d["emotion"].value_counts().sort_values()
    fig = go.Figure(go.Bar(y=[pretty(k) for k in vc.index], x=vc.values, orientation="h",
                           marker_color=[EMOTION_COLORS.get(k, C["emotion"]) for k in vc.index],
                           text=[f"{v:,}" for v in vc.values], textposition="outside", cliponaxis=False))
    fig.update_xaxes(visible=False, range=[0, vc.max() * 1.18])
    return _layout(fig, 290)


def emotions_by_topic(df: pd.DataFrame | None) -> go.Figure:
    d = ok_rows(df, ("emotion", "topic"))
    if d.empty:
        return empty_figure("No emotion data for this selection", 300)
    ct = pd.crosstab(d["topic"], d["emotion"])
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure(go.Heatmap(z=pct.values, x=[pretty(c) for c in pct.columns], y=list(pct.index),
                               colorscale=[[0, "#f5f3ff"], [1, C["emotion"]]], colorbar=dict(title="% of topic"),
                               text=[[f"{v:.0f}%" for v in row] for row in pct.values], texttemplate="%{text}",
                               hovertemplate="%{y} / %{x}: %{z:.1f}%<extra></extra>"))
    return _layout(fig, 320)


def keywords_bar(freq: list[tuple[str, int]], color: str = C["topic"]) -> go.Figure:
    if not freq:
        return empty_figure("No keywords for this selection", 320)
    words, counts = zip(*freq[::-1])
    fig = go.Figure(go.Bar(y=list(words), x=list(counts), orientation="h", marker_color=color,
                           text=list(counts), textposition="outside", cliponaxis=False))
    fig.update_xaxes(visible=False, range=[0, max(counts) * 1.15])
    return _layout(fig, max(260, 24 * len(words) + 40))


def sentiment_trend(df: pd.DataFrame | None) -> go.Figure | None:
    """Daily (or weekly for long ranges) sentiment counts. Returns None if no usable dates."""
    d = ok_rows(df, ("sentiment", "date"))
    if d.empty:
        return None
    d = d.assign(date=pd.to_datetime(d["date"], errors="coerce")).dropna(subset=["date"])
    if d.empty:
        return None
    span = (d["date"].max() - d["date"].min()).days
    freq, label = ("W", "week") if span > 120 else ("D", "day")
    g = d.groupby([pd.Grouper(key="date", freq=freq), "sentiment"]).size().unstack(fill_value=0)
    g = g.reindex(columns=config.SENTIMENT_ORDER, fill_value=0)
    fig = go.Figure()
    for s in config.SENTIMENT_ORDER:
        fig.add_scatter(x=g.index, y=g[s], mode="lines+markers", name=s.title(), line=dict(color=SENT_COLORS[s]))
    fig.update_yaxes(title=f"Posts per {label}", gridcolor="#e2e8f0")
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return _layout(fig, 320)
