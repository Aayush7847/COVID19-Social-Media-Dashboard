"""Batch processing of a whole dataset with persistent caching and progress."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import config
from src.analyzer import PostAnalyzer
from src.preprocessing import text_hash
from src.storage import PredictionCache, model_signature
from src.topic_classifier import classify_topic
from src.misinformation import screen_text

log = logging.getLogger(__name__)

ProgressCB = Callable[[int, int, float, float, str], None]   # done, total, posts/s, eta_s, message


@dataclass
class BulkRun:
    results: pd.DataFrame        # one row per post that was attempted (ok or failed)
    requested: int
    from_cache: int
    newly_analyzed: int
    failed: int
    not_processed: int           # requested but never reached (run was stopped)
    seconds: float
    stopped: bool


def _ml_columns(sents, emos, emotion_labels: list[str]) -> pd.DataFrame:
    rows = []
    for s, e in zip(sents, emos):
        if s is None or e is None:
            rows.append(None)
            continue
        row = {"sentiment": s.label, "sentiment_confidence": s.confidence,
               "p_negative": s.probs["negative"], "p_neutral": s.probs["neutral"],
               "p_positive": s.probs["positive"], "emotion": e.label, "emotion_score": e.score}
        for lab in emotion_labels:
            row[f"emo_{lab}"] = e.scores.get(lab, 0.0)
        rows.append(row)
    return rows


def add_rule_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Topic + preliminary misinformation flag (cheap, rule based, always recomputed)."""
    out = df.copy()
    topics = [classify_topic(t) for t in out["text"]]
    out["topic"] = [t.topic for t in topics]
    out["topic_keywords"] = [", ".join(t.matched) for t in topics]
    mis = [screen_text(t) for t in out["text"]]
    out["misinfo_flag"] = [m.flagged for m in mis]
    out["misinfo_terms"] = [", ".join(m.matched) for m in mis]
    return out


def run_bulk(posts: pd.DataFrame, analyzer: PostAnalyzer, cache: PredictionCache,
             batch_size: int = config.BATCH_SIZE, progress: ProgressCB | None = None,
             should_stop: Callable[[], bool] | None = None) -> BulkRun:
    """Analyse `posts` (needs a 'text' column; other columns are carried through).

    * Posts already in the cache (same text AND same models) are not re-analysed.
    * Work is done in batches; the cache is written to disk every few batches and
      again in a `finally` block, so pressing Stop never loses finished work.
    """
    t0 = time.time()
    posts = posts.reset_index(drop=True).copy()
    sig = model_signature(*analyzer.signature_parts)
    posts["_key"] = [f"{sig}:{text_hash(t)}" for t in posts["text"]]
    requested = len(posts)

    cached_mask = np.array(cache.has(posts["_key"].tolist()), dtype=bool)
    todo_idx = list(np.flatnonzero(~cached_mask))
    # same text may appear twice with different keys? No - keys are text hashes, so
    # identical texts share one key; analyse each unique key once.
    seen, unique_todo = set(), []
    for i in todo_idx:
        k = posts.at[i, "_key"]
        if k not in seen:
            seen.add(k)
            unique_todo.append(i)

    new_rows: dict[str, dict] = {}
    failures: dict[str, str] = {}
    done, stopped, n_new_ok = 0, False, 0
    n_todo = len(unique_todo)
    if progress:
        progress(0, n_todo, 0.0, 0.0, f"{int(cached_mask.sum()):,} posts found in cache; {n_todo:,} to analyse")

    try:
        for b, start in enumerate(range(0, n_todo, batch_size)):
            if should_stop and should_stop():
                stopped = True
                break
            idx = unique_todo[start:start + batch_size]
            texts = [posts.at[i, "text"] for i in idx]
            sents, emos, errs = analyzer.predict_batch(texts, batch_size)
            labels = analyzer.emotion_model.labels
            for i, row, err in zip(idx, _ml_columns(sents, emos, labels), errs):
                k = posts.at[i, "_key"]
                if row is None:
                    failures[k] = err or "Model returned no result"
                else:
                    new_rows[k] = row
                    n_new_ok += 1
            done += len(idx)
            elapsed = max(time.time() - t0, 1e-6)
            rate = done / elapsed
            if progress:
                progress(done, n_todo, rate, (n_todo - done) / rate if rate else 0.0,
                         f"Analysed {done:,} of {n_todo:,} new posts")
            if (b + 1) % config.SAVE_EVERY_N_BATCHES == 0 and new_rows:
                cache.add(pd.DataFrame.from_dict(new_rows, orient="index").rename_axis("key"))
                cache.save()
                new_rows = {}
    finally:   # also runs when Streamlit's Stop button interrupts the script
        if new_rows:
            cache.add(pd.DataFrame.from_dict(new_rows, orient="index").rename_axis("key"))
            new_rows = {}
        cache.save()

    # --- assemble results for every post that has predictions or a recorded failure ----
    ml = cache.df
    has_pred = posts["_key"].isin(ml.index)
    attempted = has_pred | posts["_key"].isin(failures)
    base = posts[attempted].reset_index(drop=True)
    ok = base["_key"].isin(ml.index).to_numpy()
    pred = ml.reindex(base["_key"]).reset_index(drop=True)      # NaN rows for failures
    out = pd.concat([base.drop(columns=["_key"]), pred], axis=1)
    out["status"] = np.where(ok, "ok", "failed")
    out["error"] = [("" if o else failures.get(k, "")) for o, k in zip(ok, base["_key"])]
    out = add_rule_columns(out) if len(out) else out

    n_failed = int((~ok).sum())
    return BulkRun(results=out, requested=requested, from_cache=int(cached_mask.sum()),
                   newly_analyzed=n_new_ok, failed=n_failed,
                   not_processed=int(requested - len(out)),
                   seconds=time.time() - t0, stopped=stopped)
