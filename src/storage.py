"""Local persistence: prediction cache + latest analysed results + CSV export.

Plain files in outputs/ - no database. Writes are atomic (temp file + rename) so a
crash or a Stop click cannot corrupt them.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd

import config

log = logging.getLogger(__name__)


def _atomic_write(df: pd.DataFrame, path: Path, **kwargs) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def model_signature(sentiment_model: str, emotion_model: str) -> str:
    """Short id so cached predictions are never mixed between different models."""
    return hashlib.sha1(f"{sentiment_model}|{emotion_model}".encode()).hexdigest()[:10]


class PredictionCache:
    """Persistent cache of EXPENSIVE model outputs only (topic rules are cheap and
    recomputed every time, so editing the keyword rules takes effect immediately)."""

    def __init__(self, path: Path = config.PREDICTION_CACHE_FILE):
        self.path = Path(path)
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        if self.path.exists():
            try:
                df = pd.read_csv(self.path, compression="gzip" if str(self.path).endswith(".gz") else None)
                return df.drop_duplicates("key").set_index("key")
            except Exception as exc:
                log.warning("Prediction cache unreadable (%s); starting fresh.", exc)
        return pd.DataFrame().rename_axis("key")

    def __len__(self) -> int:
        return len(self.df)

    def has(self, keys: list[str]) -> list[bool]:
        if self.df.empty:
            return [False] * len(keys)
        idx = self.df.index
        return [k in idx for k in keys]

    def get(self, keys: list[str]) -> pd.DataFrame:
        return self.df.loc[keys] if keys else self.df.iloc[0:0]

    def add(self, new_rows: pd.DataFrame) -> None:
        """new_rows is indexed by key and holds only successful predictions."""
        if new_rows.empty:
            return
        merged = pd.concat([self.df, new_rows[~new_rows.index.isin(self.df.index)]]) if not self.df.empty else new_rows
        self.df = merged

    def save(self) -> None:
        if self.df.empty:
            return
        _atomic_write(self.df.reset_index(), self.path, compression="gzip")

    def clear(self) -> None:
        self.df = pd.DataFrame().rename_axis("key")
        if self.path.exists():
            self.path.unlink()


# --- latest analysed results -----------------------------------------------------------
def save_results(df: pd.DataFrame, path: Path = config.LATEST_RESULTS_FILE) -> None:
    _atomic_write(df, path)


def load_results(path: Path = config.LATEST_RESULTS_FILE) -> pd.DataFrame:
    """Load the last saved results (empty DataFrame if none)."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def results_mtime(path: Path = config.LATEST_RESULTS_FILE) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV for download. UTF-8 with BOM so Excel on Windows shows emoji/accents correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")
