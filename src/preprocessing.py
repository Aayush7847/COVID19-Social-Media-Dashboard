"""Text cleaning helpers shared by single-post and bulk analysis."""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata

import pandas as pd

import config

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\ufeff]")


def normalize_for_model(text: object) -> str:
    """Prepare text for the transformer models.

    Follows the convention of the Cardiff NLP Twitter models: user mentions become
    '@user' and links become 'http'. Everything else (case, emoji, hashtags) is kept
    because the models were trained on raw tweets.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = html.unescape(str(text))
    s = unicodedata.normalize("NFKC", s)
    s = _CTRL_RE.sub(" ", s)
    s = _URL_RE.sub("http", s)
    s = _MENTION_RE.sub("@user", s)
    s = _WS_RE.sub(" ", s).strip()
    return s[: config.MAX_TEXT_CHARS]


def normalize_key(text: object) -> str:
    """Aggressive normalisation used only for de-duplication (never shown)."""
    s = normalize_for_model(text).lower()
    s = s.replace("@user", " ").replace("http", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    return _WS_RE.sub(" ", s).strip()


def is_valid_text(text: object, min_chars: int = config.MIN_TEXT_CHARS) -> bool:
    """A post is usable if it has at least `min_chars` characters incl. a letter."""
    s = normalize_for_model(text)
    return len(s) >= min_chars and any(ch.isalpha() for ch in s)


def text_hash(text: object) -> str:
    """Stable id for caching predictions of identical text."""
    return hashlib.sha1(normalize_for_model(text).encode("utf-8")).hexdigest()


def clean_dataframe(df: pd.DataFrame, text_col: str = "text") -> tuple[pd.DataFrame, dict]:
    """Drop empty/invalid and duplicate posts. Returns (clean_df, stats).

    Stats keys: input_rows, empty_or_invalid, duplicates, valid.
    """
    n_in = len(df)
    if n_in == 0:
        return df.copy(), {"input_rows": 0, "empty_or_invalid": 0, "duplicates": 0, "valid": 0}
    out = df.copy()
    out[text_col] = out[text_col].map(normalize_for_model)
    ok = out[text_col].map(is_valid_text)
    n_invalid = int((~ok).sum())
    out = out[ok]
    key = out[text_col].map(normalize_key)
    dup = key.duplicated(keep="first")
    n_dup = int(dup.sum())
    out = out[~dup].reset_index(drop=True)
    return out, {
        "input_rows": n_in,
        "empty_or_invalid": n_invalid,
        "duplicates": n_dup,
        "valid": len(out),
    }
