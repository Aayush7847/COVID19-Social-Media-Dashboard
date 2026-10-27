"""Simple, transparent keyword extraction (frequency based, stop-words removed)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_EXTRA_STOP = {
    "rt", "amp", "http", "https", "www", "com", "via", "user", "just", "like", "get", "got",
    "will", "would", "can", "could", "dont", "doesnt", "didnt", "im", "ive", "youre",
    "thats", "lol", "pls", "please", "also", "still", "even", "really", "much", "one",
    "ha", "wa", "ur", "nt", "gon", "na", "wan", "isnt", "cant", "wont", "going", "know",
    "make", "need", "say", "says", "said", "new", "today", "people", "getting", "follow",
    "back", "time", "day", "days", "want", "see", "think", "look", "way", "come", "take",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | _EXTRA_STOP
# Words that appear in almost every post of a COVID dataset and say nothing about topic.
DOMAIN_TERMS = {"covid", "covid19", "covid-19", "coronavirus", "corona", "virus", "pandemic",
                "coronaviruspandemic", "covid_19", "sarscov2", "ncov"}

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_MENTION = re.compile(r"@\w+")
_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")


def tokenize(text: object, drop_domain_terms: bool = False) -> list[str]:
    if not isinstance(text, str):
        return []
    s = _MENTION.sub(" ", _URL.sub(" ", text.lower())).replace("#", " ")
    out = []
    for tok in _TOKEN.findall(s):
        tok = tok.strip("'-")
        if len(tok) < 3 or tok in STOP_WORDS:
            continue
        if drop_domain_terms and tok in DOMAIN_TERMS:
            continue
        out.append(tok)
    return out


def extract_keywords(text: object, top_n: int = 8) -> list[str]:
    """Top keywords of ONE post: unique, ranked by frequency then first appearance."""
    toks = tokenize(text)
    if not toks:
        return []
    freq = Counter(toks)
    first = {}
    for i, t in enumerate(toks):
        first.setdefault(t, i)
    return sorted(freq, key=lambda t: (-freq[t], first[t]))[:top_n]


def top_keywords(texts: Iterable[object], top_n: int = 15) -> list[tuple[str, int]]:
    """Most frequent keywords over many posts (each post counts a word once)."""
    c: Counter = Counter()
    for t in texts:
        c.update(set(tokenize(t, drop_domain_terms=True)))
    return c.most_common(top_n)
