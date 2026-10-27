"""Explainable, keyword-based topic classifier.

HOW IT WORKS (also described on the About page)
  1. Each topic below has a list of keyword patterns (regular expressions).
  2. A pattern ending in `*` matches any word that starts that way, so
     'vaccin*' matches vaccine, vaccines, vaccinated, vaccination - and the
     stemmed form 'vaccin' found in the global dataset file.
  3. The topic score = number of DISTINCT matched words in the post. Overlapping
     matches (e.g. 'vaccin*' and 'vaccine*' both hitting 'vaccine') count once.
  4. The highest score wins. TIES are broken by TOPIC_PRIORITY (specific topics
     before broad ones, e.g. Vaccination beats Healthcare).
  5. If nothing matches the topic is 'General'.

These are rule-based labels, NOT ground truth. To tune the classifier just edit
TOPIC_KEYWORDS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Healthcare": [
        "hospital*", "icu", "ventilator*", "doctor*", "nurse*", "patient*", "healthcare",
        "health care", "clinic*", "medical", "medic*", "ppe", "symptom*", "treatment*",
        "cases", "deaths", "infected", "infection*", "test*", "testing", "frontline",
        "oxygen", "outbreak", "recover*", "mask*", "health",
    ],
    "Vaccination": [
        "vaccin*", "vaccine*", "vax*", "jab", "jabs", "booster*", "dose", "doses",
        "pfizer", "moderna", "astrazeneca", "covaxin", "sputnik", "immuniz*", "immunis*",
        "antivax", "anti-vax", "inoculat*", "herd immunity",
    ],
    "Lockdown": [
        "lockdown*", "lock down", "quarantin*", "stay at home", "stay home", "stayathome",
        "stayhome", "social distanc*", "curfew", "shutdown*", "shut down", "shelter in place",
        "self-isolat*", "self isolat*", "isolation", "restrictions", "work from home", "wfh",
    ],
    "Mental Health": [
        "anxiety", "anxious", "depress*", "stress*", "mental", "lonel*", "panic*",
        "therapy", "therapist*", "burnout", "mental health", "trauma*", "overwhelm*",
        "fear of", "afraid", "scared",
    ],
    "Economy": [
        "econom*", "unemploy*", "layoff*", "laid off", "job", "jobs", "stock", "stocks",
        "market*", "recession", "business*", "stimulus", "inflation", "gdp", "bailout",
        "oil price*", "salary", "salaries", "paycheck*", "financial", "finance", "rent",
        "bankrupt*", "supply chain", "trade",
    ],
}

# Tie-break order: more specific topics first.
TOPIC_PRIORITY = ["Vaccination", "Lockdown", "Mental Health", "Economy", "Healthcare"]
GENERAL = "General"


def _compile(pattern: str) -> re.Pattern:
    star = pattern.endswith("*")
    core = re.escape(pattern.rstrip("*")).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){core}{r'[a-z]*' if star else ''}(?![a-z0-9])", re.IGNORECASE)


_COMPILED = {t: [(p, _compile(p)) for p in pats] for t, pats in TOPIC_KEYWORDS.items()}


@dataclass
class TopicResult:
    topic: str
    matched: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)


def _distinct_matches(text: str, rules) -> list[str]:
    """All keyword hits for one topic, overlapping hits collapsed to the longest one."""
    spans = []
    for _, rx in rules:
        spans.extend((m.start(), m.end()) for m in rx.finditer(text))
    spans.sort(key=lambda se: (se[0] - se[1], se[0]))        # longest first
    taken: list[tuple[int, int]] = []
    for st, en in spans:
        if all(en <= t0 or st >= t1 for t0, t1 in taken):
            taken.append((st, en))
    words = [re.sub(r"\s+", " ", text[st:en].lower()) for st, en in sorted(taken)]
    return list(dict.fromkeys(words))


def classify_topic(text: object) -> TopicResult:
    """Return the topic for one post (never raises; empty text -> General)."""
    if not isinstance(text, str) or not text.strip():
        return TopicResult(GENERAL)
    # hashtags: '#vaccine' -> ' vaccine' so the word-boundary rules still fire
    s = text.replace("#", " ")
    hits: dict[str, list[str]] = {}
    for topic, rules in _COMPILED.items():
        found = _distinct_matches(s, rules)
        if found:
            hits[topic] = found
    if not hits:
        return TopicResult(GENERAL)
    scores = {t: len(v) for t, v in hits.items()}
    best = max(scores.values())
    tied = [t for t, sc in scores.items() if sc == best]
    winner = min(tied, key=TOPIC_PRIORITY.index)
    return TopicResult(winner, hits[winner], scores)


TOPIC_DESCRIPTIONS = {
    "Healthcare": "Related to hospitals, cases, medical facilities.",
    "Vaccination": "Related to vaccines, doses and immunisation.",
    "Lockdown": "Related to quarantine, curfews and distancing rules.",
    "Mental Health": "Related to anxiety, stress and emotional wellbeing.",
    "Economy": "Related to jobs, markets and business impact.",
    "General": "No specific topic keywords matched.",
}
