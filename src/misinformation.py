"""PRELIMINARY keyword screen for commonly circulated COVID-19 claims.

This is NOT fact-checking. It only notices words and phrases that frequently
appear in debunked claims. A flag means "a human may want to look at this";
no flag does NOT mean the post is accurate, and a flag does NOT mean it is false.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REVIEW_PATTERNS = [
    r"5g", r"hoax", r"plandemic", r"scamdemic", r"bioweapon", r"bio-weapon",
    r"microchip\w*", r"depopulation", r"miracle cure", r"drink(ing)? bleach",
    r"inject(ing)? bleach", r"cures? covid", r"cure for (covid|corona\w*)",
    r"just (a|the) flu", r"no worse than the flu", r"vaccines? (cause|causes|caused) (autism|infertility|death)",
    r"fake pandemic", r"covid is (fake|a lie|not real)", r"made in a lab", r"lab[- ]?leak",
    r"hydroxychloroquine (cures|prevents)", r"ivermectin (cures|prevents)",
    r"bill gates", r"masks? (don'?t|do not|doesn'?t) work", r"crisis actors?",
]
_RX = [re.compile(rf"(?<![a-z0-9]){p}(?![a-z0-9])", re.IGNORECASE) for p in REVIEW_PATTERNS]

DISCLAIMER = ("Preliminary keyword screening only - this is not a fact-checking system. "
              "It cannot tell whether a claim is true or false.")


@dataclass
class MisinfoResult:
    flagged: bool
    matched: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "Review suggested" if self.flagged else "Not Detected"


def screen_text(text: object) -> MisinfoResult:
    if not isinstance(text, str) or not text.strip():
        return MisinfoResult(False)
    found = []
    for rx in _RX:
        m = rx.search(text)
        if m:
            found.append(m.group(0).lower())
    found = list(dict.fromkeys(found))
    return MisinfoResult(bool(found), found)
