"""Optional: fetch the text of a PUBLIC Twitter/X post through X's official oEmbed
endpoint (no API key, no scraping, no login bypass).

Limits: works only for public posts that X still serves through oEmbed. It can fail
because of deleted/private posts, network restrictions or X changing the service.
Every failure returns a friendly message and the rest of the app keeps working.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

_URL_RE = re.compile(r"^https?://(?:www\.|mobile\.)?(?:twitter|x)\.com/[A-Za-z0-9_]{1,15}/status/\d+", re.I)
OEMBED = "https://publish.twitter.com/oembed"

FALLBACK_MESSAGE = ("Please paste the post text or upload a screenshot instead - those modes always work.")


@dataclass
class FetchResult:
    ok: bool
    text: str = ""
    message: str = ""


def is_post_url(url: str) -> bool:
    return bool(_URL_RE.match((url or "").strip()))


def parse_oembed_html(markup: str) -> str:
    """Extract the tweet text from the oEmbed <blockquote> HTML."""
    m = re.search(r"<p[^>]*>(.*?)</p>", markup or "", re.S | re.I)
    if not m:
        return ""
    inner = re.sub(r"<br\s*/?>", "\n", m.group(1), flags=re.I)
    inner = re.sub(r"<[^>]+>", "", inner)
    inner = html.unescape(inner)
    inner = re.sub(r"\s*pic\.(twitter|x)\.com/\S+", "", inner)
    return re.sub(r"[ \t]+", " ", inner).strip()


def fetch_public_post(url: str, timeout: float = 8.0) -> FetchResult:
    url = (url or "").strip()
    if not url:
        return FetchResult(False, message="Enter a post link first. " + FALLBACK_MESSAGE)
    if not is_post_url(url):
        return FetchResult(False, message="That does not look like a Twitter/X post link "
                           "(expected https://x.com/<user>/status/<id>). " + FALLBACK_MESSAGE)
    api = f"{OEMBED}?{urllib.parse.urlencode({'url': url.replace('://x.com', '://twitter.com'), 'omit_script': '1'})}"
    req = urllib.request.Request(api, headers={"User-Agent": "COVID19-Dashboard-College-Project/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        why = "The post is private, deleted or not available publicly." if exc.code in (403, 404) else f"The service answered with HTTP {exc.code}."
        return FetchResult(False, message=f"{why} {FALLBACK_MESSAGE}")
    except (urllib.error.URLError, TimeoutError, OSError):
        return FetchResult(False, message="Could not reach X (no internet, a firewall, or the service is blocked). " + FALLBACK_MESSAGE)
    except (ValueError, json.JSONDecodeError):
        return FetchResult(False, message="X returned an unreadable answer. " + FALLBACK_MESSAGE)
    text = parse_oembed_html(payload.get("html", ""))
    if not text:
        return FetchResult(False, message="No post text could be found at that link. " + FALLBACK_MESSAGE)
    return FetchResult(True, text=text, message="Post text retrieved from X's public oEmbed service.")
