"""Screenshot validation and text extraction with EasyOCR.

EasyOCR downloads its (small) detection/recognition weights from GitHub the first
time it runs, so the first screenshot needs an internet connection.
"""
from __future__ import annotations

import io
import logging
import re
import statistics
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

import config

log = logging.getLogger(__name__)
_reader = None


class OCRError(RuntimeError):
    """Message is safe to show to the user."""


class InvalidImageError(OCRError):
    pass


@dataclass
class OCRResult:
    text: str
    n_lines: int
    mean_confidence: float


def validate_image(data: bytes, filename: str = "") -> Image.Image:
    """Validate an uploaded file really is a PNG/JPEG within the size limit."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidImageError(f"The image is larger than {config.MAX_UPLOAD_MB} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").upper()
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("This file is not a readable image. Please upload a PNG or JPG screenshot.") from exc
    if fmt not in ("PNG", "JPEG"):
        raise InvalidImageError(f"Unsupported image type '{fmt or 'unknown'}'. Please upload PNG, JPG or JPEG.")
    if img.width < 20 or img.height < 20:
        raise InvalidImageError("The image is too small to contain readable text.")
    img = img.convert("RGB")
    longest = max(img.size)
    if longest > config.MAX_IMAGE_SIDE:
        scale = config.MAX_IMAGE_SIDE / longest
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


def _get_reader():
    global _reader
    if _reader is not None:
        return _reader
    try:
        import easyocr
        import torch
    except ImportError as exc:
        raise OCRError("EasyOCR is not installed. Start the app with run.bat (it installs requirements) "
                       "or run: pip install -r requirements.txt") from exc
    try:
        use_gpu = torch.cuda.is_available() and not config.FORCE_CPU
        _reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=use_gpu, verbose=False)
    except Exception as exc:
        raise OCRError("OCR could not start. The first run downloads small OCR model files from the "
                       f"internet - check your connection and try again. Details: {type(exc).__name__}: {exc}") from exc
    return _reader


# --- layout helpers (pure functions, unit-tested) -------------------------------------
def group_into_lines(boxes: list[tuple[list, str, float]]) -> list[str]:
    """Group EasyOCR word boxes into reading-order text lines."""
    items = []
    for bbox, text, conf in boxes:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({"t": text.strip(), "x": min(xs), "yc": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys)})
    items = [i for i in items if i["t"]]
    if not items:
        return []
    med_h = statistics.median(i["h"] for i in items) or 10
    items.sort(key=lambda i: (i["yc"], i["x"]))
    lines, current = [], [items[0]]
    for it in items[1:]:
        if abs(it["yc"] - statistics.mean(c["yc"] for c in current)) <= 0.6 * med_h:
            current.append(it)
        else:
            lines.append(current)
            current = [it]
    lines.append(current)
    return [" ".join(i["t"] for i in sorted(ln, key=lambda i: i["x"])) for ln in lines]


_COUNT_ONLY = re.compile(r"^[\d.,\sKkMm]+$")
_HANDLE_LINE = re.compile(r"^[^@]{0,40}@\w+\s*(?:[·.\-•]\s*)?(?:\w{3,9}\.?\s+\d{1,2}(?:,\s*\d{4})?|\d+[smhd])?$")
_ELLIPSIS = re.compile(r"^[.·•\s]{1,5}$")


def strip_tweet_chrome(lines: list[str]) -> list[str]:
    """Heuristically drop screenshot UI text (like/retweet counts, '@handle · Apr 12').

    This is only a convenience: the user can always edit the extracted text.
    """
    keep = []
    for ln in lines:
        s = ln.strip()
        if not s or _COUNT_ONLY.match(s) or _ELLIPSIS.match(s) or _HANDLE_LINE.match(s):
            continue
        keep.append(s)
    return keep


def extract_text(data: bytes, filename: str = "", strip_ui: bool = True) -> OCRResult:
    """Validate the upload, run OCR and return the recognised text."""
    img = validate_image(data, filename)
    reader = _get_reader()
    try:
        raw = reader.readtext(np.array(img), detail=1, paragraph=False)
    except Exception as exc:
        raise OCRError(f"OCR failed on this image: {type(exc).__name__}: {exc}") from exc
    good = [r for r in raw if r[2] >= config.OCR_MIN_CONFIDENCE]
    lines = group_into_lines(good)
    if strip_ui:
        cleaned = strip_tweet_chrome(lines)
        lines = cleaned or lines
    text = " ".join(lines).strip()
    conf = float(np.mean([r[2] for r in good])) if good else 0.0
    return OCRResult(text=text, n_lines=len(lines), mean_confidence=conf)
