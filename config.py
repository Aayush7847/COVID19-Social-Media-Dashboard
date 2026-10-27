"""Central configuration. Everything a beginner might want to change lives here.

All paths are relative to this file, so the project folder can be moved anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR.mkdir(exist_ok=True)

# Put your own dataset (.csv, .tsv, .json, .jsonl, or a .zip containing them)
# in the data/ folder and it will show up on the Bulk Analysis page.
SUPPORTED_DATA_SUFFIXES = (".csv", ".tsv", ".txt", ".json", ".jsonl")

# Optional: force a specific text column for every dataset (leave None = auto-detect).
TEXT_COLUMN_OVERRIDE: str | None = None

# ----------------------------------------------------------------------------
# Processing limits
# ----------------------------------------------------------------------------
# Number of posts pre-selected on the Bulk Analysis page. Raise it once you have
# seen how fast your computer is. Set to 0 to pre-select "all valid posts".
DEFAULT_PROCESS_LIMIT = int(os.environ.get("COVID_DASH_LIMIT", 300))
# Posts per model call. Lower this if you run out of memory.
BATCH_SIZE = int(os.environ.get("COVID_DASH_BATCH", 32))
# Results are written to disk after this many batches (so Stop never loses work).
SAVE_EVERY_N_BATCHES = 5
MAX_MODEL_TOKENS = 128          # tweets are short; keeps CPU inference fast
MAX_TEXT_CHARS = 1000           # longer posts are truncated before analysis
MIN_TEXT_CHARS = 3

# ----------------------------------------------------------------------------
# Pretrained models (downloaded from Hugging Face on first use, then cached)
# ----------------------------------------------------------------------------
SENTIMENT_MODEL = os.environ.get(
    "COVID_DASH_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
EMOTION_MODEL = os.environ.get(
    "COVID_DASH_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
)
FORCE_CPU = os.environ.get("COVID_DASH_FORCE_CPU", "0") == "1"

# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------
OCR_LANGUAGES = ["en"]
OCR_MIN_CONFIDENCE = 0.30
MAX_UPLOAD_MB = 10
MAX_IMAGE_SIDE = 2200           # larger screenshots are down-scaled for speed
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]

# ----------------------------------------------------------------------------
# Files written by the app
# ----------------------------------------------------------------------------
PREDICTION_CACHE_FILE = OUTPUT_DIR / "prediction_cache.csv.gz"
LATEST_RESULTS_FILE = OUTPUT_DIR / "latest_results.csv"

# ----------------------------------------------------------------------------
# Colours (matched to the reference design)
# ----------------------------------------------------------------------------
COLORS = {
    "navy": "#0b2545",
    "navy_dark": "#081a33",
    "blue": "#1d6ff2",
    "negative": "#e53935",
    "neutral": "#2f80ed",
    "positive": "#16a34a",
    "emotion": "#7c4dff",
    "topic": "#0f9d8a",
    "warning": "#f97316",
    "muted": "#64748b",
}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": COLORS["negative"],
    "neutral": COLORS["neutral"],
    "positive": COLORS["positive"],
}
TOPICS = ["Healthcare", "Vaccination", "Lockdown", "Mental Health", "Economy", "General"]

DATASET_ATTRIBUTION = (
    "'Extended Covid Twitter Datasets' (files sample_data_only_USA.csv and "
    "sample_data_global.csv), supplied by the project author. The original source "
    "and licence are not stated inside the ZIP - add them here and in README.md "
    "before you submit or share the project."
)
