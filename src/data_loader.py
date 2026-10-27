"""Dataset discovery, column detection, loading and cleaning.

Designed for the supplied 'Extended Covid Twitter Datasets' ZIP but works for any
CSV / TSV / JSON / JSONL file (optionally inside one or more ZIP archives).

What the supplied dataset looks like (verified by inspection):
  * sample_data_only_USA.csv   - 3,000 rows, has a RAW text column named `tweet`.
  * sample_data_global.csv     - 409,395 rows, NO raw text. Only `clean_tweet`
                                 (lower-cased, stop-words removed, *stemmed*).
Both are UTF-8 with a BOM, comma separated, `created_at` is YYYY-MM-DD and the
place column is `near`. There is no platform column.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

import config
from src.preprocessing import is_valid_text, normalize_for_model, normalize_key

log = logging.getLogger(__name__)

RAW_TEXT_NAMES = ["tweet", "text", "full_text", "tweet_text", "content", "body",
                  "post", "message", "original_text", "comment", "status"]
PROCESSED_TEXT_NAMES = ["clean_tweet", "cleaned_text", "clean_text", "processed_text",
                        "text_clean", "lemma_sentence"]
DATE_NAMES = ["created_at", "date", "timestamp", "datetime", "time", "tweet_date", "posted_at"]
LOCATION_NAMES = ["near", "country", "location", "place", "region", "geo"]
PLATFORM_NAMES = ["platform", "source", "site"]


class DatasetError(Exception):
    """Base class: message is safe to show directly to the user."""


class NoTextColumnError(DatasetError):
    pass


class AmbiguousTextColumnError(DatasetError):
    def __init__(self, message: str, candidates: list[str]):
        super().__init__(message)
        self.candidates = candidates


@dataclass(frozen=True)
class DataSource:
    label: str
    path: Path
    members: tuple[str, ...] = ()          # chain of names inside (nested) ZIPs
    size_bytes: int = 0

    @property
    def key(self) -> str:
        return f"{self.path}::{'!'.join(self.members)}"

    @property
    def suffix(self) -> str:
        return Path(self.members[-1] if self.members else self.path.name).suffix.lower()


@dataclass
class ColumnDetection:
    text: str
    text_is_processed: bool
    reason: str
    date: str | None = None
    location: str | None = None
    platform: str | None = None
    all_columns: list[str] = field(default_factory=list)


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    stats: dict
    detection: ColumnDetection
    source: DataSource
    date_usable: bool


# ----------------------------------------------------------------------------
# Discovery / opening
# ----------------------------------------------------------------------------
def _is_data_file(name: str) -> bool:
    return Path(name).suffix.lower() in config.SUPPORTED_DATA_SUFFIXES and not Path(name).name.startswith(("._", "~"))


def _walk_zip(zf: zipfile.ZipFile, prefix: tuple[str, ...], depth: int) -> Iterator[tuple[tuple[str, ...], int]]:
    for info in zf.infolist():
        if info.is_dir() or "__MACOSX" in info.filename:
            continue
        name = info.filename
        if _is_data_file(name):
            yield prefix + (name,), info.file_size
        elif name.lower().endswith(".zip") and depth < 3:
            try:
                with zf.open(info) as fh:
                    inner = zipfile.ZipFile(io.BytesIO(fh.read()))
                yield from _walk_zip(inner, prefix + (name,), depth + 1)
            except zipfile.BadZipFile:
                log.warning("Skipping corrupt nested ZIP %s", name)


def discover_sources(data_dir: Path = config.DATA_DIR) -> list[DataSource]:
    """Find every usable data file in `data_dir`, including inside ZIPs."""
    found: list[DataSource] = []
    if not data_dir.exists():
        return found
    for p in sorted(data_dir.iterdir()):
        if p.is_dir():
            continue
        if p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p) as zf:
                    for members, size in _walk_zip(zf, (), 0):
                        label = f"{Path(members[-1]).name}  (in {p.name})"
                        found.append(DataSource(label, p, members, size))
            except zipfile.BadZipFile:
                log.warning("Not a valid ZIP: %s", p)
        elif _is_data_file(p.name) and p.name.lower() != "readme.md":
            found.append(DataSource(p.name, p, (), p.stat().st_size))
    return found


@contextmanager
def open_source(source: DataSource):
    """Yield a binary file object for the data file (stream, no extraction)."""
    if not source.members:
        with open(source.path, "rb") as fh:
            yield fh
        return
    zf = zipfile.ZipFile(source.path)
    opened = [zf]
    try:
        current = zf
        for name in source.members[:-1]:          # descend into nested ZIPs
            with current.open(name) as fh:
                current = zipfile.ZipFile(io.BytesIO(fh.read()))
            opened.append(current)
        with current.open(source.members[-1]) as fh:
            yield fh
    finally:
        for z in reversed(opened):
            z.close()


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def _iter_chunks(source: DataSource, chunksize: int, nrows: int | None = None,
                 usecols=None) -> Iterator[pd.DataFrame]:
    suffix = source.suffix
    with open_source(source) as fh:
        if suffix in (".csv", ".tsv"):
            reader = pd.read_csv(
                fh, sep="\t" if suffix == ".tsv" else ",", dtype=str, keep_default_na=False,
                encoding="utf-8-sig", encoding_errors="replace", usecols=usecols,
                chunksize=chunksize, nrows=nrows, on_bad_lines="skip",
            )
            yield from reader
        elif suffix == ".jsonl":
            reader = pd.read_json(fh, lines=True, dtype=False, chunksize=chunksize, nrows=nrows)
            for chunk in reader:
                yield chunk.astype(object).where(chunk.notna(), "").astype(str)
        elif suffix == ".json":
            raw = json.loads(fh.read().decode("utf-8-sig", errors="replace"))
            if isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, list):
                        raw = v
                        break
            df = pd.json_normalize(raw).astype(object)
            df = df.where(df.notna(), "").astype(str)
            for i in range(0, len(df), chunksize):
                yield df.iloc[i:i + chunksize]
        elif suffix == ".txt":
            lines = fh.read().decode("utf-8-sig", errors="replace").splitlines()
            df = pd.DataFrame({"text": lines})
            for i in range(0, len(df), chunksize):
                yield df.iloc[i:i + chunksize]
        else:
            raise DatasetError(f"Unsupported file type: {suffix}")


def _lower_map(columns) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in columns}


def _looks_textual(series: pd.Series) -> tuple[float, float]:
    """Return (mean length, share of values that are list-like/numeric)."""
    s = series.astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return 0.0, 1.0
    junk = (s.str.startswith("[") | s.str.fullmatch(r"[-+]?\d+(\.\d+)?")).mean()
    return float(s.str.len().mean()), float(junk)


def detect_columns(sample: pd.DataFrame) -> ColumnDetection:
    """Choose the text column (and optional date/location/platform columns).

    Rules, in order:
      1. A well-known RAW text column name (tweet, text, full_text, ...).
      2. A well-known PROCESSED text column (clean_tweet, ...) - flagged as processed.
      3. Heuristic: the free-text-looking column with the longest average length.
         If two candidates are too close to call, raise AmbiguousTextColumnError.
    """
    cols = list(sample.columns)
    lm = _lower_map(cols)

    def usable(col) -> bool:
        mean_len, junk = _looks_textual(sample[col])
        return mean_len >= 8 and junk < 0.5

    text, processed, reason = None, False, ""
    for name in RAW_TEXT_NAMES:
        if name in lm and usable(lm[name]):
            text, reason = lm[name], f"Column '{lm[name]}' matches a known raw-text column name."
            break
    if text is None:
        for name in PROCESSED_TEXT_NAMES:
            if name in lm and usable(lm[name]):
                text, processed = lm[name], True
                reason = (f"No raw-text column exists; using the pre-processed column "
                          f"'{lm[name]}' (lower-cased, stop-words removed, stemmed).")
                break
    if text is None:
        scored = []
        for c in cols:
            if any(tok in str(c).lower() for tok in ("id", "date", "time", "token", "pos_tag")):
                continue
            mean_len, junk = _looks_textual(sample[c])
            if junk < 0.5 and mean_len >= 15:
                scored.append((mean_len, c))
        scored.sort(reverse=True)
        if not scored:
            id_like = [c for c in cols if "id" in str(c).lower()]
            hint = (" The file looks like it only contains IDs/metadata (e.g. " + ", ".join(map(str, id_like[:3])) +
                    "). Tweet IDs are not post text - the original tweets would have to be 'hydrated' "
                    "first, which this app does not do.") if id_like else ""
            raise NoTextColumnError("Could not find a column containing post text." + hint +
                                    f" Columns found: {', '.join(map(str, cols))}.")
        if len(scored) > 1 and scored[1][0] > 0.8 * scored[0][0]:
            raise AmbiguousTextColumnError(
                "Several columns could contain the post text; please choose one in the "
                "'Text column' box (or set TEXT_COLUMN_OVERRIDE in config.py): "
                + ", ".join(str(c) for _, c in scored[:5]), [str(c) for _, c in scored[:5]])
        text, reason = scored[0][1], f"Chose '{scored[0][1]}' as the longest free-text column (heuristic)."

    def pick(names):
        for n in names:
            if n in lm and lm[n] != text:
                return lm[n]
        return None

    return ColumnDetection(text=text, text_is_processed=processed, reason=reason,
                           date=pick(DATE_NAMES), location=pick(LOCATION_NAMES),
                           platform=pick(PLATFORM_NAMES), all_columns=[str(c) for c in cols])


def inspect_source(source: DataSource, text_column: str | None = None, nrows: int = 500) -> tuple[pd.DataFrame, ColumnDetection]:
    """Read a small sample and detect columns (fast, even for 200 MB files)."""
    sample = next(_iter_chunks(source, chunksize=nrows, nrows=nrows), None)
    if sample is None or sample.empty:
        raise DatasetError("The selected file is empty.")
    override = text_column or config.TEXT_COLUMN_OVERRIDE
    if override:
        lm = _lower_map(sample.columns)
        if override.strip().lower() not in lm:
            raise DatasetError(f"Text column '{override}' not found. Columns: {', '.join(map(str, sample.columns))}")
        col = lm[override.strip().lower()]
        # detect the other columns normally, then force the text column
        try:
            det = detect_columns(sample)
        except DatasetError:
            det = ColumnDetection(text=col, text_is_processed=False, reason="", all_columns=[str(c) for c in sample.columns])
        det.text = col
        det.text_is_processed = col.lower() in PROCESSED_TEXT_NAMES
        det.reason = f"Using column '{col}' (chosen manually)."
        lm2 = _lower_map(sample.columns)
        for attr, names in (("date", DATE_NAMES), ("location", LOCATION_NAMES), ("platform", PLATFORM_NAMES)):
            if getattr(det, attr) in (None, col):
                setattr(det, attr, next((lm2[n] for n in names if n in lm2 and lm2[n] != col), None))
        return sample, det
    return sample, detect_columns(sample)


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse dates without warnings: ISO first (fast path), then a tolerant fallback."""
    raw = values.astype(str).str.strip()
    parsed = pd.to_datetime(raw, format="ISO8601", errors="coerce", utc=True)
    retry = parsed.isna() & raw.ne("")
    if retry.any():
        parsed[retry] = pd.to_datetime(raw[retry], format="mixed", errors="coerce", utc=True)
    return parsed.dt.tz_localize(None)


# ----------------------------------------------------------------------------
# Full load
# ----------------------------------------------------------------------------
def load_dataset(source: DataSource, text_column: str | None = None, chunksize: int = 50_000) -> LoadedDataset:
    """Read the whole file (only the needed columns), clean and de-duplicate it.

    Returns a DataFrame with columns: text, source_row and (when present)
    date, location, platform. The stats dict reports every row removed.
    """
    _, det = inspect_source(source, text_column)
    wanted = [c for c in (det.text, det.date, det.location, det.platform) if c]
    usecols = wanted if source.suffix in (".csv", ".tsv") else None

    pieces, total_rows, n_invalid = [], 0, 0
    for chunk in _iter_chunks(source, chunksize, usecols=usecols):
        part = pd.DataFrame({"text": chunk[det.text].astype(str)})
        part["source_row"] = range(total_rows, total_rows + len(chunk))
        if det.date:
            part["date"] = _parse_dates(chunk[det.date])
        if det.location:
            part["location"] = chunk[det.location].astype(str).str.strip().replace("", pd.NA)
        if det.platform:
            part["platform"] = chunk[det.platform].astype(str).str.strip().replace("", pd.NA)
        total_rows += len(chunk)
        part["text"] = part["text"].map(normalize_for_model)
        ok = part["text"].map(is_valid_text)
        n_invalid += int((~ok).sum())
        pieces.append(part[ok])

    df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["text", "source_row"])
    key = df["text"].map(normalize_key)
    dup = key.duplicated(keep="first")
    n_dup = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    del key, pieces

    date_usable = bool("date" in df and df["date"].notna().mean() >= 0.5)
    stats = {"rows_in_file": total_rows, "empty_or_invalid": n_invalid,
             "duplicates": n_dup, "valid": len(df)}
    log.info("Loaded %s: %s", source.label, stats)
    return LoadedDataset(df=df, stats=stats, detection=det, source=source, date_usable=date_usable)


def sample_dataset(df: pd.DataFrame, limit: int, mode: str = "random", seed: int = 42) -> pd.DataFrame:
    """Return `limit` posts (0 or >= len means all). 'random' is reproducible."""
    if limit is None or limit <= 0 or limit >= len(df):
        return df.copy()
    if mode == "first":
        return df.head(limit).copy()
    return df.sample(n=limit, random_state=seed).sort_index().reset_index(drop=True)
