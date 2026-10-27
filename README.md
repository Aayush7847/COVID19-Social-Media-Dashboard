<img src="assets/virus.svg" width="64" align="left" alt="">

# COVID-19 Social Media Intelligence Dashboard

Sentiment analysis · Emotion detection · Topic classification for COVID-19 social media posts.
A single Streamlit app: analyse one post (screenshot OCR / pasted text / public link) or a whole dataset, then explore
interactive charts. Uses **pretrained** Hugging Face models - nothing is trained from scratch. No API keys, no database.

---

## 1. Quick start (Windows)

1. **Install Python 3.11** from <https://www.python.org/downloads/> - tick **"Add python.exe to PATH"**.
2. **Extract the ZIP** (right-click → *Extract All*) to any folder, e.g. `C:\Projects\COVID19_Social_Media_Dashboard`.
3. **Double-click `run.bat`.** The first run creates `.venv`, installs the requirements (5-15 min, PyTorch is large) and starts the app.
4. Your browser opens at <http://localhost:8501>. If not, open that address yourself.
5. **First analysis:** the two language models (and, for screenshots, the OCR files) are downloaded once - **internet needed**, a few hundred MB.
   After that they are cached on your computer.

Prefer PowerShell? `Set-ExecutionPolicy -Scope Process Bypass` then `.\run.ps1`.
Manual start (any OS): `python -m venv .venv`, activate it, `pip install -r requirements.txt`, `streamlit run app.py`.

### Suggested demo order
1. **Bulk Analysis** → keep the default 300 posts → *Start analysis* (shows live progress and speed).
2. **Dashboard** → the three dataset charts at the bottom are now filled. Try a pasted post or screenshot at the top.
3. **Topic Insights**, **All Results** (search / filter / CSV export), **About Project**.

---

## 2. The dataset that is integrated

`data/Extended_Covid_Twitter_Datasets.zip` is your supplied ZIP, **unchanged**. The app reads CSV files straight out of the ZIP (no extraction needed).

| File | Valid unique posts | Text | Used for |
|---|---|---|---|
| `sample_data_only_USA.csv` | 2,974 (of 3,000 rows) | **Raw tweet text** (`tweet`) | **Default** for Bulk Analysis - best accuracy |
| `sample_data_global.csv` | 376,686 (of 409,395 rows) | **Only stemmed text** (`clean_tweet`, e.g. "posit", "comun") | Optional, for large runs (up to ~100k posts) |

Important consequences:

* **The raw-text file has 2,974 usable posts - fewer than the "~100,000" target.** 100,000 posts exist only in the *global* file, whose text is
  stemmed and stripped of stop-words. The language models were trained on normal tweets, so results on that file are **less reliable**
  (the app shows a warning when you select it). You choose the file and the number of posts; nothing is silently truncated, and every count on screen
  (available / selected / analysed / failed / shown in charts) is reported separately.
* Text column detection: `tweet` (raw) is preferred over `clean_tweet` (processed). Date column `created_at`, place column `near`. There is **no platform column**, so none is shown.
* Both files also contain VADER/TextBlob/SentiWordNet columns added by the dataset author; the app ignores them and uses the transformer models.
* **Source & licence:** the ZIP does not state them. Please add the original source/licence to `config.py → DATASET_ATTRIBUTION` and here before sharing or submitting.

**Use another dataset:** drop a `.csv/.tsv/.json/.jsonl/.zip` into `data/` (or upload on *Bulk Analysis*). The text column is auto-detected; override it on the page or via `TEXT_COLUMN_OVERRIDE` in `config.py`. Files containing only tweet IDs are rejected with an explanation.

### Processing limit & speed
*Bulk Analysis → "Posts to analyse"* (default **300**, change the default in `config.py → DEFAULT_PROCESS_LIMIT`; `0` = all).
Two transformer models run on every post, so a CPU laptop needs real time for big runs - the page shows live posts/second and time left after the first batch.
Start small. Predictions are saved in `outputs/prediction_cache.csv.gz`, so re-running the same posts is instant and a cancelled run (**Stop** at the top-right) keeps finished batches.

---

## 3. Models and methods

| Part | What | Notes |
|---|---|---|
| Sentiment | [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | negative / neutral / positive probabilities; labels are read from the model config |
| Emotion | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | anger, disgust, fear, joy, neutral, sadness, surprise |
| Topic | Keyword rules in `src/topic_classifier.py` | Healthcare, Vaccination, Lockdown, Mental Health, Economy, General. Most distinct matches wins; ties → more specific topic; none → General. **Rule-based, not ground truth.** |
| Keywords | Frequency based, stop-words removed | `src/keyword_extractor.py` |
| OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Result is editable before analysis |
| Misinformation card | Keyword screen | **Preliminary only - not a fact-checker.** |

Change a model: set `SENTIMENT_MODEL` / `EMOTION_MODEL` in `config.py` (or the env vars `COVID_DASH_SENTIMENT_MODEL`, `COVID_DASH_EMOTION_MODEL`).
Single-post and bulk analysis use the same code (`src/analyzer.py`).

**Twitter/X link mode** uses X's official public *oEmbed* endpoint (no key, no scraping). It can only work for public posts and may fail (private/deleted post, no internet, X policy changes);
the app then asks you to paste text or upload a screenshot. See *Testing status* (section 8) - this path could not be tested live.

---

## 4. Everyday use

* **Export:** *Bulk Analysis* and *All Results* have **Download CSV** buttons (UTF-8 with BOM so Excel shows accents/emoji correctly). Files are also in `outputs/`.
* **CPU-only:** the Windows PyTorch build from PyPI is CPU-only already. To force CPU on a GPU machine: set the env var `COVID_DASH_FORCE_CPU=1`.
* **Settings** (`config.py`): batch size, text/token limits, OCR confidence, colours, topic list. Env vars `COVID_DASH_LIMIT`, `COVID_DASH_BATCH` also work.
* **Move the folder:** all paths are relative; just move/copy the folder (if `.venv` misbehaves, delete it and run `run.bat` again).

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| "Python was not found" | Install Python 3.11 and tick *Add to PATH*; reopen the window. |
| Installation fails / is very slow | Check internet; delete `.venv`; run `run.bat` again. Corporate proxies may need `pip` proxy settings. |
| "Could not load the model…" on first analysis | The model download needs internet/firewall access to huggingface.co. Retry; the rest of the app keeps working. |
| OCR error on first screenshot | OCR files download from GitHub on first use. Meanwhile type/paste the text instead. |
| Port 8501 busy | `.venv\Scripts\python -m streamlit run app.py --server.port 8502` |
| App feels slow | Lower *Posts to analyse*; use batch size 16-32; close other programs. |
| Charts empty | Run *Bulk Analysis* first; single-post charts need *Analyze Post*. |
| Out of memory with the global file | Reading 400k rows needs ~1 GB RAM; use the USA file or a smaller sample. |

## 6. Project layout

```
app.py                  entry point + navigation          config.py   all settings
pages/                  1_Dashboard … 6_About_Project     src/ui.py   theme, header, sidebar, cards
src/data_loader.py      ZIP/CSV/JSON reading, column detection, cleaning
src/preprocessing.py    text normalisation, dedup keys    src/storage.py  prediction cache + CSV export
src/sentiment.py, emotion.py, model_utils.py              pretrained model wrappers
src/topic_classifier.py, keyword_extractor.py, misinformation.py   rule-based parts
src/ocr.py, url_fetcher.py                                 screenshot + optional link input
src/bulk_processor.py   batching, progress, caching       src/charts.py   Plotly figures
tests/                  pytest suite                       data/       dataset ZIP + notes
outputs/                results & cache written at runtime
```

## 7. References
* Cardiff NLP Twitter RoBERTa sentiment model: <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest> (Loureiro et al., *TimeLMs*, 2022)
* Emotion model: <https://huggingface.co/j-hartmann/emotion-english-distilroberta-base> (Hartmann, 2022)
* EasyOCR: <https://github.com/JaidedAI/EasyOCR> · Transformers: <https://huggingface.co/docs/transformers> · Streamlit: <https://docs.streamlit.io> · Plotly: <https://plotly.com/python/>
* Dataset: "Extended Covid Twitter Datasets" - supplied by the project author; **add original source/licence**.
* Model licences/terms are on their Hugging Face pages - check them before any commercial use.

## 8. Testing status (honest summary)

Run the tests yourself: `.venv\Scripts\python -m pytest tests -q` (the model tests skip automatically if torch is missing).

**Tested** (Linux, Python 3.12, Streamlit 1.65, pandas 3.0, plotly 7.1, torch 2.14, transformers 4.57):
automated tests for data loading/cleaning/dedup/column detection (incl. ZIP, nested ZIP, JSON, BOM, quoted/multiline CSV, ID-only files, and the bundled dataset),
topic rules, keywords, label mapping and confidence maths, the model wrapper (with a tiny *random* local model - proves the code path, not prediction quality),
bulk batching/caching/stop/failure handling, CSV export, empty-data charts, upload validation, OCR layout helpers, and headless execution of every page.
In a real browser: screenshot upload -> real EasyOCR -> editable text -> Analyze; navigation between pages; charts rendered from processed data.

**Not tested / limited:**
* **The real sentiment and emotion models were never run** by the author (Hugging Face was unreachable). Their first download happens on your machine.
* `run.bat` / `run.ps1` were written carefully but **not executed** (no Windows available). Python 3.11 on Windows is untested.
* **Twitter/X link retrieval** is untested against the live service.
* Visual match to the reference was checked in Chromium on Linux only.
* `docs/screenshot_ocr_run.png` shows the app in that sandbox: OCR and rule-based parts working, model cards "Unavailable" because models could not download.

## 9. Limitations
Models can be biased and wrong (sarcasm, slang, humour, non-English text). Topics and the misinformation flag are heuristics.
Results describe the analysed sample only. Do not use the tool to profile individuals.
