# Data folder

The app looks for datasets **in this folder** (including inside `.zip` files, even nested ones).

## What is bundled

`Extended_Covid_Twitter_Datasets.zip` - the dataset supplied by the project author, unchanged.
It contains two CSV files (UTF-8 with BOM, comma separated):

| File | Rows | Text column | Notes |
|---|---|---|---|
| `sample_data_only_USA.csv` | 3,000 (2,974 valid & unique) | `tweet` (**raw text**) | Dates 2020-08-20 to 2020-08-23, place column `near` |
| `sample_data_global.csv` | 409,395 (376,686 valid & unique) | `clean_tweet` (**stemmed, no raw text**) | 32 places, dates 2020-01-05 to 2021-12-06 |

Both files also carry lexicon-based sentiment columns (VADER, TextBlob, SentiWordNet) produced by the
dataset author. The app does **not** use these as predictions.

**Source / licence:** not stated inside the ZIP. Add the original source and licence to `config.py`
(`DATASET_ATTRIBUTION`) and to the main README before sharing or submitting.

## Using your own data

Drop a `.csv`, `.tsv`, `.json`, `.jsonl` or `.zip` here (or upload it on the *Bulk Analysis* page).
The text column is detected automatically (`tweet`, `text`, `full_text`, `content`, ...). If it is
ambiguous the app asks you to choose; you can also set `TEXT_COLUMN_OVERRIDE` in `config.py`.
A file that only contains tweet IDs cannot be analysed - the IDs are not post text.
