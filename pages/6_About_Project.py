import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import config
from src import ui
from src.misinformation import DISCLAIMER
from src.topic_classifier import TOPIC_KEYWORDS

ui.render_shell(5)
ui.section_title("About Project")

st.markdown(f"""
### Objective
Analyse how people discussed COVID-19 on social media. Posts are scored for **sentiment** and **emotion** with
pretrained language models, grouped into **topics** with transparent keyword rules, and summarised in an interactive dashboard.
No model was trained from scratch.

### Sentiment analysis
Sentiment shows whether a post is **positive, neutral or negative**. The app uses
[`{config.SENTIMENT_MODEL}`](https://huggingface.co/{config.SENTIMENT_MODEL}), a RoBERTa-base model trained on tweets.
It returns three probabilities that sum to 1; the highest is the predicted label and its probability is the *confidence*.

### Emotion detection
Emotion goes deeper than polarity (e.g. *fear* vs *anger* are both negative). The app uses
[`{config.EMOTION_MODEL}`](https://huggingface.co/{config.EMOTION_MODEL}) (anger, disgust, fear, joy, neutral, sadness, surprise).
The label list is read from the model itself.

### Topic classification
A **rule-based** classifier counts how many distinct keywords of each topic appear in a post. The topic with the most matches wins; ties
go to the more specific topic (Vaccination, Lockdown, Mental Health, Economy, then Healthcare); no match means **General**.
Because the rules are simple, some posts will be mislabelled - treat topics as helpful groupings, not ground truth.
""")
with st.expander("See the keyword rules"):
    for t, kws in TOPIC_KEYWORDS.items():
        st.markdown(f"**{t}:** " + ", ".join(f"`{k}`" for k in kws))
    st.caption("A trailing * means 'any word starting with'. Edit src/topic_classifier.py to change the rules.")

st.markdown(f"""
### How OCR works
When you upload a screenshot, **EasyOCR** (a neural text detector + recogniser) finds text regions and reads the characters. The app joins
the lines in reading order and removes obvious screenshot UI text such as like counts. OCR can make mistakes with small, blurry or stylised text,
so the text is shown in an editable box before analysis.

### Preliminary misinformation flag
The orange card is a **keyword screen** for phrases often seen in debunked claims (for example certain conspiracy terms). {DISCLAIMER}
A flag only means a human may want to check the post; no flag does not mean a post is correct.

### Dataset
{config.DATASET_ATTRIBUTION}

* `sample_data_only_USA.csv` - contains the raw tweet text (`tweet`) and is the default for bulk analysis.
* `sample_data_global.csv` - contains **only pre-processed (stemmed) text**, so it works but gives less reliable results.
* Both files also contain lexicon scores (VADER / TextBlob / SentiWordNet) added by the dataset author. This app does **not** use them as predictions.

### Limitations
* Language models reflect biases and errors of their training data and can misread sarcasm, slang, humour, and non-English text.
* Models were not trained specifically on pandemic language; confident predictions can still be wrong.
* Keyword topics and the misinformation flag are heuristics.
* Results describe the analysed sample, not all people or all of Twitter/X.
* Dataset tweets may contain personal opinions; do not use this tool to profile individuals.
""")
