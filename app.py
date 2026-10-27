"""Entry point:  streamlit run app.py   (or double-click run.bat)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="COVID-19 Social Media Intelligence Dashboard", page_icon="🦠",
                   layout="wide", initial_sidebar_state="expanded")

from src.ui import NAV  # noqa: E402

pages = [st.Page(path, title=label, icon=icon, default=(i == 0), url_path=Path(path).stem.split("_", 1)[1].lower())
         for i, (path, label, icon) in enumerate(NAV)]
st.navigation(pages, position="hidden").run()
