"""Point Group Finder on Streamlit. Deploy with THIS file as the main file (streamlit_app.py).

All the work happens in docs/index.html: Python (Pyodide) runs inside the visitor's browser,
so Streamlit only has to serve the page. Run `python build_web.py` after changing
symmetry_core.py, web/bridge.py or web/index.src.html, then commit docs/index.html too.
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Point Group Finder", page_icon="⚛️", layout="wide")
st.markdown(
    "<style>.block-container{padding:0 !important;max-width:100% !important}"
    "[data-testid='stHeader']{display:none}</style>",
    unsafe_allow_html=True,
)
page = (Path(__file__).parent / "docs" / "index.html").read_text(encoding="utf-8")
components.html(page, height=1000, scrolling=True)
