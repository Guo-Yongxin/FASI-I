"""FAST-I M2 RSF calculator, hosted at the existing Streamlit app URL.

The interface and the locked prediction engine are read verbatim from docs/.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="FAST-I HNSCC Overall Survival Risk Calculator",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SITE_DIR = Path(__file__).resolve().parent / "docs"


@st.cache_data(show_spinner=False)
def build_page() -> str:
    """Bundle the self-contained static site into Streamlit's HTML component."""
    html = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    css = (SITE_DIR / "styles.css").read_text(encoding="utf-8")

    stylesheet = '<link rel="stylesheet" href="./styles.css" />'
    if html.count(stylesheet) != 1:
        raise ValueError("Expected stylesheet reference is missing or duplicated.")
    html = html.replace(stylesheet, f"<style>\n{css}\n</style>", 1)

    for filename in ("FAST-I-banner.png", "time-interval.png"):
        old_src = f'src="./assets/{filename}"'
        if html.count(old_src) != 1:
            raise ValueError(f"Expected image reference is missing or duplicated: {filename}")
        encoded = base64.b64encode((SITE_DIR / "assets" / filename).read_bytes()).decode("ascii")
        html = html.replace(old_src, f'src="data:image/png;base64,{encoded}"', 1)

    for filename in ("model.js", "data/model-data.js", "app.js"):
        script = (SITE_DIR / filename).read_text(encoding="utf-8")
        if "</script" in script.lower():
            raise ValueError(f"Unexpected HTML script delimiter in {filename}")
        rx = re.compile(r'<script\s+defer\s+src="\./' + re.escape(filename) + r'\?v=\d+"\s*></script>')
        html, count = rx.subn(lambda _: f"<script>\n{script}\n</script>", html)
        if count != 1:
            raise ValueError(f"Expected script reference is missing or duplicated: {filename}")

    return html


# Keep the style-only element out of the main layout so it adds no top gap.
st.html(
    """
    <style>
      header[data-testid="stHeader"] {display: none;}
      footer {display: none;}
      [data-testid="stMainBlockContainer"] {
        padding: 0 !important;
        max-width: none !important;
      }
      [data-testid="stElementContainer"] {margin: 0 !important;}
    </style>
    """
)

try:
    st.iframe(build_page(), width="stretch", height="content")
except (OSError, ValueError) as error:
    st.error(f"Unable to load the FAST-I M2 RSF calculator: {error}")
