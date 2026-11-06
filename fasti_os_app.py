"""FAST-I M2 RSF calculator, hosted at the existing Streamlit app URL.

The interface and the locked prediction engine are read verbatim from docs/.
The original Cox-era Streamlit assets remain intact in the repository.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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

    # Let the Streamlit container match the website's actual height, including
    # changes when a prediction becomes visible. No third-party site is loaded.
    resize_script = """
    <script>
      (() => {
        let lastHeight = 0;
        let queued = false;
        const updateHeight = () => {
          queued = false;
          const height = Math.ceil(Math.max(
            document.documentElement.scrollHeight,
            document.body.scrollHeight,
            document.querySelector('.page-shell')?.getBoundingClientRect().bottom || 0
          ));
          if (Math.abs(height - lastHeight) < 2) return;
          lastHeight = height;
          window.parent.postMessage({
            isStreamlitMessage: true,
            type: 'streamlit:setFrameHeight',
            height: height + 8
          }, '*');
        };
        const schedule = () => {
          if (!queued) {
            queued = true;
            requestAnimationFrame(updateHeight);
          }
        };
        window.addEventListener('resize', schedule);
        new MutationObserver(schedule).observe(document.body, {
          childList: true, subtree: true, attributes: true,
          attributeFilter: ['class', 'style', 'open']
        });
        if (window.ResizeObserver) {
          new ResizeObserver(schedule).observe(document.querySelector('.page-shell'));
        }
        schedule();
        window.addEventListener('load', schedule);
      })();
    </script>
    """
    if html.count("</body>") != 1:
        raise ValueError("Unexpected HTML body structure.")
    return html.replace("</body>", resize_script + "\n</body>", 1)


# Remove Streamlit's default margins and decoration. The static page contains
# its own responsive sidebar, banner, form, output cards and footer.
st.markdown(
    """
    <style>
      header[data-testid="stHeader"] {display: none;}
      footer {display: none;}
      [data-testid="stMainBlockContainer"] {
        padding: 0 !important;
        max-width: none !important;
      }
      [data-testid="stElementContainer"] {margin: 0 !important;}
      iframe[title="streamlit.components.v1.html"] {width: 100% !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    components.html(build_page(), height=1650, scrolling=False)
except (OSError, ValueError) as error:
    st.error(f"Unable to load the FAST-I M2 RSF calculator: {error}")
