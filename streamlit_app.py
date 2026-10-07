import logging
import asyncio
from requests.exceptions import ReadTimeout

import streamlit as st

from app.history import store_research
from app.research_graph import research

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#hello

st.set_page_config(page_title="GameVerse India", page_icon="🎮", layout="centered")
st.markdown(
    """
    <style>
    .stApp { background: #10131a; color: #f4f6fb; }
    .block-container { max-width: 900px; padding-top: 2.2rem; }
    h1 { color: #72e0a2; }
    [data-testid="stForm"] { border: 1px solid #343b49; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎮 GameVerse India")
st.caption("Research the Indian gaming ecosystem with grounded reports and live sources.")

with st.form("research_form"):
    question = st.text_area(
        "Research question",
        placeholder="How is India's online gaming market changing?",
        height=100,
    )
    submitted = st.form_submit_button("Research", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.error("Enter a research question before submitting.")
    else:
        try:
            with st.spinner("Checking reports and researching current sources..."):
                result = asyncio.run(research(question))
            persistence_failed = False
            try:
                store_research(result)
            except Exception:
                persistence_failed = True
                logger.exception("Research completed, but PostgreSQL history could not be saved.")
            st.subheader("Research brief")
            st.write(result["answer"])
            if persistence_failed:
                st.warning(
                    "Research completed, but it could not be saved to PostgreSQL. "
                    "Check DATABASE_URL and the database connection."
                )
            citations = result.get("citations", [])
            if citations:
                st.subheader("Sources")
                for source in citations:
                    title = source.get("title", "Source")
                    url = source.get("url", "")
                    label = f"[{source['id']}] {title}"
                    st.markdown(f"- [{label}]({url})" if url else f"- {label}")
            else:
                st.info("No source records were returned for this answer.")
        except ReadTimeout:
            logger.warning("NVIDIA request timed out in the Streamlit interface.", exc_info=True)
            st.error(
                "NVIDIA took too long to respond. Try again shortly. If this happens repeatedly, "
                "increase NVIDIA_REQUEST_TIMEOUT_SECONDS in .env and restart the app."
            )
        except Exception:
            logger.exception("Research failed in the Streamlit interface.")
            st.error(
                "Research could not be completed. Check your .env keys and service configuration, "
                "then review the terminal logs."
            )
