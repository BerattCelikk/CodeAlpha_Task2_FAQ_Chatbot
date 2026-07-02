"""
Enterprise-Grade FAQ Chatbot
-------------------------------
CodeAlpha Internship — Task 2

An NLP-powered FAQ chatbot using Streamlit's native chat elements.
Preprocessing with NLTK (Porter stemmer + stopword removal), matching
with scikit-learn's TfidfVectorizer and cosine similarity.
"""

import json
import logging
import re
from typing import Any, Optional

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("faq_chatbot")

# ---------------------------------------------------------------------------
# NLTK resource downloads (idempotent)
# ---------------------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FALLBACK_THRESHOLD: float = 0.2
_FALLBACK_MESSAGE: str = (
    "I'm not entirely sure I understood your question. Could you rephrase?"
)
_FAQ_PATH: str = "faqs.json"


# ---------------------------------------------------------------------------
# JSON validation
# ---------------------------------------------------------------------------
def validate_faqs(data: Any) -> list[dict[str, str]]:
    """Validate that *data* is a list of ``{"question": …, "answer": …}`` dicts.

    Args:
        data: Parsed JSON content.

    Returns:
        The validated list of FAQ dicts.

    Raises:
        ValueError: If the structure is malformed.
    """
    if not isinstance(data, list):
        raise ValueError("FAQs must be a JSON array.")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary.")
        if "question" not in item or "answer" not in item:
            raise ValueError(f"Item {i} missing 'question' or 'answer' key.")
        if not isinstance(item["question"], str) or not isinstance(item["answer"], str):
            raise ValueError(f"Item {i} has non-string question or answer.")
        if not item["question"].strip() or not item["answer"].strip():
            raise ValueError(f"Item {i} has empty question or answer.")
    return data


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_faqs(path: str) -> list[dict[str, str]]:
    """Load and validate the FAQ JSON file.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        List of validated FAQ dictionaries.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw: Any = json.load(f)
        faqs = validate_faqs(raw)
        logger.info("Loaded %d FAQs from %s", len(faqs), path)
        return faqs
    except FileNotFoundError:
        logger.critical("FAQ file not found: %s", path)
        st.error(f"Configuration error: {path} is missing. Please reinstall the app.")
        st.stop()
    except json.JSONDecodeError as exc:
        logger.critical("Invalid JSON in %s: %s", path, exc)
        st.error(f"Configuration error: {path} contains invalid JSON.")
        st.stop()
    except ValueError as exc:
        logger.critical("FAQ validation failed: %s", exc)
        st.error(f"Configuration error: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# NLP preprocessing
# ---------------------------------------------------------------------------
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Tokenise, lowercase, stem, and remove stopwords.

    Args:
        text: Raw input string.

    Returns:
        Preprocessed string suitable for TF-IDF vectorisation.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_words]
    return " ".join(tokens)


def tokenize_for_display(text: str) -> list[str]:
    """Return the stemmed tokens of *text* for display in the diagnostics UI.

    Args:
        text: Raw input string.

    Returns:
        List of processed tokens after lowercasing and stemming.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return sorted(set(_stemmer.stem(t) for t in tokens if t not in _stop_words))


# ---------------------------------------------------------------------------
# Cached vectoriser building
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_vectoriser(
    faqs_json: str,
) -> tuple[list[str], "TfidfVectorizer", "Any"]:  # noqa: F821
    """Preprocess FAQ questions and fit a TF-IDF vectoriser.

    *faqs_json* is a JSON string so that Streamlit can hash it reliably.

    Returns:
        ``(processed_questions, vectorizer, tfidf_matrix)``.
    """
    faqs: list[dict[str, str]] = json.loads(faqs_json)
    questions = [item["question"] for item in faqs]
    processed = [preprocess(q) for q in questions]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed)
    logger.info(
        "TF-IDF matrix built: %d questions, %d features",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
    )
    return processed, vectorizer, tfidf_matrix


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------
def get_best_answer(
    user_query: str,
    processed_questions: list[str],
    vectorizer: TfidfVectorizer,
    tfidf_matrix: "Any",  # sparse matrix
    threshold: float = _FALLBACK_THRESHOLD,
) -> dict:
    """Find the best FAQ answer for *user_query*.

    Args:
        user_query: Raw text from the user.
        processed_questions: Preprocessed FAQ question list.
        vectorizer: Fitted TF-IDF vectoriser.
        tfidf_matrix: Precomputed TF-IDF matrix of FAQ questions.
        threshold: Minimum similarity score to return a match.

    Returns:
        A dict with keys:
        - ``answer``: matched answer text, or ``None`` if below threshold.
        - ``matched_question``: the original FAQ question text.
        - ``best_score``: the cosine similarity score.
        - ``is_match``: ``True`` if *best_score >= threshold*.
    """
    query_vec = vectorizer.transform([preprocess(user_query)])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    faqs = json.loads(st.session_state["faqs_json"])
    matched_question = faqs[best_idx]["question"]
    answer = faqs[best_idx]["answer"] if best_score >= threshold else None

    logger.info(
        "Query %r matched intent %r with score %.3f",
        user_query[:80],
        matched_question,
        best_score,
    )

    return {
        "answer": answer,
        "matched_question": matched_question,
        "best_score": best_score,
        "is_match": best_score >= threshold,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CodeAlpha FAQ Chatbot",
    page_icon="🤖",
    layout="centered",
)

# --- Inject custom CSS to hide Streamlit chrome and polish layout ---
st.markdown(
    """
    <style>
        /* 1. Hide all Streamlit Chrome completely */
        #MainMenu, header, footer {visibility: hidden;}
        .stAppDeployButton {display: none;}
        [data-testid="collapsedControl"] {display: none !important;} /* Hides sidebar toggle arrow */
        section[data-testid="stSidebar"] {display: none !important;} /* Force kill sidebar */

        /* 2. Global Clean Background & Typography */
        .stApp {
            background-color: #fdfdfd !important;
        }
        .block-container {
            max-width: 850px !important; /* Center focus like ChatGPT */
            padding-top: 3rem !important;
            padding-bottom: 120px !important;
        }
        h1, h2, h3, p, span, div {
            color: #1a1a1a !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* 3. Title Styling */
        h1 {
            text-align: center;
            font-weight: 800 !important;
            letter-spacing: -1px;
            margin-bottom: 2rem !important;
        }

        /* 4. Fix the Bottom Input Bar (Kill the dark mode strip) */
        [data-testid="stBottomBlock"] {
            background: transparent !important; 
        }
        [data-testid="stChatInput"] {
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 24px !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01) !important;
            padding: 4px !important;
            margin-bottom: 1rem !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #111827 !important;
        }

        /* 5. Premium Chat Bubbles */
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0.5rem 0 !important;
        }
        .stChatMessage [data-testid="stMarkdownContainer"] {
            background-color: #ffffff;
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            border: 1px solid #f3f4f6;
        }

        /* 6. NLP Diagnostics Expander - Techy but elegant */
        [data-testid="stExpander"] {
            background-color: #fafafa !important;
            border: 1px dashed #d1d5db !important;
            border-radius: 12px !important;
            margin-top: 0.5rem;
        }
        [data-testid="stExpander"] summary {
            color: #6b7280 !important;
            font-size: 0.9rem !important;
        }
        code {
            color: #8b5cf6 !important;
            background-color: #f3f0ff !important;
            border-radius: 6px !important;
            padding: 0.2rem 0.5rem !important;
            font-size: 0.85em !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Eager-load data into session state (also primes cache) ---
faqs_data = load_faqs(_FAQ_PATH)
st.session_state["faqs_json"] = json.dumps(faqs_data)
_proc_qs, _vec, _tfidf = build_vectoriser(st.session_state["faqs_json"])

# --- Main interface ---
st.title("🤖 CodeAlpha SmartAssistant — FAQ Bot")
st.markdown(
    "Ask me anything about the **SmartAssistant** product. "
    "I'll match your question to our knowledge base."
)
st.divider()

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm the SmartAssistant FAQ bot. "
                "Ask me about features, pricing, setup, integrations, and more!"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input ---
if prompt := st.chat_input("Type your question here…"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base…"):
            result = get_best_answer(
                prompt, _proc_qs, _vec, _tfidf
            )

        if result["is_match"]:
            st.markdown(result["answer"])
            st.caption(f"Match confidence: {result['best_score']:.1%}")
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"]}
            )
        else:
            st.markdown(_FALLBACK_MESSAGE)
            st.session_state.messages.append(
                {"role": "assistant", "content": _FALLBACK_MESSAGE}
            )

        # --- NLP Diagnostics Expander ---
        with st.expander("⚙️ NLP Engine Diagnostics"):
            tokens = tokenize_for_display(prompt)
            st.markdown(f"**User Input Tokens:** `{tokens}`")
            st.markdown(f"**Matched Intent:** _{result['matched_question']}_")
            st.markdown(f"**Confidence Score:** {result['best_score']:.1%}")

            if not result["is_match"]:
                st.warning("⚠️ Confidence was too low for a definitive match.")
