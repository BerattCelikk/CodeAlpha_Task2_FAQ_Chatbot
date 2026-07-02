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
) -> Optional[tuple[str, float]]:
    """Find the best FAQ answer for *user_query*.

    Args:
        user_query: Raw text from the user.
        processed_questions: Preprocessed FAQ question list.
        vectorizer: Fitted TF-IDF vectoriser.
        tfidf_matrix: Precomputed TF-IDF matrix of FAQ questions.
        threshold: Minimum similarity score to return a match.

    Returns:
        ``(answer_text, confidence)`` if the best score >= *threshold*,
        else ``None``.
    """
    query_vec = vectorizer.transform([preprocess(user_query)])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    if best_score < threshold:
        logger.info(
            "Low-confidence query (%.3f): %r", best_score, user_query[:80]
        )
        return None

    faqs = json.loads(st.session_state["faqs_json"])
    return faqs[best_idx]["answer"], best_score


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CodeAlpha FAQ Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 CodeAlpha SmartAssistant — FAQ Bot")
st.markdown(
    "Ask me anything about the **SmartAssistant** product. "
    "I'll match your question to our knowledge base."
)

# --- Eager-load data into session state (also primes cache) ---
faqs_data = load_faqs(_FAQ_PATH)
st.session_state["faqs_json"] = json.dumps(faqs_data)
_proc_qs, _vec, _tfidf = build_vectoriser(st.session_state["faqs_json"])

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

        if result:
            answer, confidence = result
            st.markdown(answer)
            st.caption(f"Match confidence: {confidence:.1%}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        else:
            st.markdown(_FALLBACK_MESSAGE)
            st.session_state.messages.append(
                {"role": "assistant", "content": _FALLBACK_MESSAGE}
            )
