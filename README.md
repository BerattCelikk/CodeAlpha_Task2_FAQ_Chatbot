# CodeAlpha FAQ Chatbot

**CodeAlpha Internship — Task 2**  
An enterprise-grade, NLP-powered FAQ chatbot that uses **TF-IDF vector similarity** to answer questions about the fictional **CodeAlpha SmartAssistant** product.

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Streamlit App                       │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ faqs.json │ → │  @st.cache   │ → │ TfidfVector │ │
│  │ (18 Q&A)  │    │ _data load  │    │ -izer + fit │ │
│  └──────────┘    └──────────────┘    └──────┬──────┘ │
│                                             │        │
│  ┌──────────────────────────────────────────┘        │
│  │  User query → preprocess → cosine_similarity      │
│  │  → if ≥ 0.2 return answer else fallback           │
│  └───────────────────────────────────────────────────┘
```

## Enterprise Features

| Feature              | Implementation                                       |
|----------------------|-------------------------------------------------------|
| **Type Hints**       | Full annotations on every function signature          |
| **Docstrings**       | Google-style on all public functions                  |
| **Logging**          | `logging` module; unrecognized queries are logged     |
| **JSON Validation**  | `validate_faqs()` checks structure before loading     |
| **Caching**          | `@st.cache_data` on `load_faqs()` and `build_vectoriser()` |
| **Dynamic Threshold**| 0.20 — below that, bot asks for rephrase & logs query  |
| **Fallback Message** | "I'm not entirely sure, could you rephrase?"           |

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run chatbot.py
```

The browser opens at `http://localhost:8501`.

---

## Project Structure

```
CodeAlpha_FAQ_Chatbot/
├── chatbot.py          # Main application
├── faqs.json           # 18 technical Q&A pairs
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## Dependencies

| Package            | Version | Purpose                                |
|--------------------|---------|----------------------------------------|
| `streamlit`        | ≥ 1.28  | Web UI framework                       |
| `nltk`             | ≥ 3.8   | Tokenisation, stemming, stopwords      |
| `scikit-learn`     | ≥ 1.3   | TF‑IDF vectorisation + cosine similarity |

---
