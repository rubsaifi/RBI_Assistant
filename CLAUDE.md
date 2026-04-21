# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RBI Master Policy Assistant** - An AI-powered Streamlit chatbot that answers queries about Reserve Bank of India (RBI) Master Circular policies. Uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses from the RBI Master PDF document.

## Architecture

### High-Level Flow
1. **PDF Processing** (`utils/pdf_loader.py`): Extracts and chunks text from `Docs/rbi_master.pdf`
2. **RAG Engine** (`utils/rag_engine.py`): Creates embeddings using sentence-transformers, builds FAISS index for similarity search
3. **LLM Handler** (`utils/llm_handler.py`): Supports multiple providers (Groq, Gemini, HuggingFace, Ollama) for generating responses
4. **Streamlit App** (`app.py`): Web interface with custom RBI-themed CSS styling

### Key Components

- **RAG Pipeline**: `sentence-transformers` (all-MiniLM-L6-v2) → FAISS index → Top-k retrieval
- **Embeddings**: 384-dimensional vectors, cosine similarity search
- **Chunking**: 1000 char chunks with 200 char overlap, sentence-boundary aware
- **LLM Integration**: Provider-agnostic with fallback to keyword search

## Common Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Run with specific provider
LLM_PROVIDER=gemini GOOGLE_API_KEY=xxx streamlit run app.py

# Deployment
# Push to GitHub, then deploy via https://share.streamlit.io/
# Add secrets in Streamlit Cloud: GROQ_API_KEY, GOOGLE_API_KEY, etc.
```

## File Structure

```
Prod_RBI_Assistant/
├── app.py                    # Main Streamlit app with custom CSS
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml   # Streamlit theme configuration
├── utils/
│   ├── pdf_loader.py        # PDF text extraction and chunking
│   ├── rag_engine.py        # Embeddings + FAISS similarity search
│   └── llm_handler.py       # Multi-provider LLM API wrapper
└── Docs/
    └── rbi_master.pdf       # RBI Master Circular document
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API (fastest, 1M tokens/day free) | Optional* |
| `GOOGLE_API_KEY` | Gemini API (60 req/min free) | Optional* |
| `HUGGINGFACE_API_KEY` | HuggingFace token | Optional* |
| `LLM_PROVIDER` | Provider: groq, gemini, huggingface, ollama | Optional |

*At least one API key required for cloud deployment. Local Ollama requires no key.

## Code Patterns

### Adding a New LLM Provider
1. Create class in `utils/llm_handler.py` inheriting from `LLMProvider`
2. Add to `PROVIDERS` dictionary
3. Document in README.md

### Modifying Chat Theme
- Edit CSS in `app.py` → `load_custom_css()` function
- User messages: `.user-message` class (purple gradient)
- Bot messages: `.bot-message` class (pink/orange gradient)
- Streamlit theme: `.streamlit/config.toml`

### Document Processing
- Chunk size: 1000 chars, overlap: 200 chars
- Minimum relevance score: 0.3
- Fallback to top result if none above threshold

## Deployment Notes

- **Platform**: Streamlit Cloud (share.streamlit.io)
- **PDF Storage**: Include in repo under `Docs/` (not ignored)
- **Model Downloads**: sentence-transformers model downloads on first run (~80MB)
- **Cold Start**: Embedding creation happens on app load (cached in session state)

## Troubleshooting

- **PDF not found**: Check `Docs/rbi_master.pdf` exists
- **API errors**: Verify API keys in environment or Streamlit secrets
- **Slow responses**: First run downloads embedding model; consider smaller model
- **Memory issues**: Reduce `chunk_size` in `pdf_loader.py` if needed
