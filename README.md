# Gaurav's AI Assistant — PDF RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-005571)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-F55036)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

A personal **portfolio chatbot** that answers questions about Gaurav's professional profile — background, skills, education, and projects — using Retrieval-Augmented Generation (RAG) over his resume/profile PDF(s). Built with **Hybrid Search (Vector + Keyword) and Cross-Encoder Re-ranking** for high-quality retrieval, and **Groq (Llama 3.3 70B)** for natural-language answer generation.

---

## Table of Contents

- [Features](#features)
- [Use Case](#use-case)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How Retrieval Works](#how-retrieval-works)
- [Project Structure](#project-structure)
- [Notes](#notes)
- [License](#license)

---

## Features

- **PDF Ingestion** — loads and parses profile/resume PDFs, preserving source metadata
- **Smart Chunking** — recursive text splitting with configurable overlap
- **Hybrid Retrieval** — combines semantic (vector) search with keyword (BM25) search
- **Cross-Encoder Re-ranking** — re-scores candidates for higher-precision results
- **Fast LLM Inference** — powered by Groq's Llama 3.3 70B
- **Streamlit Chat UI** — clean, dark-themed portfolio chat interface with persistent session
- **Index Persistence** — build once, reuse via saved `.pkl` files

---

## Use Case

This bot lets recruiters, collaborators, or visitors ask natural-language questions about Gaurav's profile instead of reading a full resume/PDF. Example questions it can answer:

- What are his technical skills?
- What's his CGPA?
- Is he job ready?
- What projects has he built?

---

## Architecture

```
PDF Files
   │
   ▼
Load & Parse (PyMuPDFLoader)
   │
   ▼
Chunking (RecursiveCharacterTextSplitter)
   │
   ├──────────────────────────┐
   ▼                          ▼
Embeddings                BM25 Index
(SentenceTransformer)     (rank_bm25)
   │                          │
   ▼                          │
FAISS Vector Index            │
(IndexFlatL2)                 │
   │                          │
   └──────────┬───────────────┘
              ▼
      Query Time: Hybrid Search
   (FAISS top-k + BM25 top-k, deduped)
              │
              ▼
      Cross-Encoder Re-ranking
   (ms-marco-MiniLM-L-6-v2)
              │
              ▼
        Top-k Final Chunks
              │
              ▼
      Groq LLM (Llama 3.3 70B)
              │
              ▼
           Answer
```

---

## Tech Stack

| Component | Tool / Library |
|---|---|
| PDF Loading | `langchain_community` → `PyMuPDFLoader` |
| Text Splitting | `langchain_text_splitters` → `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` → `all-MiniLM-L6-v2` |
| Vector Store | `faiss-cpu` → `IndexFlatL2` |
| Keyword Search | `rank_bm25` → `BM25Okapi` |
| Re-ranking | `sentence-transformers` → `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) |
| LLM | `groq` → `llama-3.3-70b-versatile` |
| Persistence | `joblib` |
| UI | `streamlit` |

---

## Installation

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
langchain-community
pymupdf
langchain-text-splitters
sentence-transformers
faiss-cpu
numpy
joblib
groq
streamlit
rank_bm25
```

---

## Configuration

Set your Groq API key via Streamlit secrets — create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key_here"
```

> ⚠️ Never hardcode API keys directly in source files. Rotate any key that has been shared or committed publicly.

---

## Usage

### 1. Build the index (Notebook)

Run the ingestion pipeline once to process your PDFs and build the indexes:

```python
all_documents = load_data()                          # load PDFs
split_docs = split_documents(all_documents)          # chunk text
model = Convert_text()                               # load embedding model
embeddings = embeddings_generation(model, split_docs) # generate embeddings
index = Vector_Database(embeddings)                   # build FAISS index
bm25 = build_bm25_index(split_docs)                   # build BM25 index
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

joblib.dump(split_docs, 'split_docs.pkl')
joblib.dump(index, 'index.pkl')
joblib.dump(bm25, 'bm25.pkl')
```

This produces three files: `split_docs.pkl`, `index.pkl`, `bm25.pkl`.

### 2. Query directly (Notebook)

```python
query = "your question here"
results = retrieval(query, model, index, split_docs, bm25, cross_encoder, k=3)
answer = generate_answer(query, results, groq_client)
print(answer)
```

### 3. Launch the chat UI (Streamlit)

Place `split_docs.pkl`, `index.pkl`, and `bm25.pkl` in the same folder as the app, then:

```bash
streamlit run app.py
```

---

## How Retrieval Works

1. **Vector Search** — the query is embedded and matched against `split_docs` via FAISS (`fetch_k` candidates)
2. **Keyword Search (BM25)** — the query is tokenized and scored against the same chunks for exact-term matches (`fetch_k` candidates)
3. **Combine + Deduplicate** — both candidate sets are merged, removing duplicate chunk indices
4. **Cross-Encoder Re-ranking** — each `[query, chunk]` pair is scored jointly by the cross-encoder for higher-precision relevance
5. **Top-k Selection** — the highest-scoring `k` chunks are passed as context to the LLM

This hybrid approach catches both **semantic matches** (meaning-based) and **exact matches** (names, numbers, specific terms) that pure vector search can miss.

---

## Project Structure

```
.
├── pdf_loader.ipynb      # Ingestion + indexing pipeline (notebook)
├── app.py                # Streamlit chat UI
├── requirements.txt      # Dependencies
├── split_docs.pkl        # Saved chunks (generated)
├── index.pkl             # Saved FAISS index (generated)
└── bm25.pkl              # Saved BM25 index (generated)
```

> The `pdf_directory` fed into `load_data()` should contain Gaurav's resume/profile PDF(s).

---

## Notes

- Rebuild the indexes (`split_docs.pkl`, `index.pkl`, `bm25.pkl`) whenever the source PDFs change.
- The embedding model (`all-MiniLM-L6-v2`) and cross-encoder (`ms-marco-MiniLM-L-6-v2`) download automatically on first run and are cached locally.
- Chunk size, overlap, `k`, and `fetch_k` are tunable parameters — adjust based on document length and desired answer granularity.

---

## License

This project is licensed under the MIT License.
