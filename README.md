# RAG Project — Document & Manual Search

Two lightweight RAG (Retrieval-Augmented Generation) applications built to understand
the end-to-end AI pipeline: chunking, embedding, vector storage, retrieval, and grounded generation.

---

## Project

### 1. Course PDF Chatbot
Upload lecture slides or course notes and ask questions about them.
The bot retrieves the most relevant sections and answers using only your uploaded material — no hallucination from general knowledge.

**Use case:** "What did the lecture say about backpropagation?" → retrieves the relevant slide chunk → Claude answers from it.
---

## How it works (RAG pipeline)

```
PDF / text file
      ↓
  Extract text (PyPDF2)
      ↓
  Chunk into overlapping pieces (~400–500 chars)
      ↓
  Embed each chunk (sentence-transformers, runs locally)
      ↓
  Store vectors in ChromaDB (local, no account needed)
      ↓
  User asks a question
      ↓
  Embed the question → similarity search → retrieve top-K chunks
      ↓
  Inject chunks + question into LLM prompt
      ↓
  LLM generates a grounded answer citing the source
```

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| UI | Streamlit | Fast to prototype, easy to iterate |
| PDF parsing | PyPDF2 | Lightweight, no external API |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Runs fully locally, free |
| Vector DB | ChromaDB | Local persistent store, no account needed |
| LLM | Claude (Anthropic API) / Gemini | Swappable — RAG is model-agnostic |
| Language | Python 3.10+ | — |

---

## Project structure

```
RAG_Projects/
├── requirements.txt
├── Course_Chatbot/
│   └── main.py
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-projects
cd rag-projects

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

---

## Run

```bash
# Course Chatbot
streamlit run main.py
```

Then open `http://localhost:8501` in your browser.

---

## Usage

**Course Chatbot**
1. Upload one or more PDF lecture files in the sidebar
2. Click "Index uploaded PDFs"
3. Ask questions in the chat input

---

## Key design decisions

**Why overlapping chunks?**
A sentence at the boundary of two chunks shouldn't disappear from both.
Overlap of 50–80 chars ensures boundary context is preserved.

**Why local embeddings instead of an embeddings API?**
`sentence-transformers` runs on CPU with no API cost or latency.
For a demo project this is faster and free. In production you'd evaluate
hosted embeddings (OpenAI, Cohere) for higher quality.

**Why ChromaDB?**
Zero setup — it writes to a local folder. Swappable for Pinecone or
pgvector in production with minimal code changes since the interface is the same.

**Why is the LLM swappable?**
The RAG pipeline (retrieval + prompt construction) is completely independent
of which LLM generates the answer. Switching from Claude to Gemini required
changing ~10 lines. This is intentional — good RAG architecture doesn't
couple retrieval logic to a specific model provider.

---

## What I'd add in a production system

- **Hybrid retrieval** — combine vector similarity with keyword search (BM25) for better precision on exact lookups like part numbers
- **Reranking** — use a cross-encoder model to re-score retrieved chunks before sending to the LLM
- **Semantic chunking** — split at paragraph/section boundaries instead of fixed character count
- **Eval pipeline** — golden Q&A test set with automated scoring on every change
- **Access control** — users should only retrieve documents they're authorized to see
- **Incremental indexing** — re-index only changed documents rather than full re-ingestion

---

## Author

Shauryavir Singh Manhas
MS Information Systems Management, Carnegie Mellon University (Heinz College)
[smanhas@andrew.cmu.edu](mailto:smanhas@andrew.cmu.edu) · [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN)
