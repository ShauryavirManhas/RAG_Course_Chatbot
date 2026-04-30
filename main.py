import os
import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2

# ── Config ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "cmu_docs"
CHUNK_SIZE       = 500   # characters per chunk
CHUNK_OVERLAP    = 50    # overlap between chunks so context isn't lost at edges
TOP_K            = 4     # how many chunks to retrieve per query
MODEL            = "claude-sonnet-4-6"

# ── Clients (cached so they don't reload on every Streamlit rerun) ───────────
# Replace get_clients() with:
@st.cache_resource
def get_clients():
    genai.configure(api_key="<Enter Key Here>")
    gemini_client = genai.GenerativeModel("gemini-flash-latest")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )
    return gemini_client, collection


# ── PDF → chunks ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, source_name: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "id":     f"{source_name}_{chunk_id}",
                "text":   chunk,
                "source": source_name
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_id += 1
    return chunks

def index_pdf(uploaded_file, collection) -> int:
    """Extract, chunk, and store a PDF in ChromaDB. Returns chunk count."""
    text   = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text, uploaded_file.name)

    # ChromaDB upsert: if same ID exists it updates, so re-uploading is safe
    collection.upsert(
        ids       = [c["id"]   for c in chunks],
        documents = [c["text"] for c in chunks],
        metadatas = [{"source": c["source"]} for c in chunks]
    )
    return len(chunks)

# ── Retrieval + generation ───────────────────────────────────────────────────
def retrieve(query: str, collection) -> list[dict]:
    """Find the TOP_K most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=TOP_K)
    print(results)
    chunks  = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""You are a helpful study assistant for CMU graduate students.
Answer the question using ONLY the context provided below.
If the answer isn't in the context, say "I couldn't find that in your uploaded materials."
Always mention which source document your answer comes from.

CONTEXT:
{context_block}

QUESTION: {query}"""

def ask_gemini(query, collection, gemini_client):
    chunks  = retrieve(query, collection)
    prompt  = build_prompt(query, chunks)
    response = gemini_client.generate_content(prompt)
    return response.text, chunks

# ── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="CMU Course Chatbot", page_icon="📚", layout="wide")
    st.title("📚 CMU Course PDF Chatbot")
    st.caption("Upload your lecture slides or notes, then ask questions.")

    gemini_client, collection = get_clients()

    # Sidebar: upload PDFs
    with st.sidebar:
        st.header("Upload Course Materials")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Index uploaded PDFs", type="primary"):
                for f in uploaded_files:
                    with st.spinner(f"Indexing {f.name}..."):
                        n = index_pdf(f, collection)
                    st.success(f"✓ {f.name} — {n} chunks indexed")

        # Show what's already indexed
        total = collection.count()
        st.divider()
        st.metric("Chunks in database", total)
        if st.button("Clear database"):
            collection.delete(where={"source": {"$ne": ""}})
            st.rerun()

    # Main: chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources used"):
                    for s in msg["sources"]:
                        st.caption(f"**{s['source']}**: ...{s['text'][:200]}...")

    if query := st.chat_input("Ask anything about your course materials..."):
        if collection.count() == 0:
            st.warning("Please upload and index some PDFs first.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching your notes..."):
                    answer, sources = ask_gemini(query, collection, gemini_client)
                st.write(answer)
                with st.expander("📎 Sources used"):
                    for s in sources:
                        st.caption(f"**{s['source']}**: ...{s['text'][:200]}...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

if __name__ == "__main__":
    main()