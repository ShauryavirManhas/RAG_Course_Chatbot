import os
import uuid
import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2

# ── Config ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 4

# ── Per-session client init (NOT cached globally) ────────────────────────────
def get_clients():
    # Session ID — unique per browser tab/session
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "collection" not in st.session_state:
        genai.configure(api_key=st.secrets["api_key"])

        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # EphemeralClient = in-memory only, never written to disk
        chroma_client = chromadb.EphemeralClient()
        st.session_state.collection = chroma_client.create_collection(
            name=f"user_{st.session_state.session_id}",
            embedding_function=embed_fn
        )

    if "gemini_client" not in st.session_state:
        genai.configure(api_key=st.secrets["api_key"])
        st.session_state.gemini_client = genai.GenerativeModel("gemini-flash-latest")

    return st.session_state.gemini_client, st.session_state.collection


# ── PDF → chunks ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, source_name: str) -> list[dict]:
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
    text   = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text, uploaded_file.name)
    collection.upsert(
        ids       = [c["id"]   for c in chunks],
        documents = [c["text"] for c in chunks],
        metadatas = [{"source": c["source"]} for c in chunks]
    )
    return len(chunks)

# ── Retrieval + generation ───────────────────────────────────────────────────
def retrieve(query: str, collection) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=TOP_K)
    chunks  = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""You are a helpful study assistant for graduate students.
Answer the question using ONLY the context provided below.
If the answer isn't in the context, say "I couldn't find that in your uploaded materials."
Always mention which source document your answer comes from.

CONTEXT:
{context_block}

QUESTION: {query}"""

def ask_gemini(query, collection, gemini_client):
    chunks   = retrieve(query, collection)
    prompt   = build_prompt(query, chunks)
    response = gemini_client.generate_content(prompt)
    return response.text, chunks

# ── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="RAG Course Chatbot", page_icon="📚", layout="wide")
    st.title("📚 RAG Course PDF Chatbot")
    st.caption("Upload your lecture slides or notes, then ask questions.")

    gemini_client, collection = get_clients()

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

        total = collection.count()
        st.divider()
        st.metric("Chunks in database", total)
        if st.button("Clear database"):
            # Delete the collection and reset session state
            st.session_state.pop("collection", None)
            st.session_state.pop("session_id", None)
            st.rerun()

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
