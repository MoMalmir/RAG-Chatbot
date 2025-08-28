import time
from pathlib import Path
from typing import List

import streamlit as st

from ragchatbot.ingest import ingest_paths
from ragchatbot.query import RAGSimple
from ragchatbot.utils import device_info


# --------------------------- page setup -------------------------------

st.set_page_config(page_title="RAG Chatbot — Streamlit", layout="wide")
st.title("RAG Chatbot — Streamlit")

tab_ingest, tab_query = st.tabs(["📥 Ingest Documents", "❓ Ask a Question"])


# ---------------------------- side panel ------------------------------

with st.sidebar:
    dev, is_cuda = device_info()
    st.caption(f"**Device:** {dev} | **CUDA:** {bool(is_cuda)}")

    index_dir = st.text_input("Index directory", "index")
    embed_model = st.text_input("Embedding model", "sentence-transformers/all-mpnet-base-v2")

    st.markdown("---")
    st.caption("Retrieval")
    initial_k = st.slider("initial_k (retriever fan-out)", 4, 32, 12, step=1)
    final_k = st.slider("final_k (contexts kept)", 1, 12, 6, step=1)
    use_reranker = st.checkbox("Use cross-encoder reranker (BAAI/bge-reranker-base)", value=True)
    max_bullets = st.slider("Max bullets (evidence sentences)", 2, 10, 6, step=1)

    st.markdown("---")
    st.caption("Summarization (OpenRouter)")
    mode = st.radio("Mode", ["Evidence-only (no LLM)", "Summarize with OpenRouter"], index=0)
    or_key = st.text_input("OpenRouter API key", type="password", placeholder="sk-or-v1-...")
    or_model = st.text_input("OpenRouter model", value="meta-llama/llama-3.1-70b-instruct")


# ---------------------------- ingest tab ------------------------------

with tab_ingest:
    st.subheader("Ingest Documents")
    st.caption("Upload files/folders and build/update the FAISS index.")

    inputs = st.text_area(
        "Paths (one per line)", 
        value="docs", 
        help="You can list files and/or directories. Relative paths resolved from the project root."
    )
    recursive = st.checkbox("Recurse folders", True)
    include = st.text_input("Include globs (comma-separated)", "*.pdf,*.docx,*.md,*.txt,*.html,*.htm")
    exclude = st.text_input("Exclude globs (comma-separated)", "")
    chunk_size = st.number_input("chunk_size", 200, 4000, 1000, step=50)
    chunk_overlap = st.number_input("chunk_overlap", 0, 1000, 200, step=10)

    if st.button("Build / Update Index"):
        try:
            paths: List[str] = [p.strip() for p in inputs.splitlines() if p.strip()]
            out = ingest_paths(
                inputs=paths,
                index_dir=index_dir,
                recursive=recursive,
                include=include or None,
                exclude=exclude or None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embed_model=embed_model,
            )
            st.success(f"Indexed to: {out}")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")


# ----------------------------- qa tab ---------------------------------

with tab_query:
    st.subheader("Ask the RAG Chatbot")

    with st.form("qa_form"):
        q = st.text_input("Your question", value="What is the monthly rent?")
        show_ctx = st.checkbox("Show retrieved contexts", value=False)
        submitted = st.form_submit_button("Run", type="primary")

    if submitted and q.strip():
        t0 = time.time()
        try:
            rag = RAGSimple(
                index_dir=index_dir,
                embed_model=embed_model,
                initial_k=initial_k,
                final_k=final_k,
                use_reranker=use_reranker,
            )
        except Exception as e:
            st.error(f"Failed to load index/models: {e}")
            st.stop()

        try:
            if mode.startswith("Evidence-only"):
                result = rag.ask_evidence(q, max_bullets=max_bullets)
                mode_label = "evidence"
            else:
                result = rag.ask_summarized(q, api_key=or_key, model=or_model, max_bullets=max_bullets)
                mode_label = "summarize (OpenRouter)" if or_key else "evidence (no key)"
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()

        dt = time.time() - t0

        st.markdown("### Answer")
        st.write(result.get("answer", "").strip())
        st.caption(f"Latency: {dt:.2f}s | Mode: {mode_label} | final_k={final_k} | reranker={use_reranker}")

        if show_ctx:
            st.markdown("### Retrieved Contexts")
            # Re-run retrieval just to display the docs in the same order used for citations
            docs = rag._retrieve_top(q)
            for i, d in enumerate(docs, start=1):
                meta = d.metadata or {}
                label = Path(meta.get('source','')).name if meta.get('source') else 'unknown'
                page = meta.get('page')
                if page is not None:
                    label = f"{label} (p.{page+1})"
                with st.expander(f"[{i}] {label}"):
                    st.write((d.page_content or "").strip())

        st.markdown("### Citations")
        for i, c in enumerate(result.get("citations", []), start=1):
            st.write(f"[{i}] **{c['label']}**")
            st.caption(f"Source: `{c['source']}` • filetype: {c.get('filetype','n/a')}")
