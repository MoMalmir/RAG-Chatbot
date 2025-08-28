# app.py
import os
import sys
from pathlib import Path
import time
import streamlit as st

# Make sure local src/ is importable without pip-install
sys.path.append("src")

from ragchatbot.query import RAGQuery
from ragchatbot.ingest import ingest_paths
from ragchatbot.utils import device_info

APP_TITLE = "RAG Chatbot — Streamlit"
DATA_DIR = Path("docs")
INDEX_DIR = Path("index")

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# --- Sidebar: settings -------------------------------------------------
st.sidebar.header("Settings")

# Device info
dev, is_cuda = device_info()
st.sidebar.info(f"**Device:** {dev} | **CUDA:** {is_cuda}")

# Index + models
index_dir = st.sidebar.text_input("Index directory", value=str(INDEX_DIR))
embed_model = st.sidebar.text_input(
    "Embedding model",
    value="sentence-transformers/all-mpnet-base-v2",
    help="Small & solid alternative: sentence-transformers/all-MiniLM-L6-v2",
)

mode = st.sidebar.radio("Mode", ["generative", "extractive"], index=0)

qa_model = st.sidebar.text_input(
    "Extractive QA model", value="bert-large-uncased-whole-word-masking-finetuned-squad"
)
gen_model = st.sidebar.text_input(
    "Generative model", value="google/flan-t5-base"
)

# Retrieval knobs
initial_k = st.sidebar.slider("initial_k (retriever fan-out)", 4, 40, 12, 1)
final_k = st.sidebar.slider("final_k (contexts kept)", 1, 12, 6, 1)
use_reranker = st.sidebar.checkbox("Use cross-encoder reranker (BAAI/bge-reranker-base)", value=True)

# Ingestion knobs
st.sidebar.markdown("---")
st.sidebar.subheader("Ingestion")
chunk_size = st.sidebar.slider("chunk_size", 300, 2000, 1000, 50)
chunk_overlap = st.sidebar.slider("chunk_overlap", 0, 500, 200, 10)
include_globs = st.sidebar.text_input(
    "Include globs (comma-separated)", "*.pdf,*.docx,*.md,*.txt,*.html,*.htm"
)
exclude_globs = st.sidebar.text_input("Exclude globs (optional)", "")

# --- Helpers -----------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_query_instance(_index_dir: str,
                        _embed_model: str,
                        _qa_model: str,
                        _gen_model: str,
                        _mode: str,
                        _initial_k: int,
                        _final_k: int,
                        _use_reranker: bool) -> RAGQuery:
    return RAGQuery(
        index_dir=_index_dir,
        embed_model=_embed_model,
        qa_model_name=_qa_model,
        gen_model_name=_gen_model,
        mode=_mode,
        initial_k=_initial_k,
        final_k=_final_k,
        use_reranker=_use_reranker,
    )

def _invalidate_cached_query():
    _get_query_instance.clear()

def _save_uploads(files):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = DATA_DIR / f.name
        with open(dest, "wb") as out:
            out.write(f.read())
        saved.append(str(dest))
    return saved

# --- Tabs --------------------------------------------------------------
tab_ingest, tab_query = st.tabs(["📥 Ingest Documents", "❓ Ask a Question"])

with tab_ingest:
    st.subheader("Add or update your knowledge base")
    uploads = st.file_uploader(
        "Upload files (PDF, DOCX, MD, TXT, HTML)", type=["pdf", "docx", "md", "txt", "html", "htm"],
        accept_multiple_files=True
    )
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Ingest", type="primary", use_container_width=True, disabled=(uploads is None or len(uploads)==0)):
            paths = _save_uploads(uploads)
            st.write(f"Saved {len(paths)} file(s) to `{DATA_DIR}/`.")

            with st.spinner("Building FAISS index..."):
                out = ingest_paths(
                    inputs=paths if paths else [str(DATA_DIR)],
                    index_dir=index_dir,
                    recursive=True,
                    include=include_globs,
                    exclude=(exclude_globs or None),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embed_model=embed_model,
                )
            st.success(f"Indexed to: `{out}`")
            _invalidate_cached_query()
    with colB:
        if st.button("Rebuild Index from docs/ (no new uploads)", use_container_width=True):
            with st.spinner("Rebuilding index from existing docs/..."):
                out = ingest_paths(
                    inputs=[str(DATA_DIR)],
                    index_dir=index_dir,
                    recursive=True,
                    include=include_globs,
                    exclude=(exclude_globs or None),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embed_model=embed_model,
                )
            st.success(f"Indexed to: `{out}`")
            _invalidate_cached_query()

    st.caption("Tip: You can push new files to the Space and press **Rebuild** anytime.")

with tab_query:
    st.subheader("Ask the RAG Chatbot")
    q = st.text_input("Your question", value="Why do I need letters of recommendation?")

    # NEW: strict toggle and show-contexts
    strict = st.checkbox("Strict mode (extract best sentences, no generation)", value=True)
    show_ctx = st.checkbox("Show retrieved contexts", value=False)

    run = st.button("Run", type="primary")

    if run and q.strip():
        try:
            rag = _get_query_instance(
                index_dir, embed_model, qa_model, gen_model, ("extractive" if strict else mode), initial_k, final_k, use_reranker
            )
        except Exception as e:
            st.error(f"Failed to load index/models: {e}")
            st.stop()

        with st.spinner("Thinking..."):
            import time
            t0 = time.time()
            try:
                result = rag.ask_strict(q) if strict else rag.ask(q)
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()
            dt = time.time() - t0

        st.markdown("### Answer")
        st.write(result.get("answer", "").strip())
        st.caption(f"Latency: {dt:.2f}s | Mode: {'strict' if strict else rag.mode} | final_k={final_k} | reranker={use_reranker}")

        # optional: show the actual contexts used
        if show_ctx:
            st.markdown("### Retrieved Contexts")
            # pull fresh docs using the retriever
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

