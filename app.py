# app.py
import os
import time
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from ragchatbot.ingest import ingest_paths
from ragchatbot.query import RAGQuery
from ragchatbot.utils import device_info

st.set_page_config(page_title="RAG Chatbot — Streamlit", layout="wide")

# ---------- Sidebar ----------
st.sidebar.header("Settings")
dev, is_cuda = device_info()
st.sidebar.info(f"**Device:** {dev} | **CUDA:** {is_cuda}")

index_dir   = st.sidebar.text_input("Index directory (fallback / shared)", value="index")
embed_model = st.sidebar.text_input("Embedding model", value="sentence-transformers/all-mpnet-base-v2")

st.sidebar.subheader("Retrieval")
initial_k    = st.sidebar.slider("initial_k (retriever fan-out)", 4, 24, 12, 1)
final_k      = st.sidebar.slider("final_k (contexts kept)", 1, 10, 6, 1)
use_reranker = st.sidebar.checkbox("Use cross-encoder reranker (BAAI/bge-reranker-base)", value=True)
max_bullets  = st.sidebar.slider("Max bullets (evidence sentences)", 3, 10, 6, 1)

st.sidebar.subheader("Answer mode")
mode = st.sidebar.radio("Choose", ["Evidence (no LLM)", "Summarize with OpenRouter"], index=0)

# Only show key/model when Summarize is selected
or_key_env = os.getenv("OPENROUTER_API_KEY", "")
or_key, or_model = "", "meta-llama/llama-3.1-70b-instruct"
if mode == "Summarize with OpenRouter":
    st.sidebar.subheader("LLM (OpenRouter)")
    or_key   = st.sidebar.text_input("OpenRouter API key", value=or_key_env, type="password")
    or_model = st.sidebar.text_input("OpenRouter model", value=or_model)

# ---------- Main ----------
st.title("RAG Chatbot — Streamlit")
ingest_tab, ask_tab = st.tabs(["📥 Ingest Documents", "❓ Ask a Question"])

# ===== Ingest tab =====
with ingest_tab:
    st.markdown("Upload files. These will be processed **ephemerally** for your session and then deleted.")
    uploaded = st.file_uploader(
        "Drop files (PDF, DOCX, MD, TXT, HTML)",
        type=["pdf", "docx", "md", "txt", "html", "htm"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        chunk_size = st.number_input("chunk_size", min_value=200, max_value=4000, step=100, value=1000)
    with col_b:
        chunk_overlap = st.number_input("chunk_overlap", min_value=0, max_value=1000, step=50, value=200)

    if st.button("Build in-memory index for my session"):
        if not uploaded:
            st.warning("Please upload at least one file.")
        else:
            try:
                # Write uploads to a temp folder so loaders can read paths
                tmpdir = tempfile.mkdtemp(prefix="uploads_")
                paths = []
                for up in uploaded:
                    p = Path(tmpdir) / up.name
                    p.write_bytes(up.read())
                    paths.append(str(p))

                # Build FAISS in memory, then delete temp files right away
                with st.spinner("Indexing…"):
                    vs = ingest_paths(
                        inputs=paths,
                        index_dir=index_dir,
                        recursive=True,
                        include="*.pdf,*.docx,*.md,*.txt,*.html,*.htm",
                        exclude=None,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        embed_model=embed_model,
                        save_index=False,          # <— return FAISS object
                    )
                shutil.rmtree(tmpdir, ignore_errors=True)

                # Store the vectorstore in this user's session only
                st.session_state["vs"] = vs
                st.success("In-memory index built for your session. (Uploads have been deleted.)")

            except Exception as e:
                st.error(f"Ingestion failed: {e}")

# ===== Ask tab =====
# ===== Ask tab =====
with ask_tab:
    question = st.text_input("Your question", value="", placeholder="Type a question and press Enter")
    run = st.button("Run", type="primary", disabled=st.session_state.get("busy", False))

    if run and question.strip():
        st.session_state["busy"] = True
        t0 = time.time()

        # Big, visible status box with spinner
        with st.status("🔄 Working on your answer…", state="running", expanded=True) as status:
            try:
                status.update(label="🔎 Retrieving evidence…", state="running")

                # Prefer the session's in-memory vectorstore if present
                vs = st.session_state.get("vs", None)
                rag = RAGQuery(
                    index_dir=index_dir,
                    embed_model=embed_model,
                    initial_k=initial_k,
                    final_k=final_k,
                    use_reranker=use_reranker,
                    reranker_model="BAAI/bge-reranker-base",
                    vectorstore=vs,
                )

                if mode == "Evidence (no LLM)":
                    res = rag.ask_evidence(question, max_bullets=max_bullets)
                else:
                    api_key = or_key or or_key_env or ""
                    if not api_key:
                        st.warning("No OpenRouter API key provided; please enter one in the sidebar or set OPENROUTER_API_KEY.")
                    status.update(label="🧠 Summarizing with OpenRouter…", state="running")
                    res = rag.ask_summarized(question, api_key=api_key, model=or_model, max_bullets=max_bullets)

                dt = time.time() - t0
                status.update(label="✅ Done", state="complete", expanded=False)

                st.markdown("## Answer")
                st.write(res["answer"])
                st.caption(
                    f"Latency: {dt:.2f}s | Mode: {'evidence' if mode=='Evidence (no LLM)' else 'summarize'} | "
                    f"final_k={final_k} | reranker={'True' if use_reranker else 'False'} | "
                    f"index: {'in-memory (session)' if vs is not None else 'disk (' + index_dir + ')'}"
                )

                st.markdown("## Citations")
                for i, c in enumerate(res["citations"], start=1):
                    label = c.get("label", "unknown")
                    src = c.get("source")
                    pg  = c.get("page")
                    ft  = c.get("filetype")
                    meta = []
                    if src: meta.append(f"source: {src}")
                    if ft:  meta.append(f"filetype: {ft}")
                    if pg is not None: meta.append(f"page: {pg+1}")
                    st.markdown(f"[{i}] **{label}**  \n<sub>{' • '.join(meta)}</sub>", unsafe_allow_html=True)

            except Exception as e:
                status.update(label="❌ Failed", state="error", expanded=True)
                st.error(f"Query failed: {e}")

            finally:
                st.session_state["busy"] = False
