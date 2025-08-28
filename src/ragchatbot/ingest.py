# src/ragchatbot/ingest.py
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, BSHTMLLoader, TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .utils import DEVICE, iter_input_files, stable_id

# ---- Loader registry (unchanged) ----
def _load_with_loader(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    if ext in {".html", ".htm"}:
        return BSHTMLLoader(str(path)).load()
    if ext in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    return TextLoader(str(path), encoding="utf-8").load()

def _normalize_metadata(docs, src_path: Path):
    doc_id = stable_id(src_path)
    ft = src_path.suffix.lower().lstrip(".")
    for d in docs:
        meta = d.metadata or {}
        meta.setdefault("source", str(src_path))
        meta.setdefault("filetype", ft)
        meta.setdefault("doc_id", doc_id)
        d.metadata = meta
    return docs

# ---- Public API ----
def ingest_paths(
    inputs: List[str],
    index_dir: str = "index",
    recursive: bool = True,
    include: str | None = "*.pdf,*.docx,*.md,*.txt,*.html,*.htm",
    exclude: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    save_index: bool = True,           # <-- NEW: if False, return FAISS object
):
    """
    Load files/folders, split, embed, and either:
      - save a FAISS index to `index_dir` (default), returning the dir path
      - OR return an in-memory FAISS vectorstore when `save_index=False`

    Supports PDF, DOCX, MD, TXT, HTML.
    """
    files = iter_input_files(inputs, recursive=recursive, include=include, exclude=exclude)
    if not files:
        raise FileNotFoundError("No files matched the given paths/patterns.")

    all_docs = []
    for f in files:
        raw_docs = _load_with_loader(Path(f))
        all_docs.extend(_normalize_metadata(raw_docs, Path(f)))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=True
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.from_documents(chunks, embeddings)

    if save_index:
        outdir = Path(index_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(outdir))
        return str(outdir)
    else:
        return vs
