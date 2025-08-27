from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .utils import DEVICE, iter_input_files, stable_id


# ---- Loader registry ---------------------------------------------------------
def _load_with_loader(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()  # preserves page metadata
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    if ext in {".html", ".htm"}:
        return BSHTMLLoader(str(path)).load()
    if ext in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    # Fallback to text (best-effort)
    return TextLoader(str(path), encoding="utf-8").load()


def _normalize_metadata(docs, src_path: Path):
    """Ensure we always have {source, filetype, doc_id, page?} for citations."""
    doc_id = stable_id(src_path)
    ft = src_path.suffix.lower().lstrip(".")
    for d in docs:
        meta = d.metadata or {}
        meta.setdefault("source", str(src_path))
        meta.setdefault("filetype", ft)
        meta.setdefault("doc_id", doc_id)
        # some loaders don’t set page; that’s fine—query will handle it
        d.metadata = meta
    return docs


# ---- Public API --------------------------------------------------------------
def ingest_paths(
    inputs: List[str],
    index_dir: str = "index",
    recursive: bool = True,
    include: str | None = "*.pdf,*.docx,*.md,*.txt,*.html,*.htm",
    exclude: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
) -> str:
    """
    Load files/folders, split, embed, and persist a FAISS index.
    Supports PDF, DOCX, MD, TXT, HTML out of the box.
    """
    files = iter_input_files(inputs, recursive=recursive, include=include, exclude=exclude)
    if not files:
        raise FileNotFoundError("No files matched the given paths/patterns.")

    all_docs = []
    for f in files:
        raw_docs = _load_with_loader(f)
        all_docs.extend(_normalize_metadata(raw_docs, f))

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

    outdir = Path(index_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(outdir))
    return str(outdir)
