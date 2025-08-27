from pathlib import Path
from typing import Any, Dict, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from .utils import DEVICE


def _fmt_citation(meta: Dict[str, Any]) -> Dict[str, Any]:
    src = meta.get("source")
    page = meta.get("page", None)  # some loaders set int, some None
    ftype = meta.get("filetype")
    label = Path(src).name if src else "unknown"
    if page is not None:
        label = f"{label} (p.{page})"
    return {
        "label": label,
        "source": src,
        "page": page,
        "filetype": ftype,
    }


class RAGQuery:
    def __init__(
        self,
        index_dir: str = "index",
        embed_model: str = "sentence-transformers/all-mpnet-base-v2",
        qa_model_name: str = "bert-large-uncased-whole-word-masking-finetuned-squad",
        k: int = 4,
    ) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vs = FAISS.load_local(
            index_dir, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vs.as_retriever(search_kwargs={"k": k})

        tok = AutoTokenizer.from_pretrained(qa_model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            qa_model_name,
            torch_dtype=(__import__("torch").float16 if DEVICE == "cuda" else __import__("torch").float32),
        ).to(DEVICE)
        self.qa = pipeline(
            "question-answering", model=qa_model, tokenizer=tok, device=0 if DEVICE == "cuda" else -1
        )

    def ask(self, question: str, max_ctx_chars: int = 8000) -> Dict[str, Any]:
        docs = self.retriever.get_relevant_documents(question)
        context_parts: List[str] = []
        citations: List[Dict[str, Any]] = []

        for d in docs:
            context_parts.append(d.page_content)
            citations.append(_fmt_citation(d.metadata or {}))

        context = "\n\n".join(context_parts)[:max_ctx_chars]
        out = self.qa(question=question, context=context)

        return {
            "answer": out.get("answer"),
            "score": out.get("score"),
            "citations": citations,
        }
