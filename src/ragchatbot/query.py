"""
Two-mode RAG (domain-agnostic):

1) Evidence-only: return top-matching sentences with [n] citations
   - Retrieval = BM25 (lexical) + FAISS (vector) + MMR + optional cross-encoder rerank
   - No LLM involved

2) Summarize with OpenRouter: summarize ONLY those evidence bullets into 2–4
   sentences, preserving [n] citations. If API key is missing or the call fails,
   we gracefully fall back to evidence-only.

Keep this file compact and easy to explain in interviews.
"""

from curses import raw
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib
import re
import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from .utils import DEVICE
from .reranker import Reranker


# ----------------------------- helpers ---------------------------------

def _fmt_citation(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Format a display-friendly citation entry from doc metadata."""
    src = meta.get("source")
    page = meta.get("page", None)  # 0-based (PyPDFLoader)
    ftype = meta.get("filetype")
    label = Path(src).name if src else "unknown"
    if page is not None:
        label = f"{label} (p.{page + 1})"
    return {"label": label, "source": src, "page": page, "filetype": ftype}


def _doc_key(d) -> str:
    """Stable doc key for deduping across BM25/FAISS results."""
    meta = d.metadata or {}
    src = meta.get("source", "")
    page = str(meta.get("page", ""))
    head = (d.page_content or "")[:200]
    h = hashlib.md5(head.encode("utf-8", errors="ignore")).hexdigest()
    return f"{src}::{page}::{h}"

def _normalize_text(s: str) -> str:
    # join hyphenated line breaks, collapse newlines/spaces
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _is_junk_sentence(s: str) -> bool:
    s_stripped = s.strip()
    if len(s_stripped) < 20:
        return True
    if re.fullmatch(r"[\W_]+", s_stripped):
        return True
    lower = s_stripped.lower()
    if re.fullmatch(r"(part|section|appendix)\s+\d+[:.)]?", lower):
        return True
    if s_stripped.isupper() and len(s_stripped) < 80:
        return True  # short ALL-CAPS headings like "DOCUMENTS"
    return False

# ------------------------ sentence & context utils ----------------------

def _filter_sentences_for_question(text: str, question: str, max_chars: int = 1200) -> str:
    """
    Domain-agnostic pre-filter:
    keep sentences that share words with the question.
    Falls back to the first couple sentences if no overlap is found.
    """
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())

    picked = [s for s in sentences if any(w in s.lower() for w in q_words)]
    if not picked:
        picked = sentences[:2]  # graceful fallback

    out, used = [], 0
    for s in picked:
        s = s.strip()
        if not s:
            continue
        if used + len(s) > max_chars:
            break
        out.append(s)
        used += len(s) + 1

    return " ".join(out) if out else (text or "")[:max_chars]


# ----------------------------- core class ------------------------------

class RAGQuery:
    """
    Retrieval + (optional) OpenRouter summarization.

    Embeddings: sentence-transformers/all-mpnet-base-v2
    Lexical:    rank-bm25
    Vector:     FAISS
    Reranker:   BAAI/bge-reranker-base (optional)
    """

    def __init__(
        self,
        index_dir: str = "index",
        embed_model: str = "sentence-transformers/all-mpnet-base-v2",
        initial_k: int = 10,                  # retriever fan-out
        final_k: int = 4,                     # contexts kept
        use_reranker: bool = True,            # cross-encoder reranker
        reranker_model: str = "BAAI/bge-reranker-base",
    ) -> None:
        self.initial_k = initial_k
        self.final_k = max(1, final_k)
        self.use_reranker = use_reranker

        # Embeddings + FAISS
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vs = FAISS.load_local(index_dir, self.embeddings, allow_dangerous_deserialization=True)

        # Doc list for BM25
        self._docs: List[Any] = list(self.vs.docstore._dict.values())
        corpus = [(d.page_content or "") for d in self._docs]
        self._bm25_all = BM25Okapi([t.lower().split() for t in corpus])

        # FAISS retriever with MMR diversity
        self.retriever = self.vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": initial_k, "fetch_k": max(20, initial_k * 4), "lambda_mult": 0.5},
        )

        # Optional cross-encoder reranker
        self.reranker = Reranker(model_name=reranker_model) if use_reranker else None

    # ------------------------- retrieval --------------------------------

    def _hybrid_candidates(self, question: str) -> List[Any]:
        # BM25 lexical candidates
        ids = self._bm25_all.get_top_n(question.lower().split(), list(range(len(self._docs))), n=self.initial_k)
        lex_docs = [self._docs[i] for i in ids]
        # FAISS vector candidates
        vec_docs = self.retriever.invoke(question)
        # merge + dedupe
        merged: Dict[str, Any] = {}
        for d in lex_docs + vec_docs:
            merged[_doc_key(d)] = d
        return list(merged.values())

    def _retrieve_top(self, question: str) -> List[Any]:
        cands = self._hybrid_candidates(question)
        if self.use_reranker and cands:
            ranked = self.reranker.rerank(question, cands)  # [(score, doc), ...]
            return [doc for (_s, doc) in ranked[: self.final_k]]
        return cands[: self.final_k]

    # ---------------------- contexts / citations ------------------------

    def _build_contexts(
        self,
        docs: List[Any],
        question: str,
        max_ctx_chars: int = 8000,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Return (citations, numbered_context_string).
        We keep a numbered context for easy citing in prompts and UI.
        """
        parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        seen = set()

        for i, d in enumerate(docs, start=1):
            raw = _normalize_text((d.page_content or "").strip().replace("**", ""))
            chunk = _filter_sentences_for_question(raw, question, 1200)
            parts.append(f"Context [{i}]: {chunk}")

            meta = d.metadata or {}
            key = (meta.get("source"), meta.get("page"))
            if key not in seen:
                citations.append(_fmt_citation(meta))
                seen.add(key)

        ctx = "\n\n".join(parts)[:max_ctx_chars]
        return citations, ctx

    # ----------------------- evidence extraction ------------------------

    def _top_sentences(self, question: str, docs: List[Any], max_bullets: int = 6) -> List[Tuple[str, int]]:
        """
        Score individual sentences with BM25 and return up to max_bullets of
        (sentence_text, context_id) where context_id matches [n] in numbered context.
        """
        sents: List[str] = []
        ctx_ids: List[int] = []

        for idx, d in enumerate(docs, start=1):
            raw = (d.page_content or "")
            raw = _normalize_text(raw)
            for s in re.split(r"(?<=[.!?])\s+", raw):
                s2 = _normalize_text(s)
                if s2 and not _is_junk_sentence(s2):
                    sents.append(s2)
                    ctx_ids.append(idx)

        if not sents:
            return []

        bm = BM25Okapi([s.lower().split() for s in sents])
        ids = bm.get_top_n(question.lower().split(), list(range(len(sents))), n=max(20, max_bullets * 3))

        picked: List[Tuple[str, int]] = []
        seen_hashes = set()
        for i in ids:
            snippet = sents[i]
            h = hashlib.md5(snippet[:160].encode("utf-8", errors="ignore")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            picked.append((snippet, ctx_ids[i]))
            if len(picked) >= max_bullets:
                break
        return picked

    # --------------------------- public API -----------------------------

    def ask_evidence(self, question: str, max_bullets: int = 6) -> Dict[str, Any]:
        """Evidence-only mode: no LLM. Returns bullets with [n] plus citations."""
        docs = self._retrieve_top(question)
        citations, numbered_ctx = self._build_contexts(docs, question)
        pairs = self._top_sentences(question, docs, max_bullets=max_bullets)

        if not pairs:
            return {
                "answer": "I couldn't find a matching sentence. Try rephrasing or increase final_k.",
                "citations": citations,
                "numbered_context": numbered_ctx,
            }

        bullets = "\n".join([f"- {txt} [{cid}]" for (txt, cid) in pairs])
        return {"answer": bullets, "citations": citations, "numbered_context": numbered_ctx}

    # --------------------- OpenRouter summarization ---------------------

    @staticmethod
    def _openrouter_chat(api_key: str, model: str, messages: List[Dict[str, str]], max_tokens: int = 300) -> str:
        """
        Minimal OpenRouter Chat Completions call. Returns the assistant text,
        raises on non-200 or missing content.
        """
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-org/your-app",
            "X-Title": "RAG Chatbot",
        }
        resp = requests.post(
            url,
            headers=headers,
            json={"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("OpenRouter returned empty content")
        return content.strip()

    def ask_summarized(
        self,
        question: str,
        api_key: str,
        model: str = "meta-llama/llama-3.1-70b-instruct",
        max_bullets: int = 6,
    ) -> Dict[str, Any]:
        """
        Summarize ONLY the evidence bullets using an OpenRouter LLM.
        If api_key missing or call fails -> fall back to evidence-only.
        """
        base = self.ask_evidence(question, max_bullets=max_bullets)
        bullets = base["answer"].strip()
        citations = base["citations"]
        numbered_ctx = base["numbered_context"]

        if not api_key:
            return {"answer": bullets, "citations": citations, "numbered_context": numbered_ctx}

        system = (
            "You are a careful assistant. You will write a concise answer using ONLY the provided evidence. "
            "Write 2–4 sentences, paraphrased for clarity. Preserve inline numeric citations like [1], [2] "
            "that refer to the numbered contexts. Do not invent facts or citations. If evidence is insufficient, say so."
        )
        user = (
            f"Question: {question}\n\n"
            f"Numbered Contexts:\n{numbered_ctx}\n\n"
            f"Evidence bullets (each ends with its context number in [n]):\n{bullets}\n\n"
            "Answer (2–4 sentences, keep [n] citations):"
        )
        try:
            text = self._openrouter_chat(api_key=api_key, model=model, messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            max_ref = numbered_ctx.count("Context [")
            text = re.sub(r"\[(\d+)\]", lambda m: f"[{m.group(1)}]" if 1 <= int(m.group(1)) <= max_ref else "", text)
            cleaned = re.sub(r"\s+", " ", text).strip()
            if not cleaned:
                cleaned = bullets  # fallback
            return {"answer": cleaned, "citations": citations, "numbered_context": numbered_ctx}
        except Exception:
            return {"answer": bullets, "citations": citations, "numbered_context": numbered_ctx}
