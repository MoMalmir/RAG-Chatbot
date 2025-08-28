from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import hashlib

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from rank_bm25 import BM25Okapi

from .utils import DEVICE
from .reranker import Reranker


def _fmt_citation(meta: Dict[str, Any]) -> Dict[str, Any]:
    src = meta.get("source")
    page = meta.get("page", None)  # 0-based from PyPDFLoader
    ftype = meta.get("filetype")
    label = Path(src).name if src else "unknown"
    if page is not None:
        label = f"{label} (p.{page + 1})"
    return {"label": label, "source": src, "page": page, "filetype": ftype}


# -------------------- token helpers (no HF warnings) --------------------

def _safe_encode_ids(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "_tokenizer") and tokenizer._tokenizer is not None:
        return tokenizer._tokenizer.encode(text, add_special_tokens=False).ids
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"]
    return ids if isinstance(ids[0], int) else ids[0]

def _token_len(tokenizer, text: str) -> int:
    return len(_safe_encode_ids(tokenizer, text))

def _trim_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = _safe_encode_ids(tokenizer, text)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


# ------------------------ sentence & context utils ----------------------

_LEASE_KEYS = {
    "guest", "guests", "visitor", "visitors", "overnight", "occupancy", "tenant",
    "landlord", "rent", "deposit", "security", "fee", "key", "parking", "policy",
    "rule", "rules", "violation", "notice"
}
_NIW_KEYS = {
    "letter", "letters", "recommendation", "recommendations",
    "reference", "references", "referee",
    "independent", "dependent", "support", "evidence", "expert"
}

def _filter_sentences_for_question(text: str, question: str, max_chars: int = 1200) -> str:
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    keys = q_words | _LEASE_KEYS | _NIW_KEYS
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    picked = [s for s in sentences if any(k in s.lower() for k in keys)]
    if not picked:
        picked = sentences[:2]
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

def _build_grounded_prompt(question: str, numbered_context: str) -> str:
    return (
        "You are a helpful assistant. Answer the user's question using ONLY the numbered context snippets.\n"
        "Paraphrase; do not copy full sentences. Write 1–3 concise sentences, then add inline citations like [1], [2]. "
        "Do not begin with a citation and do not answer with only citations.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{numbered_context}\n\n"
        "Answer:"
    )

def _doc_key(d) -> str:
    meta = d.metadata or {}
    src = meta.get("source", "")
    page = str(meta.get("page", ""))
    h = hashlib.md5((d.page_content or "")[:200].encode("utf-8", errors="ignore")).hexdigest()
    return f"{src}::{page}::{h}"


class RAGQuery:
    def __init__(
        self,
        index_dir: str = "index",
        embed_model: str = "sentence-transformers/all-mpnet-base-v2",
        qa_model_name: str = "bert-large-uncased-whole-word-masking-finetuned-squad",
        gen_model_name: str = "google/flan-t5-base",
        mode: str = "extractive",             # "extractive" | "generative"
        initial_k: int = 10,
        final_k: int = 4,
        use_reranker: bool = True,            # default ON now
        reranker_model: str = "BAAI/bge-reranker-base",
    ) -> None:
        self.mode = mode
        self.qa_model_name = qa_model_name
        self.gen_model_name = gen_model_name
        self.initial_k = initial_k
        self.final_k = max(1, final_k)

        # vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vs = FAISS.load_local(index_dir, self.embeddings, allow_dangerous_deserialization=True)

        # build a BM25 index over *all* chunk texts from the vector store
        self._docs: List[Any] = list(self.vs.docstore._dict.values())
        corpus = [(d.page_content or "") for d in self._docs]
        self._bm25 = BM25Okapi([t.lower().split() for t in corpus])

        # diverse vector retriever (MMR)
        self.retriever = self.vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.initial_k, "fetch_k": max(20, self.initial_k * 4), "lambda_mult": 0.5},
        )

        # reranker
        self.use_reranker = use_reranker
        self.reranker = Reranker(model_name=reranker_model) if use_reranker else None

        # lazy models
        self._qa_pipe = None
        self._qa_tok = None
        self._gen_pipe = None
        self._gen_tok = None

    # --------------------------- models --------------------------------

    def _ensure_extractive(self):
        if self._qa_pipe is not None:
            return
        tok = AutoTokenizer.from_pretrained(self.qa_model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            self.qa_model_name,
            torch_dtype=(__import__("torch").float16 if DEVICE == "cuda" else __import__("torch").float32),
        ).to(DEVICE)
        self._qa_tok = tok
        self._qa_pipe = pipeline(
            "question-answering", model=qa_model, tokenizer=tok, device=0 if DEVICE == "cuda" else -1
        )

    def _ensure_generative(self):
        if self._gen_pipe is not None:
            return
        tok = AutoTokenizer.from_pretrained(self.gen_model_name)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(self.gen_model_name).to(DEVICE)
        self._gen_tok = tok
        self._gen_pipe = pipeline(
            "text2text-generation", model=gen_model, tokenizer=tok, device=0 if DEVICE == "cuda" else -1
        )

    # -------------------- retrieval / context building ------------------

    def _hybrid_candidates(self, question: str) -> List[Any]:
        # lexical
        lex_ids = self._bm25.get_top_n(question.lower().split(), list(range(len(self._docs))), n=self.initial_k)
        lex_docs = [self._docs[i] for i in lex_ids]

        # vector
        vec_docs = self.retriever.invoke(question)

        # merge & dedupe by (source,page,head hash)
        merged: Dict[str, Any] = {}
        for d in lex_docs + vec_docs:
            merged[_doc_key(d)] = d
        return list(merged.values())

    def _retrieve_top(self, question: str) -> List[Any]:
        cands = self._hybrid_candidates(question)
        if self.use_reranker and cands:
            ranked = self.reranker.rerank(question, cands)  # -> list[(score, doc)]
            return [doc for (_s, doc) in ranked[: self.final_k]]
        return cands[: self.final_k]

    def _build_citations_and_context(self, docs: List[Any], max_ctx_chars: int = 8000, question: str = "") -> Tuple[List[Dict[str, Any]], str]:
        parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        seen = set()

        for i, d in enumerate(docs, start=1):
            chunk = (d.page_content or "").strip().replace("**", "")
            chunk = _filter_sentences_for_question(chunk, question, max_chars=1200)
            parts.append(f"Context [{i}]: {chunk}")
            meta = d.metadata or {}
            key = (meta.get("source"), meta.get("page"))
            if key not in seen:
                citations.append(_fmt_citation(meta))
                seen.add(key)

        context = "\n\n".join(parts)[:max_ctx_chars]
        return citations, context

    # ----------------------------- readers ------------------------------

    def _answer_extractive(self, question: str, context: str) -> Dict[str, Any]:
        self._ensure_extractive()
        max_input = getattr(self._qa_tok, "model_max_length", 512)
        buffer_tokens = 32
        q_len = _token_len(self._qa_tok, question)
        ctx_budget = max(16, max_input - q_len - buffer_tokens)
        context_trimmed = _trim_to_tokens(self._qa_tok, context, ctx_budget)
        out = self._qa_pipe(question=question, context=context_trimmed)
        return {"answer": out.get("answer"), "score": out.get("score")}

    def _answer_generative(self, question: str, numbered_context: str) -> Dict[str, Any]:
        self._ensure_generative()
        prompt = _build_grounded_prompt(question, numbered_context)
        max_input = getattr(self._gen_tok, "model_max_length", 512)

        if _token_len(self._gen_tok, prompt) > max_input:
            header_only = _build_grounded_prompt(question, "")
            header_tokens = _token_len(self._gen_tok, header_only)
            ctx_budget = max(64, max_input - header_tokens - 16)
            trimmed = _trim_to_tokens(self._gen_tok, numbered_context, ctx_budget)
            prompt = _build_grounded_prompt(question, trimmed)

        enc = self._gen_tok(prompt, return_tensors="pt", truncation=True, max_length=max_input)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        model = self._gen_pipe.model
        gen_ids = model.generate(
            **enc,
            max_new_tokens=160,
            min_new_tokens=24,
            do_sample=False,
            no_repeat_ngram_size=3,
            pad_token_id=self._gen_tok.pad_token_id or self._gen_tok.eos_token_id,
        )
        text = self._gen_tok.decode(gen_ids[0], skip_special_tokens=True).strip()

        # strip leading lone citation & guard "only citations"
        text = re.sub(r"^\s*(\[\s*\d+(?:\s*,\s*\d+)*\s*\]|\(?\s*context\s*\[\d+\]\s*\)?)\s*[:-]?\s*", "", text, flags=re.I)
        only_cites = re.fullmatch(r"\s*(\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*)+$", text)
        if only_cites:
            text = f"This is defined in the cited clause {text}"
        return {"answer": text, "score": None}

    # --------------------------- strict mode ----------------------------

    def _answer_strict_sentences(self, question: str, docs: List[Any], top_sentences: int = 4) -> str:
        # collect sentences with their context id
        sents, ctx_ids = [], []
        for idx, d in enumerate(docs, start=1):
            for s in re.split(r"(?<=[.!?])\s+", (d.page_content or "")):
                s2 = s.strip()
                if s2:
                    sents.append(s2)
                    ctx_ids.append(idx)
        if not sents:
            return ""

        bm = BM25Okapi([s.lower().split() for s in sents])
        ids = bm.get_top_n(question.lower().split(), list(range(len(sents))), n=top_sentences)
        picked = []
        seen = set()
        for i in ids:
            snippet = sents[i]
            # de-duplicate near-identical sentences
            h = hashlib.md5(snippet[:120].encode("utf-8", errors="ignore")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            picked.append(f"- {snippet} [{ctx_ids[i]}]")
        if not picked:
            return ""
        return "\n".join(picked)

    # ------------------------------ API --------------------------------

    def ask(self, question: str, max_ctx_chars: int = 8000) -> Dict[str, Any]:
        docs = self._retrieve_top(question)
        citations, numbered_context = self._build_citations_and_context(docs, max_ctx_chars=max_ctx_chars, question=question)

        if self.mode == "generative":
            ans = self._answer_generative(question, numbered_context)
        else:
            ans = self._answer_extractive(question, numbered_context)

        ans["citations"] = citations
        return ans

    def ask_strict(self, question: str, max_ctx_chars: int = 8000, top_sentences: int = 4) -> Dict[str, Any]:
        """
        No LM generation. Return the top BM25-matched sentences with inline [n] citations.
        Rock-solid for policy/lease Qs.
        """
        docs = self._retrieve_top(question)
        citations, _ = self._build_citations_and_context(docs, max_ctx_chars=max_ctx_chars, question=question)
        answer = self._answer_strict_sentences(question, docs, top_sentences=top_sentences)
        if not answer:
            answer = "I couldn't find an exact sentence. Try rephrasing or reduce final_k."
        return {"answer": answer, "score": None, "citations": citations}
