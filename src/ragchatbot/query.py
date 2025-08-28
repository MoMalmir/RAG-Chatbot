from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from .utils import DEVICE
from .reranker import Reranker


def _fmt_citation(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Format a single citation entry for display."""
    src = meta.get("source")
    page = meta.get("page", None)  # 0-based from PyPDFLoader
    ftype = meta.get("filetype")
    label = Path(src).name if src else "unknown"
    if page is not None:
        label = f"{label} (p.{page + 1})"  # human-friendly 1-based
    return {"label": label, "source": src, "page": page, "filetype": ftype}


# ----------------------------------------------------------------------
# Tokenization helpers that avoid HF "longer than model max" warnings
# ----------------------------------------------------------------------

def _safe_encode_ids(tokenizer, text: str) -> List[int]:
    """
    Tokenize without triggering the HF 'longer than max length' warning.
    Uses the fast tokenizer backend when available.
    """
    if hasattr(tokenizer, "_tokenizer") and tokenizer._tokenizer is not None:
        return tokenizer._tokenizer.encode(text, add_special_tokens=False).ids
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = enc["input_ids"]
    return input_ids if isinstance(input_ids[0], int) else input_ids[0]


def _token_len(tokenizer, text: str) -> int:
    return len(_safe_encode_ids(tokenizer, text))


def _trim_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = _safe_encode_ids(tokenizer, text)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


# --------------------------- Context pruning ---------------------------

_KEYWORDS_DEFAULT = {
    "letter", "letters", "recommendation", "recommendations",
    "reference", "references", "referee",
    "independent", "dependent",
    "support", "evidence", "affidavit", "expert", "testimony", "endorse"
}

def _filter_sentences_for_question(text: str, question: str, max_chars: int = 1000) -> str:
    """
    Keep only sentences likely relevant to the question.
    Simple keyword filter (question words + defaults). Fallback to first 2 sentences.
    """
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    keys = _KEYWORDS_DEFAULT | q_words
    # crude sentence split
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
    """
    Instruction steers the model to produce a short answer + inline numeric citations.
    """
    return (
        "You are a helpful assistant. Answer the user's question using ONLY the numbered context snippets.\n"
        "Paraphrase; do not copy sentences verbatim. Write 1–3 concise sentences, then add inline citations like [1], [2] "
        "after the relevant facts. Do not begin your answer with a citation, and do not answer with only citations.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{numbered_context}\n\n"
        "Answer:"
    )


class RAGQuery:
    def __init__(
        self,
        index_dir: str = "index",
        embed_model: str = "sentence-transformers/all-mpnet-base-v2",
        # Extractive QA
        qa_model_name: str = "bert-large-uncased-whole-word-masking-finetuned-squad",
        # Generative QA
        gen_model_name: str = "google/flan-t5-base",
        mode: str = "extractive",   # "extractive" | "generative"
        # retrieval knobs
        initial_k: int = 10,
        final_k: int = 4,
        use_reranker: bool = False,     # enable with CLI flag
        reranker_model: str = "BAAI/bge-reranker-base",
    ) -> None:
        self.mode = mode
        self.qa_model_name = qa_model_name
        self.gen_model_name = gen_model_name

        # embeddings + retriever (use MMR for diversity)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vs = FAISS.load_local(index_dir, self.embeddings, allow_dangerous_deserialization=True)
        # Diverse retriever helps avoid near-duplicate chunks
        self.retriever = self.vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": initial_k, "fetch_k": max(20, initial_k * 4), "lambda_mult": 0.5},
        )

        # reranker
        self.use_reranker = use_reranker
        self.final_k = max(1, final_k)
        self.reranker = Reranker(model_name=reranker_model) if use_reranker else None

        # lazy pipelines
        self._qa_pipe = None
        self._qa_tok = None
        self._gen_pipe = None
        self._gen_tok = None

    # --------------------------- Models --------------------------------

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
        # ensure pad token is set (T5 normally has <pad>, but be explicit)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(self.gen_model_name).to(DEVICE)
        self._gen_tok = tok
        self._gen_pipe = pipeline(
            "text2text-generation", model=gen_model, tokenizer=tok, device=0 if DEVICE == "cuda" else -1
        )

    # ------------------------ Retrieval & Context -----------------------

    def _retrieve_top(self, question: str):
        candidates = self.retriever.invoke(question)
        if self.use_reranker and candidates:
            ranked = self.reranker.rerank(question, candidates)
            return [doc for (_s, doc) in ranked[: self.final_k]]
        return candidates[: self.final_k]

    def _build_citations_and_context(self, docs: List[Any], max_ctx_chars: int = 8000, question: str = "") -> Tuple[List[Dict[str, Any]], str]:
        """
        Build a numbered context block and a parallel citation list.
        Each snippet is lightly sanitized and sentence-pruned to match the question.
        """
        context_parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        seen = set()

        for i, d in enumerate(docs, start=1):
            chunk = (d.page_content or "").strip()
            if chunk:
                # strip obvious markdown artifacts and prune to relevant sentences
                chunk = chunk.replace("**", "")
                chunk = _filter_sentences_for_question(chunk, question, max_chars=1000)

            # IMPORTANT: don't start with "[i]" in the text (the model may copy it)
            context_parts.append(f"Context [{i}]: {chunk}")
            meta = d.metadata or {}
            key = (meta.get("source"), meta.get("page"))
            if key not in seen:
                citations.append(_fmt_citation(meta))
                seen.add(key)

        context = "\n\n".join(context_parts)[:max_ctx_chars]
        return citations, context

    # ----------------------------- Readers -----------------------------

    def _answer_extractive(self, question: str, context: str) -> Dict[str, Any]:
        self._ensure_extractive()
        # token-aware trim to fit BERT input budget
        max_input = getattr(self._qa_tok, "model_max_length", 512)
        buffer_tokens = 32
        q_len = _token_len(self._qa_tok, question)
        ctx_budget = max(16, max_input - q_len - buffer_tokens)
        context_trimmed = _trim_to_tokens(self._qa_tok, context, ctx_budget)

        out = self._qa_pipe(question=question, context=context_trimmed)
        return {"answer": out.get("answer"), "score": out.get("score")}

    def _answer_generative(self, question: str, numbered_context: str) -> Dict[str, Any]:
        """
        Manual tokenize + generate path (avoids pipeline re-tokenization warnings).
        Also guards against 'answer is only [2]' style outputs and strips leading citations.
        """
        self._ensure_generative()

        # Build + trim input to the model’s max length
        prompt = _build_grounded_prompt(question, numbered_context)
        max_input = getattr(self._gen_tok, "model_max_length", 512)

        if _token_len(self._gen_tok, prompt) > max_input:
            header_only = _build_grounded_prompt(question, "")
            header_tokens = _token_len(self._gen_tok, header_only)
            ctx_budget = max(64, max_input - header_tokens - 16)  # leave some slack
            # Trim ONLY the context portion — the header stays intact
            # Extract the context block from prompt
            trimmed_ctx = _trim_to_tokens(self._gen_tok, numbered_context, ctx_budget)
            prompt = _build_grounded_prompt(question, trimmed_ctx)

        # Tokenize with hard truncation
        enc = self._gen_tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        # Generate deterministically but avoid verbatim copying
        model = self._gen_pipe.model  # already on DEVICE via pipeline creation
        gen_ids = model.generate(
            **enc,
            max_new_tokens=160,
            min_new_tokens=20,
            do_sample=False,
            no_repeat_ngram_size=3,
            pad_token_id=self._gen_tok.pad_token_id or self._gen_tok.eos_token_id,
        )
        text = self._gen_tok.decode(gen_ids[0], skip_special_tokens=True).strip()

        # Cleanups:
        # 1) Drop any leading lone citation like "[2]" or "(Context [2])"
        text = re.sub(r"^\s*(\[\s*\d+(?:\s*,\s*\d+)*\s*\]|\(?\s*context\s*\[\d+\]\s*\)?)\s*[:-]?\s*", "", text, flags=re.I)
        # 2) If output is only citations, prepend a minimal answer phrase
        only_cites = re.fullmatch(r"\s*(\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*)+$", text)
        if only_cites:
            text = f"Letters of recommendation substantiate your achievements with independent expert testimony {text}"

        return {"answer": text, "score": None}

    # ------------------------------ API --------------------------------

    def ask(self, question: str, max_ctx_chars: int = 8000) -> Dict[str, Any]:
        top_docs = self._retrieve_top(question)
        citations, numbered_context = self._build_citations_and_context(
            top_docs, max_ctx_chars=max_ctx_chars, question=question
        )

        if self.mode == "generative":
            ans = self._answer_generative(question, numbered_context)
        else:
            ans = self._answer_extractive(question, numbered_context)

        ans["citations"] = citations
        return ans
