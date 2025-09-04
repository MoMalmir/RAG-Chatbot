---
title: RAG Chatbot — Streamlit
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# 🤖 RAG Chatbot — Streamlit (HF Spaces) 

**Live demo:** https://huggingface.co/spaces/momalmir/rag-chatbot-doc-qa  
**Stack:** Streamlit • LangChain • FAISS • sentence-transformers • BM25 • MMR • Cross‑encoder re‑ranker • OpenRouter (optional)

This project is a **document‑question‑answering** app that lets users upload files, build a temporary or shared index, and **ask grounded questions**. It runs as a **Streamlit** app on **Hugging Face Spaces** and can also be used locally.

[![🤗 Spaces](https://img.shields.io/badge/🤗%20Spaces-Deployed-blue)](https://huggingface.co/spaces/momalmir/rag-chatbot-doc-qa)

---

![Demo](assets/demo.gif)



## 🧭 Two‑Mode RAG (domain‑agnostic)

### 1) **Evidence‑only**
- Returns **top‑matching sentences** with inline **`[n]` citations** (no LLM).  
- Retrieval pipeline:
  - **BM25** (lexical) builds a corpus over all chunks and gives sparse scores.
  - **FAISS** (vector) does semantic ANN search using Hugging Face embeddings.
  - **MMR** (**m**aximal **m**arginal **r**elevance) on the FAISS retriever for diversity.
  - **Optional Cross‑encoder re‑rank** (`BAAI/bge-reranker-base`) for precision at top‑k.
- Output = **bulleted evidence list** (up to `Max bullets`) with `[n]` mapping to numbered contexts.

### 2) **Summarize with OpenRouter**
- Uses an LLM (via OpenRouter) to generate **2–4 sentences** that **only summarize the evidence bullets**, preserving `[n]` citations.
- If the API key is missing or the call fails, the app cleanly **falls back** to Evidence‑only mode.
- Great for compact, interview‑ready answers that remain **grounded** in retrieved text.

---

## 🔎 Hybrid Retrieval (BM25 + FAISS + MMR)

```
Question → BM25 over chunks  → top lexical docs
         → FAISS (MMR)       → top semantic docs (diverse)
         → Merge + dedupe    → candidate set
         → Cross‑encoder     → rerank for precision (optional)
         → Keep top final_k  → contexts
```

- **BM25** (`rank_bm25`) excels at **exact term matching**, headers, and tabular text.
- **FAISS** + Hugging Face embeddings excels at **semantic similarity**.
- **MMR** (via `as_retriever(search_type="mmr")`) reduces near‑duplicates and improves coverage.
- **Cross‑encoder** (bge‑reranker‑base) reads **(query, passage)** pairs and re‑scores them for **precision@k**.

> UI knobs map to this pipeline: `initial_k` (candidate breadth), `final_k` (kept contexts), `Use cross‑encoder reranker` (on/off).

---

## 🔗 Citation Mechanism: `[n]`

- We build a **numbered context block**:
  ```
  Context [1]: ... (sentences clipped for question)
  Context [2]: ...
  ```
- Evidence bullets end with `[n]` where `n` is the **context number** the sentence came from.  
- The UI also shows a **citation panel** with entries like
  - `filename.pdf (p.3)` or `notes.md` (page optional)
- In **Summarize mode**, the prompt **requires the LLM to keep `[n]`** in the final text and avoid inventing citations.

### Why this design?
- `[n]` is language‑agnostic, compact, and easy to render in Streamlit.
- It keeps the answer auditable: users can jump to **Context [n]** and verify.

---

## 🧱 Text & Sentence Handling

- **Normalization**: join hyphenated line breaks, collapse spaces/newlines, strip Markdown artifacts (`**`), etc.
- **Question‑aware filtering**: from each chunk we keep sentences that **share words** with the question (fallback to first sentences).
- **Junk filter**: drop very short fragments, all‑caps headings, section labels, and symbol lines.
- **Top‑sentences**: score individual sentences with **BM25** over sentences; keep diverse top‑N bullets (de‑dup by hash).

---

## ⚙️ Sidebar Controls (UI → Internals)

| UI Control | Meaning | Internal |
|---|---|---|
| **Embedding model** | sentence-transformers encoder | `HuggingFaceEmbeddings(model_name=...)` |
| **initial_k** | retriever fan‑out | FAISS `k` + BM25 top‑K for hybrid merge |
| **final_k** | contexts kept | top `final_k` after (optional) re‑rank |
| **Use cross‑encoder reranker** | precision boost | `BAAI/bge-reranker-base` |
| **Max bullets** | evidence sentences | top‑N via BM25(on sentences) |
| **chunk_size / chunk_overlap** | chunking strategy | affects recall and context integrity |
| **Index directory** | persistence option | `index/` (shared) or in‑memory session |

---

## 🛠️ Local Run

```bash
git clone https://github.com/<your-username>/RAG-Chatbot.git
cd RAG-Chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**.env (optional for summarize mode)**
```ini
OPENROUTER_API_KEY=your_key_here
```

---


## 🧪 Usage Pattern

1) **Ingest Documents**: Drop files, set `chunk_size/overlap`, build **in‑memory** index (per‑session) or persist to `/index` (shared).  
2) **Ask a Question**: Switch tab, choose **Evidence** or **Summarize**, tune `initial_k/final_k`, toggle **re‑ranker**.  
3) **Read result**: Bullets with `[n]` + citation panel; or a short LLM‑based summary that preserves `[n]`.

---

## ❓ Troubleshooting

- **Low recall** → increase `initial_k`, reduce `chunk_size`, enable MMR.
- **Off‑topic bullets** → enable cross‑encoder, lower `final_k`.
- **LLM fails** → check `OPENROUTER_API_KEY`; fallback should show evidence bullets.
- **Scanned PDFs** → OCR not included; pre‑OCR or add an OCR step.

---

## 📜 License & Credits

- MIT License
- sentence-transformers, FAISS, BAAI/bge‑reranker‑base, LangChain, Hugging Face Spaces.

---

## 👤 Author

**Mostafa Malmir (MoMalmir)**  
🔗 HF Space: [huggingface.co/spaces/momalmir](https://huggingface.co/spaces/momalmir)  
🔗 GitHub: [github.com/momalmir](https://github.com/momalmir)
🔗 LinkedIn: [linkedin.com/in/mostafa-malmir](https://linkedin.com/in/mostafa-malmir)

