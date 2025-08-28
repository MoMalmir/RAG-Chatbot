import os
import json
import typer
from rich import print
from typing import List

from .ingest import ingest_paths
from .query import RAGQuery
from .utils import device_info

app = typer.Typer(add_completion=False, help="RAG Chatbot CLI")


@app.command()
def info():
    dev, is_cuda = device_info()
    print({"device": dev, "cuda": is_cuda})


@app.command()
def ingest(
    inputs: List[str] = typer.Argument(..., help="Files and/or directories"),
    index_dir: str = typer.Option("index", help="Where to store the FAISS index"),
    recursive: bool = typer.Option(True, help="Recurse folders"),
    include: str = typer.Option("*.pdf,*.docx,*.md,*.txt,*.html,*.htm", help="Include glob(s), comma-separated"),
    exclude: str = typer.Option(None, help="Exclude glob(s), comma-separated"),
    chunk_size: int = typer.Option(1000),
    chunk_overlap: int = typer.Option(200),
    embed_model: str = typer.Option("sentence-transformers/all-mpnet-base-v2"),
):
    out = ingest_paths(
        inputs=inputs,
        index_dir=index_dir,
        recursive=recursive,
        include=include,
        exclude=exclude,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
    )
    print({"indexed_to": out})


@app.command()
def evidence(
    question: str = typer.Argument(..., help="Your question"),
    index_dir: str = typer.Option("index"),
    embed_model: str = typer.Option("sentence-transformers/all-mpnet-base-v2"),
    initial_k: int = typer.Option(10, "--initial-k", "--initial_k", help="Retriever candidates before rerank"),
    final_k: int = typer.Option(4, "--final-k", "--final_k", help="Contexts kept"),
    use_reranker: bool = typer.Option(True, "--use-reranker/--no-use-reranker", help="Enable reranker"),
    reranker_model: str = typer.Option("BAAI/bge-reranker-base", help="Cross-encoder model"),
    max_bullets: int = typer.Option(6, help="Evidence sentences to return"),
):
    rag = RAGQuery(
        index_dir=index_dir,
        embed_model=embed_model,
        initial_k=initial_k,
        final_k=final_k,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
    )
    result = rag.ask_evidence(question, max_bullets=max_bullets)
    print(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def summarize(
    question: str = typer.Argument(..., help="Your question"),
    index_dir: str = typer.Option("index"),
    embed_model: str = typer.Option("sentence-transformers/all-mpnet-base-v2"),
    initial_k: int = typer.Option(10, "--initial-k", "--initial_k", help="Retriever candidates before rerank"),
    final_k: int = typer.Option(4, "--final-k", "--final_k", help="Contexts kept"),
    use_reranker: bool = typer.Option(True, "--use-reranker/--no-use-reranker", help="Enable reranker"),
    reranker_model: str = typer.Option("BAAI/bge-reranker-base", help="Cross-encoder model"),
    max_bullets: int = typer.Option(6, help="Evidence sentences to summarize"),
    or_key: str = typer.Option(None, "--or-key", help="OpenRouter API key", envvar="OPENROUTER_API_KEY"),
    or_model: str = typer.Option("meta-llama/llama-3.1-70b-instruct", "--or-model", help="OpenRouter model"),
):
    rag = RAGQuery(
        index_dir=index_dir,
        embed_model=embed_model,
        initial_k=initial_k,
        final_k=final_k,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
    )
    result = rag.ask_summarized(question, api_key=or_key or "", model=or_model, max_bullets=max_bullets)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
