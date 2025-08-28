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
def query(
    question: str = typer.Argument(..., help="Your question"),
    index_dir: str = typer.Option("index"),
    embed_model: str = typer.Option("sentence-transformers/all-mpnet-base-v2"),
    qa_model: str = typer.Option("bert-large-uncased-whole-word-masking-finetuned-squad"),
    mode: str = typer.Option("extractive", help="extractive | generative"),
    gen_model: str = typer.Option("google/flan-t5-base", help="Generative model name"),
    initial_k: int = typer.Option(10, "--initial-k", "--initial_k", help="Retriever candidates before rerank"),
    final_k: int = typer.Option(4, "--final-k", "--final_k", help="Contexts kept"),
    use_reranker: bool = typer.Option(False, "--use-reranker/--no-use-reranker", help="Enable reranker"),
    reranker_model: str = typer.Option("BAAI/bge-reranker-base", help="Cross-encoder model"),
):
    rag = RAGQuery(
        index_dir=index_dir,
        embed_model=embed_model,
        qa_model_name=qa_model,
        gen_model_name=gen_model,
        mode=mode,
        initial_k=initial_k,
        final_k=final_k,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
    )
    result = rag.ask(question)
    score = result.get("score")
    print({
        "answer": result["answer"],
        "score": round(score, 4) if isinstance(score, (int, float)) else None,
        "citations": result["citations"],
    })


if __name__ == "__main__":
    app()
