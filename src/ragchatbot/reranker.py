# src/ragchatbot/reranker.py
from typing import List, Tuple
from sentence_transformers import CrossEncoder
import torch

class Reranker:
    """
    Cross-encoder reranker using a bi-encoder retrieval candidate pool.
    Default model: BAAI/bge-reranker-base (great CPU choice).
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None):
        # device: "cuda" | "cpu" | None (auto)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)

    def score(self, query: str, passages: List[str], batch_size: int = 16) -> List[float]:
        pairs = [(query, p) for p in passages]
        # predict returns a numpy array of scores
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return scores.tolist()

    def rerank(self, query: str, docs) -> List[Tuple[float, any]]:
        """
        Returns a list of (score, doc) sorted descending by score.
        `docs` is a list of LC Documents.
        """
        passages = [d.page_content for d in docs]
        scores = self.score(query, passages)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return ranked
