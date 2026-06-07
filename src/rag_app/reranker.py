from __future__ import annotations

from typing import Iterable
from sentence_transformers import CrossEncoder

from .config import settings


class CrossEncoderReranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(settings.RERANKER_MODEL)

    def rerank(self, query: str, candidates: Iterable[dict], top_k: int = 10) -> list[dict]:
        if not candidates:
            return []

        pairs = [[query, candidate["text"]] for candidate in candidates]
        scores = self.model.predict(pairs, convert_to_numpy=True)

        ranked: list[dict] = []
        for candidate, score in zip(candidates, scores.tolist()):
            ranked.append(
                {
                    "id": candidate["id"],
                    "text": candidate["text"],
                    "metadata": candidate["metadata"],
                    "score": float(score),
                    "distance": candidate.get("distance"),
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]
