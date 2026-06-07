from __future__ import annotations

from typing import Any
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .config import settings
from .schema import SourceDocument
import json


class ChromaVectorStore:
    COLLECTION_NAME = "rag_documents"

    def __init__(self) -> None:
        self.persist_path = settings.persist_path
        try:
            self.client = chromadb.Client(
                ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.persist_path),
                )
            )
        except ValueError:
            # Newer chromadb versions use a different client configuration.
            # Fall back to a default client (in-memory) to allow the app to run.
            # For persistent storage or migration, follow Chroma migration docs.
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    def ingest(self, documents: list[SourceDocument]) -> dict[str, Any]:
        if not documents:
            return {"added": 0}

        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        # Chroma metadata values must be primitive types, lists, or None.
        # Flatten any nested dicts (e.g., `extra`) into JSON strings.
        metadatas = []
        for doc in documents:
            md = doc.metadata.model_dump()
            safe_md = {}
            for k, v in md.items():
                if isinstance(v, dict):
                    safe_md[k] = json.dumps(v)
                else:
                    safe_md[k] = v
            metadatas.append(safe_md)
        embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )
        return {"added": len(documents)}

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["ids", "documents", "metadatas", "distances"],
        )

        hits = []
        for doc_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )
        return hits

    def reset_collection(self) -> None:
        self.client.delete_collection(name=self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)
