from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.rag_app.config import settings
from src.rag_app.schema import SourceDocument
from src.rag_app.vector_store import ChromaVectorStore


def load_documents_from_file(path: Path) -> list[SourceDocument]:
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("Document input file must be a JSON array of document objects.")

    documents = []
    for item in data:
        documents.append(SourceDocument(**item))
    return documents


def main() -> None:
    load_dotenv(dotenv_path=Path(".env"))
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--source-file", type=Path, required=True, help="JSON file with documents to ingest.")
    args = parser.parse_args()

    documents = load_documents_from_file(args.source_file)
    store = ChromaVectorStore()
    result = store.ingest(documents)
    print(f"Ingested {result['added']} documents into {settings.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
