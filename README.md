# RAG Service

A production-ready Retrieval-Augmented Generation (RAG) application built with Python, FastAPI, LangChain, ChromaDB, Sentence Transformers, and Cross-Encoder reranking.

## Features

- Document embedding using `sentence-transformers`
- Persistent vector storage with `chromadb`
- Two-stage retrieval: vector search + cross-encoder reranking
- OpenAI or local LLM support
- FastAPI REST API with request validation
- Environment configuration via `.env`

## Quickstart

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy the example environment file and set your values:

```powershell
copy .env.example .env
```

4. Ingest documents using the sample script or your own ingestion flow.

5. Start the API:

```powershell
uvicorn app:app --reload
```

6. Query the service:

```powershell
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query":"What is RAG?"}'
```

## Environment variables

See `.env.example`.

## Project structure

- `src/rag_app/` - application modules
- `scripts/ingest.py` - document ingestion script
- `app.py` - FastAPI entrypoint
- `requirements.txt` - Python dependencies
- `.env.example` - environment variable template
