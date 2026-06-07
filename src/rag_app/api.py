from fastapi import FastAPI, HTTPException

from .config import settings
from .llm import LLMService
from .reranker import CrossEncoderReranker
from .schema import IngestRequest, QueryRequest, QueryResponse, SourceResponse
from .vector_store import ChromaVectorStore


app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation API with a two-stage retrieval pipeline.",
    version="1.0",
)

store = ChromaVectorStore()
reranker = CrossEncoderReranker()
llm_service = LLMService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, int]:
    result = store.ingest(request.documents)
    return {"status": 200, "ingested": result["added"]}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    candidates = store.search(payload.query, top_k=settings.CANDIDATE_POOL)
    if not candidates:
        raise HTTPException(status_code=404, detail="No documents are currently indexed.")

    reranked = reranker.rerank(payload.query, candidates, top_k=settings.TOP_K_RERANK)
    sources = [
        SourceResponse(
            id=item["id"],
            text=item["text"],
            metadata=item["metadata"],
            score=float(item["score"]),
        )
        for item in reranked
    ]

    answer = llm_service.generate_answer(payload.query, sources)

    return QueryResponse(answer=answer, sources=sources)
