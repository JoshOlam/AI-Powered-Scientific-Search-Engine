from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .chunking import chunk_documents
from .config import get_settings
from .miner import mine_documents
from .reasoning import compose_answer, decompose_question

app = FastAPI(title="Scientific Search API", version="0.1.0")


def get_vector_store_class():
    # Delayed import keeps API startup/tests lightweight.
    from .vector_store import VectorStore

    return VectorStore


class BuildIndexRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    arxiv_docs: int = Field(default=8, ge=0, le=20)
    pubmed_docs: int = Field(default=0, ge=0, le=20)
    chunk_size: int = Field(default=550, ge=200, le=4000)
    chunk_overlap: int = Field(default=1, ge=0, le=10)

    @model_validator(mode="after")
    def validate_sources(self) -> "BuildIndexRequest":
        if self.arxiv_docs + self.pubmed_docs <= 0:
            raise ValueError("At least one source document count must be positive.")
        return self


class IndexedDocument(BaseModel):
    source: str
    doc_id: str
    title: str
    url: str
    pdf_url: str | None = None
    pdf_path: str | None = None


class BuildIndexResponse(BaseModel):
    topic: str
    mined_documents: int
    chunks: int
    out_dir: str
    documents: List[IndexedDocument]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResult(BaseModel):
    chunk_id: str
    source: str
    doc_id: str
    title: str
    url: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    top_k: int
    results: List[QueryResult]


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_substeps: int = Field(default=4, ge=1, le=8)
    top_k_per_step: int = Field(default=3, ge=1, le=10)


class AnswerStep(BaseModel):
    sub_question: str
    results: List[QueryResult]


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    score: float


class AnswerResponse(BaseModel):
    question: str
    sub_questions: List[str]
    steps: List[AnswerStep]
    answer: str
    citations: List[Citation]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/build-index", response_model=BuildIndexResponse)
def build_index(payload: BuildIndexRequest) -> BuildIndexResponse:
    try:
        settings = get_settings()
        vector_store_class = get_vector_store_class()
        docs = mine_documents(
            topic=payload.topic,
            arxiv_docs=payload.arxiv_docs,
            pubmed_docs=payload.pubmed_docs,
        )
        chunks = chunk_documents(
            docs,
            max_chars=payload.chunk_size,
            overlap_sentences=payload.chunk_overlap,
        )
        store = vector_store_class()
        store.build(chunks)
        store.save(settings.vector_store_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {exc}") from exc

    return BuildIndexResponse(
        topic=payload.topic,
        mined_documents=len(docs),
        chunks=len(chunks),
        out_dir=settings.vector_store_dir,
        documents=[
            IndexedDocument(
                source=doc.source,
                doc_id=doc.doc_id,
                title=doc.title,
                url=doc.url,
                pdf_url=doc.pdf_url,
                pdf_path=doc.pdf_path,
            )
            for doc in docs
        ],
    )


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        settings = get_settings()
        vector_store_class = get_vector_store_class()
        store = vector_store_class.load(settings.vector_store_dir)
        results = store.search(payload.query, top_k=payload.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vector store not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query index: {exc}") from exc

    normalized = [QueryResult(**item) for item in results]
    return QueryResponse(query=payload.query, top_k=payload.top_k, results=normalized)


@app.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest) -> AnswerResponse:
    try:
        settings = get_settings()
        vector_store_class = get_vector_store_class()
        store = vector_store_class.load(settings.vector_store_dir)

        sub_questions = decompose_question(payload.question, max_steps=payload.max_substeps)
        if not sub_questions:
            sub_questions = [payload.question]

        raw_steps = []
        for sub_question in sub_questions:
            results = store.search(sub_question, top_k=payload.top_k_per_step)
            normalized = [QueryResult(**item) for item in results]
            raw_steps.append({"sub_question": sub_question, "results": normalized})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Vector store not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc

    answer_text = compose_answer(
        payload.question,
        [
            {
                "sub_question": step["sub_question"],
                "results": [item.model_dump() for item in step["results"]],
            }
            for step in raw_steps
        ],
    )

    citation_map: dict[str, Citation] = {}
    for step in raw_steps:
        for result in step["results"]:
            if result.chunk_id in citation_map:
                continue
            citation_map[result.chunk_id] = Citation(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=result.title,
                url=result.url,
                score=result.score,
            )

    return AnswerResponse(
        question=payload.question,
        sub_questions=sub_questions,
        steps=[AnswerStep(sub_question=step["sub_question"], results=step["results"]) for step in raw_steps],
        answer=answer_text,
        citations=list(citation_map.values()),
    )
