"""
FastAPI Backend — RAG Chatbot Luật Viễn thông.
"""

import json
import os
import time
import requests
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ChatResponse, SourceInfo, HealthResponse
from src.api.dependencies import get_engine
from src.core.rag_engine import RAGEngine
from src.core.config import DEVICE, OLLAMA_BASE_URL, LLM_MODEL, QDRANT_DIR
from src.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="RAG Chatbot Luật Viễn thông",
    description="API chatbot hỏi đáp về Luật Viễn thông Việt Nam (Luật số 24/2023/QH15).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse, summary="Hỏi chatbot")
def chat(req: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    start = time.time()
    history = [{"role": m.role, "content": m.content} for m in req.history]

    try:
        result = engine.query(req.question, history)
    except Exception as e:
        logger.error(f"RAG error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

    elapsed = time.time() - start

    sources = [
        SourceInfo(
            source=s.get("source", ""),
            dieu=s.get("dieu"),
            khoan=s.get("khoan"),
            tieu_de=s.get("tieu_de"),
            chuong=s.get("chuong"),
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        time_seconds=round(elapsed, 2),
    )


@app.post("/api/chat/stream", summary="Streaming response (SSE)")
def chat_stream(req: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    history = [{"role": m.role, "content": m.content} for m in req.history]

    try:
        generator, sources = engine.query_stream(req.question, history)
    except Exception as e:
        logger.error(f"RAG streaming error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

    def event_stream():
        for chunk in generator:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        sources_payload = json.dumps(
            [
                {
                    "source": s.get("source", ""),
                    "dieu": s.get("dieu"),
                    "khoan": s.get("khoan"),
                    "tieu_de": s.get("tieu_de"),
                    "chuong": s.get("chuong"),
                }
                for s in sources
            ],
            ensure_ascii=False,
        )
        yield f"data: [SOURCES]{sources_payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/health", response_model=HealthResponse, summary="Health check")
def health():
    ollama_status = "unknown"
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.ok:
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(LLM_MODEL in m for m in models):
                ollama_status = f"ok ({LLM_MODEL} loaded)"
            else:
                ollama_status = f"running but {LLM_MODEL} not found"
        else:
            ollama_status = "error"
    except Exception:
        ollama_status = "offline"

    vs_status = "ok" if os.path.exists(QDRANT_DIR) else "not built"

    return HealthResponse(
        status="ok" if ollama_status.startswith("ok") and vs_status == "ok" else "degraded",
        engine="RAG Hybrid Search + Reranking",
        device=DEVICE,
        ollama=ollama_status,
        vectorstore=vs_status,
    )


@app.get("/", summary="Root")
def root():
    return {
        "name": "RAG Chatbot Luật Viễn thông",
        "docs": "/docs",
        "health": "/api/health",
    }
