"""FastAPI dependencies — RAG Engine singleton."""

from src.core.rag_engine import RAGEngine

_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    """Get or create RAG Engine singleton."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.initialize()
    return _engine
