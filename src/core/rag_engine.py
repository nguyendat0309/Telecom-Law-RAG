"""
RAG Engine — Hybrid Search + Reranking cho Luật Viễn thông.
"""

import json
import os
import torch
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from src.core.config import (
    DEVICE, CHUNKS_FILE, QDRANT_DIR,
    OLLAMA_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    RERANKER_MODEL, COLLECTION_NAME,
    BM25_WEIGHT, SEMANTIC_WEIGHT, RRF_K,
    TOP_K_RETRIEVE, TOP_K_RERANK, RERANK_SCORE_THRESHOLD,
    SYSTEM_PROMPT, CONDENSE_PROMPT,
    LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


class RAGEngine:
    """RAG Engine singleton."""

    def __init__(self):
        self._parent_map: dict[int, str] = {}
        self._bm25_index = None
        self._bm25_documents = None
        self._reranker = None
        self._vectorstore = None
        self._llm = None
        self._prompt = None
        self._condense_prompt = None

        self._reference_words = {
            "nó", "đó", "điều đó", "điều này", "vấn đề này", "vấn đề đó",
            "họ", "chúng", "các điều trên", "như vậy", "vậy thì", "còn",
            "thế nào", "còn gì", "thêm gì", "cụ thể hơn", "rõ hơn",
            "điều luật đó", "chương đó", "khoản đó",
        }

    def _lf_callback(self, question: str = None):
        """Return LangChain CallbackHandler cho Langfuse v3. None nếu chưa cấu hình."""
        if not (LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY):
            return None
        try:
            from langfuse.langchain import CallbackHandler
            ctx = {"name": "rag-query", "input": {"question": question}} if question else {}
            return CallbackHandler(trace_context=ctx)
        except Exception as e:
            logger.warning(f"Langfuse callback error: {e}")
            return None

    def load_chunks(self) -> list[Document]:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        documents = []
        for chunk in chunks:
            dieu = chunk["dieu"]
            if dieu not in self._parent_map:
                self._parent_map[dieu] = chunk.get("parent_content", chunk["noi_dung"])

            doc = Document(
                page_content=chunk["noi_dung"],
                metadata={
                    "dieu": dieu,
                    "khoan": chunk.get("khoan"),
                    "tieu_de": chunk["tieu_de"],
                    "chuong": chunk["chuong"],
                    "chuong_tieu_de": chunk["chuong_tieu_de"],
                    "source": chunk.get("source", f"Chương {chunk['chuong']} - Điều {dieu}"),
                },
            )
            documents.append(doc)
        return documents

    def get_parent_content(self, dieu: int) -> str:
        if not self._parent_map:
            self.load_chunks()
        return self._parent_map.get(dieu, "")

    def expand_to_parents(self, docs: list[Document]) -> list[Document]:
        seen_dieu = set()
        parent_docs = []
        for doc in docs:
            dieu = doc.metadata.get("dieu")
            if dieu in seen_dieu:
                continue
            seen_dieu.add(dieu)
            parent_content = self.get_parent_content(dieu) or doc.page_content
            parent_docs.append(Document(page_content=parent_content, metadata=doc.metadata.copy()))
        return parent_docs

    @staticmethod
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_vectorstore(self) -> QdrantVectorStore:
        logger.info("Building Qdrant vector store...")
        documents = self.load_chunks()
        embeddings = self.get_embeddings()

        if os.path.exists(QDRANT_DIR):
            import shutil
            shutil.rmtree(QDRANT_DIR)

        client = QdrantClient(path=QDRANT_DIR)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        self._vectorstore = QdrantVectorStore(
            client=client, collection_name=COLLECTION_NAME, embedding=embeddings,
        )
        self._vectorstore.add_documents(documents)
        logger.info(f"Vector store built: {len(documents)} docs")
        return self._vectorstore

    def load_vectorstore(self) -> QdrantVectorStore:
        if self._vectorstore is None:
            embeddings = self.get_embeddings()
            client = QdrantClient(path=QDRANT_DIR)
            self._vectorstore = QdrantVectorStore(
                client=client, collection_name=COLLECTION_NAME, embedding=embeddings,
            )
        return self._vectorstore

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        try:
            from underthesea import word_tokenize
            return word_tokenize(text.lower(), format="text").split()
        except Exception:
            return text.lower().split()

    def get_bm25_index(self):
        if self._bm25_index is None:
            logger.info("Building BM25 index...")
            self._bm25_documents = self.load_chunks()
            corpus = [self._tokenize(doc.page_content) for doc in self._bm25_documents]
            self._bm25_index = BM25Okapi(corpus)
            logger.info(f"BM25 index built: {len(self._bm25_documents)} docs")
        return self._bm25_index, self._bm25_documents

    def bm25_search(self, query_text: str, top_k: int = TOP_K_RETRIEVE) -> list[Document]:
        bm25, documents = self.get_bm25_index()
        scores = bm25.get_scores(self._tokenize(query_text))
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def hybrid_search(self, query_text: str, top_k: int = TOP_K_RETRIEVE) -> list[Document]:
        vectorstore = self.load_vectorstore()
        bm25_results = self.bm25_search(query_text, top_k=top_k)
        semantic_results = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k},
        ).invoke(query_text)

        rrf_scores = defaultdict(float)
        doc_map = {}

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.metadata.get("source", str(rank))
            rrf_scores[doc_id] += BM25_WEIGHT / (RRF_K + rank + 1)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(semantic_results):
            doc_id = doc.metadata.get("source", str(rank))
            rrf_scores[doc_id] += SEMANTIC_WEIGHT / (RRF_K + rank + 1)
            doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[:top_k]]

    def get_reranker(self):
        if self._reranker is None:
            logger.info(f"Loading reranker (device={DEVICE})...")
            self._reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
            logger.info("Reranker loaded.")
        return self._reranker

    def rerank_documents(self, query_text: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return documents

        reranker = self.get_reranker()
        pairs = [(query_text, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)

        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        logger.info(f"Reranking score for '{query_text[:50]}'")
        for score, doc in scored_docs:
            kept = "[KEPT]" if float(score) >= RERANK_SCORE_THRESHOLD else "[DROP]"
            logger.info(f"{kept} {score:+.2f} {doc.metadata.get('source', '?')}")

        result = [doc for score, doc in scored_docs if float(score) >= RERANK_SCORE_THRESHOLD][:TOP_K_RERANK]
        return result if result else [scored_docs[0][1]]

    def _init_llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
                temperature=0.1, num_ctx=4096,
            )
            self._prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT), ("human", "{question}"),
            ])
            self._condense_prompt = ChatPromptTemplate.from_template(CONDENSE_PROMPT)

    def condense_question(self, question: str, chat_history: list[dict]) -> str:
        if not chat_history or not any(ref in question.lower().strip() for ref in self._reference_words):
            return question
        self._init_llm()
        chain = self._condense_prompt | self._llm | StrOutputParser()
        return chain.invoke({
            "chat_history": self._format_chat_history(chat_history),
            "question": question,
        }).strip()

    @staticmethod
    def _format_chat_history(history: list[dict]) -> str:
        if not history:
            return "Không có lịch sử."
        return "\n".join(
            f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content']}"
            for m in history
        )

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def extract_sources(self, docs: list[Document]) -> list[dict]:
        sources, seen = [], set()
        for doc in docs:
            key = doc.metadata.get("source")
            if key in seen:
                continue
            seen.add(key)
            dieu = doc.metadata.get("dieu")
            sources.append({
                "dieu": dieu, "khoan": doc.metadata.get("khoan"),
                "tieu_de": doc.metadata.get("tieu_de"), "chuong": doc.metadata.get("chuong"),
                "source": key, "full_text": self.get_parent_content(dieu) if dieu else doc.page_content,
            })
        return sources

    def initialize(self):
        logger.info("Initializing RAG Engine...")
        device_spec = f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (CPU)"
        logger.info(f"Device: {DEVICE}{device_spec}")
        self.load_vectorstore()
        self.get_bm25_index()
        self.get_reranker()
        self._init_llm()
        logger.info("RAG Engine ready.")

    def query(self, question: str, chat_history: list[dict] = None) -> dict:
        if chat_history is None:
            chat_history = []

        self._init_llm()

        standalone = self.condense_question(question, chat_history)
        retrieved = self.hybrid_search(standalone)
        reranked = self.rerank_documents(standalone, retrieved)
        parents = self.expand_to_parents(reranked)
        context = self._format_docs(parents)

        cb = self._lf_callback(question=standalone)
        chain = self._prompt | self._llm | StrOutputParser()
        answer = chain.invoke(
            {"context": context, "question": standalone},
            config={"callbacks": [cb] if cb else []},
        )

        return {
            "answer": answer,
            "sources": self.extract_sources(reranked),
            "contexts": [doc.page_content for doc in reranked],
            "retrieved_sources": [doc.metadata.get("source", "") for doc in reranked],
        }

    def query_stream(self, question: str, chat_history: list[dict] = None):
        if chat_history is None:
            chat_history = []

        self._init_llm()

        standalone = self.condense_question(question, chat_history)
        retrieved = self.hybrid_search(standalone)
        reranked = self.rerank_documents(standalone, retrieved)
        parents = self.expand_to_parents(reranked)
        context = self._format_docs(parents)
        sources = self.extract_sources(reranked)

        cb = self._lf_callback(question=standalone)
        chain = self._prompt | self._llm | StrOutputParser()

        def stream_gen():
            for chunk in chain.stream(
                {"context": context, "question": standalone},
                config={"callbacks": [cb] if cb else []},
            ):
                yield chunk

        return stream_gen(), sources
