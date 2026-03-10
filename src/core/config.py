"""
Centralized configuration for RAG Chatbot.
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "processed", "chunks.json")
QDRANT_DIR = os.path.join(DATA_DIR, "qdrant_db")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = "qwen3:8b"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

COLLECTION_NAME = "telecom_law"
BM25_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5
RRF_K = 60
TOP_K_RETRIEVE = 8
TOP_K_RERANK = 5
RERANK_SCORE_THRESHOLD = -3.0

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

SYSTEM_PROMPT = """Bạn là trợ lý pháp luật về Luật Viễn thông Việt Nam (Luật số 24/2023/QH15).

Nhiệm vụ: Trả lời ĐÚNG TRỌNG TÂM câu hỏi, chỉ dựa vào context bên dưới.

Quy tắc bắt buộc:
1. CHỈ trả lời những gì được hỏi, không giải thích thêm.
2. TRÍCH DẪN bằng format "**Điều X Khoản Y:**" in đậm trước mỗi ý.
3. Nếu có nhiều ý, dùng danh sách đánh số (1. 2. 3.) với mỗi ý trên một dòng riêng.
4. Nếu context KHÔNG chứa thông tin, chỉ viết: "Không có quy định về vấn đề này trong Luật Viễn thông."
5. KHÔNG thêm nhận xét, đánh giá, kết luận chung chung.
6. KHÔNG viết lại nội dung có trong context theo cách dài hơn.
7. Luôn xuống dòng giữa các ý, KHÔNG viết liền nhau.

Context:
{context}

/no_think"""

CONDENSE_PROMPT = """Viết lại câu hỏi mới thành câu hỏi độc lập dựa trên lịch sử hội thoại.
Chỉ trả về câu hỏi, không giải thích.

Lịch sử:
{chat_history}

Câu hỏi mới: {question}

Câu hỏi độc lập: /no_think"""
