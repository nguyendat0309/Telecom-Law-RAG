"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' hoặc 'assistant'")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi về Luật Viễn thông")
    history: list[ChatMessage] = Field(default=[], description="Lịch sử hội thoại")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Viễn thông là gì?",
                    "history": [],
                }
            ]
        }
    }


class SourceInfo(BaseModel):
    source: str = Field(..., description="Nguồn: Chương X - Điều Y - Khoản Z")
    dieu: int | None = None
    khoan: int | str | None = None
    tieu_de: str | None = None
    chuong: str | None = None


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời")
    sources: list[SourceInfo] = Field(default=[], description="Nguồn trích dẫn")
    time_seconds: float = Field(..., description="Thời gian xử lý (giây)")


class HealthResponse(BaseModel):
    status: str
    engine: str
    device: str
    ollama: str
    vectorstore: str
