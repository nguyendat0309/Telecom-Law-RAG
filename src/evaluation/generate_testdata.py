"""
Sinh dữ liệu đánh giá (test questions) cho RAG Chatbot Luật Viễn thông.
Dùng Google Gemini API để sinh câu hỏi chất lượng cao.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "processed", "chunks.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "evaluations", "test_questions.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

from src.core.logger import get_logger
logger = get_logger(__name__)

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
_current_model_idx = 0


def load_chunks_by_dieu() -> dict:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    dieu_map = {}
    for chunk in chunks:
        dieu = chunk["dieu"]
        if dieu not in dieu_map:
            dieu_map[dieu] = {
                "dieu": dieu,
                "tieu_de": chunk["tieu_de"],
                "chuong": chunk["chuong"],
                "chuong_tieu_de": chunk["chuong_tieu_de"],
                "parent_content": chunk.get("parent_content", chunk["noi_dung"]),
                "khoans": [],
                "sources": [],
            }
        if chunk.get("khoan"):
            dieu_map[dieu]["khoans"].append(chunk["khoan"])
            dieu_map[dieu]["sources"].append(
                f"Chương {chunk['chuong']} - Điều {dieu} - Khoản {chunk['khoan']}"
            )
        else:
            dieu_map[dieu]["sources"].append(
                f"Chương {chunk['chuong']} - Điều {dieu}"
            )
    return dieu_map


def call_gemini(prompt: str) -> str:
    global _current_model_idx
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    tried = 0

    while tried < len(GEMINI_MODELS):
        model_name = GEMINI_MODELS[_current_model_idx]

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=2000,
                ),
            )
            return response.text.strip()

        except Exception as e:
            err = str(e)
            if "429" in err or "404" in err:
                old_model = model_name
                _current_model_idx = (_current_model_idx + 1) % len(GEMINI_MODELS)
                tried += 1
                if tried < len(GEMINI_MODELS):
                    reason = "hết quota" if "429" in err else "không tìm thấy"
                    logger.warning(f"{old_model} {reason} -> thử {GEMINI_MODELS[_current_model_idx]}...")
                    continue
                else:
                    logger.error("Tất cả model đều lỗi!")
                    raise
            raise

    return ""


def generate_questions_for_dieu(dieu_info: dict) -> list[dict]:
    content = dieu_info["parent_content"]
    tieu_de = dieu_info["tieu_de"]
    dieu_num = dieu_info["dieu"]
    chuong = dieu_info["chuong"]
    num_khoans = len(dieu_info["khoans"])

    if num_khoans >= 3:
        category = "multi_clause"
        num_q = 2
    else:
        category = "single_lookup"
        num_q = 1

    prompt = f"""Bạn là chuyên gia pháp luật viễn thông Việt Nam. Dựa vào nội dung điều luật dưới đây, hãy tạo {num_q} câu hỏi để kiểm tra chatbot.

YÊU CẦU NGHIÊM NGẶT:
- Câu hỏi phải TỰ NHIÊN, giống người dùng thật sẽ hỏi
- TUYỆT ĐỐI KHÔNG nhắc đến số Điều/Khoản trong câu hỏi
- Câu hỏi phải cụ thể, có thể trả lời được từ nội dung
- Câu trả lời mẫu (ground_truth) phải NGẮN GỌN, CHÍNH XÁC, trích dẫn trực tiếp từ nội dung luật
- ground_truth KHÔNG được paraphrase, phải giữ nguyên câu từ gốc trong luật

Nội dung Điều {dieu_num} - {tieu_de}:
---
{content[:3000]}
---

Trả lời ĐÚNG format JSON array (KHÔNG thêm gì khác, KHÔNG markdown code block):
[
  {{
    "question": "câu hỏi tự nhiên bằng tiếng Việt",
    "ground_truth": "câu trả lời ngắn gọn, trích dẫn đúng từ nội dung luật"
  }}
]"""

    try:
        response = call_gemini(prompt)
        if not response:
            return []

        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
            questions = []
            for item in parsed[:num_q]:
                questions.append({
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "expected_sources": dieu_info["sources"][:3],
                    "category": category,
                    "difficulty": "medium" if num_khoans >= 2 else "easy",
                    "dieu": dieu_num,
                    "chuong": chuong,
                })
            return questions
    except Exception as e:
        logger.warning(f"Failed generating queries for {dieu_num}: {e}")
    return []


MANUAL_QUESTIONS = [
    {
        "question": "Viễn thông là gì?",
        "ground_truth": "Viễn thông là việc gửi, truyền, nhận và xử lý ký hiệu, tín hiệu, số liệu, chữ viết, hình ảnh, âm thanh hoặc dạng thông tin khác qua mạng viễn thông.",
        "expected_sources": ["Chương I - Điều 3 - Khoản 1"],
        "category": "single_lookup", "difficulty": "easy",
        "dieu": 3, "chuong": "I",
    },
    {
        "question": "Doanh nghiệp viễn thông cần những điều kiện gì để được cấp giấy phép?",
        "ground_truth": "Doanh nghiệp cần đáp ứng các điều kiện về vốn, nhân lực, kỹ thuật và phương án kinh doanh theo quy định.",
        "expected_sources": ["Chương IV - Điều 35 - Khoản 1", "Chương IV - Điều 35 - Khoản 2"],
        "category": "multi_clause", "difficulty": "medium",
        "dieu": 35, "chuong": "IV",
    },
    {
        "question": "Luật viễn thông quy định gì về trí tuệ nhân tạo?",
        "ground_truth": "Không có quy định về vấn đề này trong Luật Viễn thông.",
        "expected_sources": [],
        "category": "negative", "difficulty": "easy",
        "dieu": None, "chuong": None,
    },
    {
        "question": "Mạng viễn thông công cộng là gì?",
        "ground_truth": "Mạng viễn thông công cộng là mạng viễn thông do doanh nghiệp viễn thông thiết lập để cung cấp dịch vụ viễn thông cho mọi tổ chức, cá nhân.",
        "expected_sources": ["Chương I - Điều 3 - Khoản 6"],
        "category": "single_lookup", "difficulty": "easy",
        "dieu": 3, "chuong": "I",
    },
    {
        "question": "Những hành vi nào bị nghiêm cấm trong hoạt động viễn thông?",
        "ground_truth": "Các hành vi bị cấm bao gồm: lợi dụng viễn thông xâm phạm an ninh quốc gia, trật tự an toàn xã hội; trộm cắp, sử dụng trái phép tài nguyên viễn thông; cản trở hoạt động viễn thông hợp pháp.",
        "expected_sources": ["Chương I - Điều 9 - Khoản 1", "Chương I - Điều 9 - Khoản 2", "Chương I - Điều 9 - Khoản 3"],
        "category": "multi_clause", "difficulty": "medium",
        "dieu": 9, "chuong": "I",
    },
    {
        "question": "Quyền của người sử dụng dịch vụ viễn thông gồm những gì?",
        "ground_truth": "Người sử dụng có quyền: lựa chọn doanh nghiệp và dịch vụ viễn thông; được bảo đảm bí mật thông tin; được cung cấp thông tin đầy đủ về dịch vụ; khiếu nại khi quyền lợi bị xâm phạm.",
        "expected_sources": ["Chương VII - Điều 60 - Khoản 1", "Chương VII - Điều 60 - Khoản 2", "Chương VII - Điều 60 - Khoản 3"],
        "category": "multi_clause", "difficulty": "medium",
        "dieu": 60, "chuong": "VII",
    },
    {
        "question": "Quy định về giá cước dịch vụ viễn thông như thế nào?",
        "ground_truth": "Giá cước dịch vụ viễn thông được xác định theo cơ chế thị trường, trừ trường hợp Nhà nước quy định giá cước.",
        "expected_sources": ["Chương VII - Điều 56 - Khoản 1", "Chương VII - Điều 56 - Khoản 2"],
        "category": "multi_clause", "difficulty": "medium",
        "dieu": 56, "chuong": "VII",
    },
]


def select_diverse_dieu(dieu_map: dict, target: int = 25) -> list[int]:
    by_chapter = {}
    for dieu_num, info in dieu_map.items():
        ch = info["chuong"]
        if ch not in by_chapter:
            by_chapter[ch] = []
        by_chapter[ch].append(dieu_num)

    manual_dieu = {q["dieu"] for q in MANUAL_QUESTIONS if q["dieu"]}
    selected = []
    per_chapter = max(2, target // len(by_chapter))

    for chapter in sorted(by_chapter.keys()):
        available = sorted([d for d in by_chapter[chapter] if d not in manual_dieu])
        if len(available) <= per_chapter:
            selected.extend(available)
        else:
            step = len(available) / per_chapter
            for i in range(per_chapter):
                selected.append(available[min(int(i * step), len(available) - 1)])

    return selected[:target]


def main():
    logger.info("SINH DỮ LIỆU ĐÁNH GIÁ RAG (Gemini only)")

    if not GEMINI_API_KEY:
        logger.error("Chưa có GEMINI_API_KEY trong .env!")
        sys.exit(1)

    dieu_map = load_chunks_by_dieu()
    logger.info(f"Tổng: {len(dieu_map)} Điều")

    all_questions = list(MANUAL_QUESTIONS)
    logger.info(f"{len(MANUAL_QUESTIONS)} câu hỏi thủ công")

    selected = select_diverse_dieu(dieu_map, target=25)
    logger.info(f"Sinh câu hỏi cho {len(selected)} Điều...")

    llm_count = 0
    for i, dieu_num in enumerate(selected):
        info = dieu_map[dieu_num]
        
        questions = generate_questions_for_dieu(info)
        if questions:
            all_questions.extend(questions)
            llm_count += len(questions)
            logger.info(f"[{i+1}/{len(selected)}] Điều {dieu_num}: {info['tieu_de'][:50]}... -> {len(questions)} câu")
        else:
            logger.info(f"[{i+1}/{len(selected)}] Điều {dieu_num}: {info['tieu_de'][:50]}... -> Failed")

        time.sleep(4)

    for i, q in enumerate(all_questions):
        q["id"] = i + 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    cats = {}
    for q in all_questions:
        cats[q.get("category", "?")] = cats.get(q.get("category", "?"), 0) + 1

    logger.info(f"HOÀN TẤT: {len(all_questions)} câu hỏi")
    logger.info(f"Thủ công: {len(MANUAL_QUESTIONS)}, Gemini: {llm_count}")
    for cat, count in sorted(cats.items()):
        logger.info(f"- {cat}: {count}")
    logger.info(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
