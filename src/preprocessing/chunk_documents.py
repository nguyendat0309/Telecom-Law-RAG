"""
Chunking dữ liệu Luật Viễn thông — Parent-Child strategy.
"""

import json
import os
import re
from src.core.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "processed", "telecom_rule_final.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed", "chunks.json")

KHOAN_PATTERN = re.compile(r"^(\d+)\.\s+", re.MULTILINE)


def parse_chapters_and_articles(content: str) -> list[dict]:
    chapter_pattern = re.compile(
        r"Chương\s+([IVXLCDM]+)\s*\n([A-ZÀ-Ỹ\s,]+?)(?=\n)"
    )
    chapter_map = []
    for m in chapter_pattern.finditer(content):
        chapter_map.append((m.start(), m.group(1), m.group(2).strip()))

    article_pattern = re.compile(r"Điều\s+(\d+)\.\s+(.+)")
    article_starts = []
    for m in article_pattern.finditer(content):
        article_starts.append((m.start(), int(m.group(1)), m.group(2).strip()))

    chunks = []

    for i, (start, article_num, article_title) in enumerate(article_starts):
        end = article_starts[i + 1][0] if i + 1 < len(article_starts) else len(content)
        article_full = content[start:end].strip()

        lines = article_full.split("\n", 1)
        article_header = lines[0].strip()
        article_body = lines[1].strip() if len(lines) > 1 else ""

        parent_content = article_full

        current_chapter = ""
        current_chapter_title = ""
        for ch_start, ch_num, ch_title in chapter_map:
            if ch_start <= start:
                current_chapter = ch_num
                current_chapter_title = ch_title
            else:
                break

        matches = list(KHOAN_PATTERN.finditer(article_body))

        if not matches:
            chunks.append({
                "dieu": article_num,
                "khoan": None,
                "tieu_de": article_title,
                "chuong": current_chapter,
                "chuong_tieu_de": current_chapter_title,
                "noi_dung": article_full,
                "parent_content": parent_content,
                "source": f"Chương {current_chapter} - Điều {article_num}",
            })
        else:
            for j, m in enumerate(matches):
                khoan_num = int(m.group(1))
                k_start = m.start()
                k_end = matches[j + 1].start() if j + 1 < len(matches) else len(article_body)
                khoan_text = article_body[k_start:k_end].strip()

                child_content = f"{article_header}\n{khoan_text}"

                if j == 0:
                    preamble = article_body[:matches[0].start()].strip()
                    if preamble:
                        child_content = f"{article_header}\n{preamble}\n{khoan_text}"

                chunks.append({
                    "dieu": article_num,
                    "khoan": khoan_num,
                    "tieu_de": article_title,
                    "chuong": current_chapter,
                    "chuong_tieu_de": current_chapter_title,
                    "noi_dung": child_content,
                    "parent_content": parent_content,
                    "source": f"Chương {current_chapter} - Điều {article_num} - Khoản {khoan_num}",
                })

    return chunks


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = parse_chapters_and_articles(content)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    dieu_only = sum(1 for c in chunks if c["khoan"] is None)
    khoan_chunks = len(chunks) - dieu_only
    logger.info("Chunking hoàn tất (Parent-Child strategy)")
    logger.info(f"Tổng chunks: {len(chunks)}")
    logger.info(f"Điều đơn (không có Khoản): {dieu_only}")
    logger.info(f"Chunks theo Khoản: {khoan_chunks}")
    logger.info(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
