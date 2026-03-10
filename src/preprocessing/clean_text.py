"""
Script clean dữ liệu Luật Viễn thông cho RAG Chatbot.
"""

import re
import os
from src.core.logger import get_logger

logger = get_logger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "telecom_rule_extracted.txt")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "telecom_rule_cleaned.txt")


def clean_text(content: str) -> str:
    content = content.replace("\x07", "")
    content = content.replace("\xa0", " ")

    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    content = "\n".join(cleaned_lines)
    content = re.sub(r" {2,}", " ", content)

    return content


def report_stats(original: str, cleaned: str):
    logger.info("KẾT QUẢ CLEAN DỮ LIỆU LUẬT VIỄN THÔNG")

    bel_count = original.count("\x07")
    nbsp_count = original.count("\xa0")

    logger.info(f"File gốc:  {INPUT_FILE} -> Kích thước: {len(original):,} ký tự, Số dòng: {len(original.splitlines()):,}, BEL: {bel_count}, NBSP: {nbsp_count}")
    
    bel_after = cleaned.count("\x07")
    nbsp_after = cleaned.count("\xa0")

    logger.info(f"File clean: {OUTPUT_FILE} -> Kích thước: {len(cleaned):,} ký tự, Số dòng: {len(cleaned.splitlines()):,}, BEL: {bel_after}, NBSP: {nbsp_after}")

    articles = sorted(set(int(m) for m in re.findall(r"Điều (\d+)", cleaned)))
    logger.info(f"Cấu trúc -> Số Điều: {len(articles)}")

    chapters = re.findall(r"Chương ([IVXLCDM]+)", cleaned)
    logger.info(f"Cấu trúc -> Số Chương: {len(chapters)} ({', '.join(chapters)})")
    logger.info("Clean dữ liệu hoàn tất.")


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        original = f.read()

    cleaned = clean_text(original)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(cleaned)

    report_stats(original, cleaned)


if __name__ == "__main__":
    main()
