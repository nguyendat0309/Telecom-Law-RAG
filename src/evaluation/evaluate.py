"""
Evaluation Pipeline cho RAG Chatbot Luật Viễn thông.
Chạy 2 phase riêng biệt:
  Phase 1 (--phase 1): Chạy RAG → lưu answers + custom metrics → eval_answers.json
  Phase 2 (--phase 2): Chạy RAGAS trên eval_answers.json → evaluation_report.json
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_FILE = os.path.join(DATA_DIR, "evaluations", "test_questions.json")

sys.path.insert(0, PROJECT_ROOT)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "qwen3:8b"

from src.core.logger import get_logger
logger = get_logger(__name__)


def load_test_data() -> list[dict]:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def init_rag_engine():
    from src.core.rag_engine import RAGEngine
    engine = RAGEngine()
    engine.initialize()
    return engine


def run_rag_query(engine, question: str) -> dict:
    result = engine.query(question)
    return {
        "answer": result["answer"],
        "contexts": result["contexts"],
        "retrieved_sources": result["retrieved_sources"],
    }


def compute_hit_rate(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    if not expected_sources:
        return -1.0
    hit = any(any(exp in ret for ret in retrieved_sources) for exp in expected_sources)
    return 1.0 if hit else 0.0


def compute_rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    except ImportError:
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words or not pred_words:
            return 0.0
        overlap = pred_words & ref_words
        p = len(overlap) / len(pred_words)
        r = len(overlap) / len(ref_words)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run_phase1(engine_type: str) -> list[dict]:
    logger.info("PHASE 1: Chạy RAG Pipeline + Custom Metrics")
    test_data = load_test_data()
    logger.info(f"Test data: {len(test_data)} câu hỏi")

    engine = init_rag_engine()
    eval_results = []

    for i, tc in enumerate(test_data):
        question = tc["question"]
        ground_truth = tc["ground_truth"]
        expected_sources = tc.get("expected_sources", [])
        category = tc.get("category", "unknown")

        start = time.time()
        result = run_rag_query(engine, question)
        elapsed = time.time() - start

        hr = compute_hit_rate(result["retrieved_sources"], expected_sources)
        rouge = compute_rouge_l(result["answer"], ground_truth)

        status = 'HIT' if hr == 1.0 else 'MISS' if hr == 0.0 else 'N/A'
        logger.info(f"[{i+1}/{len(test_data)}] {question[:60]}... -> {status} ROUGE={rouge:.2f} ({elapsed:.1f}s)")

        eval_results.append({
            "id": tc.get("id", i + 1),
            "question": question,
            "ground_truth": ground_truth,
            "answer": result["answer"],
            "contexts": result["contexts"],
            "category": category,
            "expected_sources": expected_sources,
            "retrieved_sources": result["retrieved_sources"],
            "hit_rate": hr if hr >= 0 else None,
            "rouge_l": round(rouge, 3),
            "time_seconds": round(elapsed, 2),
        })

    answers_file = os.path.join(DATA_DIR, f"eval_answers_{engine_type}.json")
    with open(answers_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    hit_rates = [r["hit_rate"] for r in eval_results if r["hit_rate"] is not None]
    rouge_scores = [r["rouge_l"] for r in eval_results]

    logger.info("PHASE 1 KẾT QUẢ:")
    logger.info(f"Hit Rate: {avg(hit_rates):.1%}")
    logger.info(f"ROUGE-L: {avg(rouge_scores):.3f}")
    logger.info(f"Avg Time: {avg([r['time_seconds'] for r in eval_results]):.1f}s")
    logger.info(f"Đã lưu: {answers_file}")

    return eval_results


def run_phase2(engine_type: str) -> dict:
    answers_file = os.path.join(DATA_DIR, f"eval_answers_{engine_type}.json")

    if not os.path.exists(answers_file):
        logger.error(f"Chưa có {answers_file}! Chạy Phase 1 trước.")
        return None

    with open(answers_file, "r", encoding="utf-8") as f:
        eval_results = json.load(f)

    logger.info(f"PHASE 2: RAGAS Evaluation on {len(eval_results)} questions from {answers_file}")

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        )
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Thiếu package: {e} - Run: pip install ragas datasets")
        return None

    ragas_llm = None
    ragas_embeddings = None
    judge_name = "unknown"

    if GEMINI_API_KEY:
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_community.embeddings import HuggingFaceEmbeddings

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
            )
            ragas_llm = LangchainLLMWrapper(llm)

            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-base",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
            judge_name = "Gemini (gemini-2.5-flash-lite)"
            logger.info(f"Judge LLM: {judge_name}")
        except ImportError:
            logger.warning("langchain-google-genai chưa cài, thử Ollama...")

    if ragas_llm is None:
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_ollama import ChatOllama
            from langchain_community.embeddings import HuggingFaceEmbeddings

            llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
            ragas_llm = LangchainLLMWrapper(llm)

            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-base",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
            judge_name = f"Ollama ({LLM_MODEL})"
            logger.info(f"Judge LLM: {judge_name}")
        except Exception as e:
            logger.error(f"Không thể khởi tạo judge LLM: {e}")
            return None

    ragas_data = {
        "question": [r["question"] for r in eval_results],
        "answer": [r["answer"] for r in eval_results],
        "contexts": [r["contexts"] for r in eval_results],
        "ground_truth": [r["ground_truth"] for r in eval_results],
    }
    dataset = Dataset.from_dict(ragas_data)

    logger.info(f"Đang chạy RAGAS ({len(eval_results)} câu)...")

    try:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        def safe_avg(val):
            if isinstance(val, list):
                valid = [v for v in val if v is not None and not (isinstance(v, float) and v != v)]
                return round(sum(valid) / len(valid), 3) if valid else 0.0
            return round(float(val), 3)

        ragas_scores = {
            "faithfulness": safe_avg(result["faithfulness"]),
            "answer_relevancy": safe_avg(result["answer_relevancy"]),
            "context_precision": safe_avg(result["context_precision"]),
            "context_recall": safe_avg(result["context_recall"]),
            "judge_llm": judge_name,
        }

        logger.info("RAGAS hoàn tất!")
        logger.info(f"Faithfulness: {ragas_scores['faithfulness']:.3f}")
        logger.info(f"Answer Relevancy: {ragas_scores['answer_relevancy']:.3f}")
        logger.info(f"Context Precision: {ragas_scores['context_precision']:.3f}")
        logger.info(f"Context Recall: {ragas_scores['context_recall']:.3f}")

        return ragas_scores

    except Exception as e:
        logger.error(f"RAGAS error: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_report(engine_type: str, ragas_scores: dict = None):
    answers_file = os.path.join(DATA_DIR, f"eval_answers_{engine_type}.json")
    with open(answers_file, "r", encoding="utf-8") as f:
        eval_results = json.load(f)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    hit_rates = [r["hit_rate"] for r in eval_results if r["hit_rate"] is not None]
    rouge_scores = [r["rouge_l"] for r in eval_results]

    report = {
        "meta": {
            "engine": engine_type,
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(eval_results),
            "llm_model": LLM_MODEL,
        },
        "summary": {
            "custom_metrics": {
                "hit_rate": round(avg(hit_rates), 3),
                "rouge_l": round(avg(rouge_scores), 3),
                "avg_time_seconds": round(avg([r["time_seconds"] for r in eval_results]), 2),
            },
            "ragas": ragas_scores,
        },
        "by_category": {},
        "details": eval_results,
    }

    cats = {}
    for r in eval_results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"rouge_l": [], "hit_rate": []}
        cats[cat]["rouge_l"].append(r["rouge_l"])
        if r["hit_rate"] is not None:
            cats[cat]["hit_rate"].append(r["hit_rate"])

    for cat, m in cats.items():
        report["by_category"][cat] = {
            "count": len(m["rouge_l"]),
            "rouge_l": round(avg(m["rouge_l"]), 3),
            "hit_rate": round(avg(m["hit_rate"]), 3) if m["hit_rate"] else None,
        }

    report_file = os.path.join(DATA_DIR, f"evaluation_report_{engine_type}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    s = report["summary"]
    logger.info("BÁO CÁO TỔNG HỢP")
    logger.info("Custom Metrics:")
    logger.info(f"Hit Rate: {s['custom_metrics']['hit_rate']:.1%}")
    logger.info(f"ROUGE-L: {s['custom_metrics']['rouge_l']:.3f}")
    logger.info(f"Avg Time: {s['custom_metrics']['avg_time_seconds']:.1f}s")

    if ragas_scores:
        logger.info(f"RAGAS Scores (judge: {ragas_scores.get('judge_llm', '?')}):")
        logger.info(f"Faithfulness: {ragas_scores['faithfulness']:.3f}")
        logger.info(f"Answer Relevancy: {ragas_scores['answer_relevancy']:.3f}")
        logger.info(f"Context Precision: {ragas_scores['context_precision']:.3f}")
        logger.info(f"Context Recall: {ragas_scores['context_recall']:.3f}")

    logger.info("Phân loại:")
    for cat, info in report["by_category"].items():
        hr_str = f", Hit={info['hit_rate']:.1%}" if info['hit_rate'] is not None else ""
        logger.info(f"{cat} ({info['count']}): ROUGE={info['rouge_l']:.3f}{hr_str}")

    logger.info(f"Report: {report_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG Chatbot")
    parser.add_argument("--engine", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="1=RAG only, 2=RAGAS only, default=both")
    args = parser.parse_args()

    ragas_scores = None

    if args.phase is None or args.phase == 1:
        run_phase1(args.engine)

    if args.phase is None or args.phase == 2:
        ragas_scores = run_phase2(args.engine)

    answers_file = os.path.join(DATA_DIR, f"eval_answers_{args.engine}.json")
    if os.path.exists(answers_file):
        build_report(args.engine, ragas_scores)
