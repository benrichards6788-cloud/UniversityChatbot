import json
import time
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_utilization

from query_policies import retrieve_policies
from answer_policies_llama import answer_question

# Top-k values to test
K_VALUES = [3, 5, 8, 10, 15]

TEST_QUESTIONS = [
    "What should I do if I miss an exam due to illness?",
    "How long does the university have to return marked work?",
    "What is the policy on anonymous marking?",
    "Can I appeal my exam results and if so how?",
    "What counts as acceptable evidence for personal circumstances?",
    "What is the university policy on academic misconduct?",
    "How do I apply for an extension on a coursework deadline?",
    "What support is available for students with disabilities?",
    "What are the rules around late submission of coursework?",
    "How does the university handle complaints from students?",
    "What is the minimum pass mark for a module?",
    "Can I resit a failed exam and what are the conditions?",
    "What happens if I am suspected of plagiarism?",
    "How do I request a deferral of assessments?",
    "What is the policy on recording lectures?",
    "Who do I contact if I have a problem with my supervisor?",
    "What are the attendance requirements for students?",
    "How are degree classifications calculated at Strathclyde?",
]

METRICS = [faithfulness, answer_relevancy, context_utilization]
RETRIEVAL_METHOD = "hybrid"
SLEEP_SECONDS = 1
OUTPUT_PATH = Path("ragas_topk_results.json")


def score_one(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Score a single QA instance with RAGAS.
    Uses a fresh event loop to avoid asyncio issues seen with older RAGAS versions.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=METRICS)

    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_utilization": float(result["context_utilization"]),
    }


def answer_question_with_chunks(question: str, k: int) -> tuple[str, list[dict], list[str]]:
    chunks = retrieve_policies(question, k=k, retrieval_method=RETRIEVAL_METHOD)
    contexts = [c["text"] for c in chunks]
    answer = answer_question(question, k=k)
    return answer, chunks, contexts


def compute_averages(scores: list[dict]) -> dict:
    if not scores:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_utilization": None,
            "num_questions": 0,
        }

    return {
        "faithfulness": sum(s["faithfulness"] for s in scores) / len(scores),
        "answer_relevancy": sum(s["answer_relevancy"] for s in scores) / len(scores),
        "context_utilization": sum(s["context_utilization"] for s in scores) / len(scores),
        "num_questions": len(scores),
    }


def print_summary_table(summary: dict[int, dict]) -> None:
    print("\n=== TOP-K SUMMARY TABLE ===")
    print(
        f"{'k':<6}"
        f"{'faithfulness':<16}"
        f"{'answer_relevancy':<20}"
        f"{'context_utilization':<22}"
        f"{'n':<6}"
    )
    print("-" * 70)

    for k in K_VALUES:
        avg = summary.get(k, {})
        f = avg.get("faithfulness")
        a = avg.get("answer_relevancy")
        c = avg.get("context_utilization")
        n = avg.get("num_questions", 0)

        f_str = f"{f:.3f}" if isinstance(f, (int, float)) else "N/A"
        a_str = f"{a:.3f}" if isinstance(a, (int, float)) else "N/A"
        c_str = f"{c:.3f}" if isinstance(c, (int, float)) else "N/A"

        print(f"{k:<6}{f_str:<16}{a_str:<20}{c_str:<22}{n:<6}")


def main() -> None:
    print("=== Top-k RAGAS Evaluation — Strathclyde Policy Assistant ===")
    print(f"Retrieval method fixed to: {RETRIEVAL_METHOD}")
    print(f"Testing k values: {K_VALUES}\n")

    overall_output = {
        "retrieval_method": RETRIEVAL_METHOD,
        "k_values": K_VALUES,
        "questions": TEST_QUESTIONS,
        "results_by_k": {},
    }

    summary = {}

    for k in K_VALUES:
        print(f"\n{'=' * 80}")
        print(f"Evaluating top-k = {k}")
        print(f"{'=' * 80}")

        all_scores = []
        per_question_results = []

        for i, question in enumerate(TEST_QUESTIONS, start=1):
            print(f"[k={k}] [{i}/{len(TEST_QUESTIONS)}] {question}")

            try:
                answer, chunks, contexts = answer_question_with_chunks(question, k=k)
                print(f"  Retrieved {len(chunks)} chunks, answer length: {len(answer)} chars")
            except Exception as e:
                print(f"  PIPELINE ERROR: {e}")
                per_question_results.append({
                    "question": question,
                    "error": f"PIPELINE ERROR: {str(e)}",
                })
                continue

            try:
                scores = score_one(question, answer, contexts)
                all_scores.append(scores)

                per_question_results.append({
                    "question": question,
                    "k": k,
                    "answer": answer,
                    "num_contexts": len(contexts),
                    "contexts": contexts,
                    "scores": scores,
                })

                print(
                    f"  Faithfulness: {scores['faithfulness']:.3f} | "
                    f"Answer Relevancy: {scores['answer_relevancy']:.3f} | "
                    f"Context Utilization: {scores['context_utilization']:.3f}"
                )
            except Exception as e:
                print(f"  RAGAS ERROR: {e}")
                per_question_results.append({
                    "question": question,
                    "k": k,
                    "answer": answer,
                    "num_contexts": len(contexts),
                    "contexts": contexts,
                    "error": f"RAGAS ERROR: {str(e)}",
                })
                continue

            time.sleep(SLEEP_SECONDS)

        averages = compute_averages(all_scores)
        summary[k] = averages

        print(f"\n=== AVERAGES FOR k = {k} ===")
        print(f"  Faithfulness:        {averages['faithfulness']:.3f}" if averages["faithfulness"] is not None else "  Faithfulness:        N/A")
        print(f"  Answer Relevancy:    {averages['answer_relevancy']:.3f}" if averages["answer_relevancy"] is not None else "  Answer Relevancy:    N/A")
        print(f"  Context Utilization: {averages['context_utilization']:.3f}" if averages["context_utilization"] is not None else "  Context Utilization: N/A")
        print(f"  Questions scored:    {averages['num_questions']}")

        overall_output["results_by_k"][str(k)] = {
            "averages": averages,
            "per_question": per_question_results,
        }

        OUTPUT_PATH.write_text(json.dumps(overall_output, indent=2), encoding="utf-8")
        print(f"\nIntermediate results saved to {OUTPUT_PATH}")

    print_summary_table(summary)

    # Pick a simple "best" by highest faithfulness, then answer relevancy, then context utilization
    valid = [
        (k, vals) for k, vals in summary.items()
        if vals["faithfulness"] is not None
    ]
    if valid:
        best_k, best_vals = sorted(
            valid,
            key=lambda item: (
                item[1]["faithfulness"],
                item[1]["answer_relevancy"],
                item[1]["context_utilization"],
            ),
            reverse=True,
        )[0]

        overall_output["best_k_by_priority"] = {
            "priority_order": [
                "faithfulness",
                "answer_relevancy",
                "context_utilization",
            ],
            "k": best_k,
            "scores": best_vals,
        }

        OUTPUT_PATH.write_text(json.dumps(overall_output, indent=2), encoding="utf-8")
        print("\n=== BEST k (priority: faithfulness > answer relevancy > context utilization) ===")
        print(f"Best k: {best_k}")
        print(
            f"Scores -> Faithfulness: {best_vals['faithfulness']:.3f}, "
            f"Answer Relevancy: {best_vals['answer_relevancy']:.3f}, "
            f"Context Utilization: {best_vals['context_utilization']:.3f}"
        )

    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()