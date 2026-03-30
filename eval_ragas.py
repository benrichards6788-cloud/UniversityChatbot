import json
import time
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_utilization

from query_policies import retrieve_policies
from answer_policies_llama import answer_question

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


def score_one(question: str, answer: str, contexts: list) -> dict:
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


def main():
    print("=== RAGAS Evaluation — Strathclyde Policy Assistant ===\n")

    all_scores = []
    results_per_question = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")

        # run RAG pipeline
        try:
            chunks = retrieve_policies(question, k=5, retrieval_method="hybrid")
            contexts = [c["text"] for c in chunks]
            answer = answer_question(question, k=5)
            print(f"  Retrieved {len(contexts)} chunks, answer: {len(answer)} chars")
        except Exception as e:
            print(f"  PIPELINE ERROR: {e}")
            continue

        # score with RAGAS (one at a time)
        try:
            scores = score_one(question, answer, contexts)
            all_scores.append(scores)
            results_per_question.append({
                "question": question,
                "answer": answer,
                "scores": scores,
            })
            print(f"  Faithfulness: {scores['faithfulness']:.3f} | "
                  f"Answer Relevancy: {scores['answer_relevancy']:.3f} | "
                  f"Context Utilization: {scores['context_utilization']:.3f}")
        except Exception as e:
            print(f"  RAGAS ERROR: {e}")
            continue

        # small delay to avoid API rate limits
        time.sleep(1)

    if not all_scores:
        print("\nNo scores collected — check errors above.")
        return

    # compute averages
    avg = {
        "faithfulness":        sum(s["faithfulness"] for s in all_scores) / len(all_scores),
        "answer_relevancy":    sum(s["answer_relevancy"] for s in all_scores) / len(all_scores),
        "context_utilization": sum(s["context_utilization"] for s in all_scores) / len(all_scores),
        "num_questions":       len(all_scores),
    }

    print("\n=== FINAL RAGAS SCORES ===")
    print(f"  Faithfulness:        {avg['faithfulness']:.3f}")
    print(f"  Answer Relevancy:    {avg['answer_relevancy']:.3f}")
    print(f"  Context Utilization: {avg['context_utilization']:.3f}")
    print(f"  Questions scored:    {avg['num_questions']}")

    # save results
    out = {
        "averages": avg,
        "per_question": results_per_question,
    }
    Path("ragas_results.json").write_text(json.dumps(out, indent=2))
    print("\nFull results saved to ragas_results.json")


if __name__ == "__main__":
    main()