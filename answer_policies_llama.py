#!/usr/bin/env python3

"""
No HTTP, no API keys needed
"""

import textwrap
import ollama  
from query_policies import retrieve_policies  

MODEL_NAME = "llama3.1"  

def call_llama(system_prompt: str, user_prompt: str) -> str:
    full_prompt = f"""SYSTEM:
{system_prompt}

USER QUESTION AND CONTEXT:
{user_prompt}
"""

    # ollama.generate returns a dict like: {"model": ..., "response": "...", ...}
    response = ollama.generate(model=MODEL_NAME, prompt=full_prompt)
    return response["response"].strip()


#formats retrieved chunks into readable context block for model
def build_context(chunks, max_chars: int = 5000) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[Source {i} | {c.get('doc_title')} | {c.get('section')}]"
        body = c["text"]
        blocks.append(f"{header}\n{body}")

    context = "\n\n".join(blocks)
    return context[:max_chars]


def answer_question(question: str, k: int = 10) -> str:
    """
    Full RAG step:
    1. Retrieve relevant policy chunks using FAISS.
    2. Build a prompt.
    3. Ask Llama for an answer grounded in those chunks.
    """
    chunks = retrieve_policies(question, k=k)
    if not chunks:
        return "I couldn't find any relevant policy sections to answer that."

    context = build_context(chunks)

    system_prompt = (
    "You are an assistant for University of Strathclyde students and staff. "
    "You answer questions using ONLY the provided sources. "
    "The sources may include formal policy documents, official University guidance, "
    "and curated FAQs derived from authoritative Strathclyde webpages. "
    "Treat official guidance and curated FAQs as valid and reliable sources of information. "

    "If the retrieved sources clearly answer the question, give a direct and helpful answer first. "
    "Only say you are unsure if no relevant source has been retrieved. "

    "Do not guess or add information that is not supported by the sources. "
    "Do not invent contacts or procedures unless they are explicitly mentioned in the sources. "

    "When explaining your answer, refer to the sources as '[1] : [Section Name]', '[2]: [Section Name]', etc."
    )

    user_prompt = textwrap.dedent(f"""
    Question:
    {question}

    Policy extracts:
    {context}
    """)

    return call_llama(system_prompt, user_prompt)


def main():
    print("Ask a policy question (or 'exit'):\n")
    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        ans = answer_question(q, k=10)
        print("\n=== ANSWER ===")
        print(ans)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
