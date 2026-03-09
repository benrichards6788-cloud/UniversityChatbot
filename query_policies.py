#!/usr/bin/env python3

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# configurations

VECTORSTORE_DIR = Path("vectorstore")
INDEX_PATH = VECTORSTORE_DIR / "policy.index"
META_PATH = VECTORSTORE_DIR / "meta.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DEFAULT = 5

# load resources

print("[INFO] Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("[INFO] Loading metadata...")
with META_PATH.open("r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

print(f"[INFO] Index size: {index.ntotal}")
print(f"[INFO] Metadata entries: {len(CHUNKS)}")

if index.ntotal != len(CHUNKS):
    print("[WARN] Index size and metadata length differ. "
          "Something may be wrong with your indexing step.")

print(f"[INFO] Loading embedding model: {MODEL_NAME}")
EMB = SentenceTransformer(MODEL_NAME)

# retrieval function (embed natural language query, search faiss and return top k chunks)
# returns a list of dictionaries 

def retrieve_policies(query: str, k: int = TOP_K_DEFAULT):
    """
    FAISS semantic similarity with keyword fallback
    Returns a list of dictionaries with:
        - score: FAISS similarity score
        - idx: index in CHUNKS
        - text, doc_title, section, etc.
    """
    # embed query
    q_vec = EMB.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # search FAISS
    scores, idxs = index.search(q_vec, k)
    scores = scores[0]
    idxs = idxs[0]

    results = []

    for score, idx in zip(scores, idxs):
        if idx == -1:
            continue
        chunk = CHUNKS[idx]
        results.append({
            "score": float(score),
            "idx": int(idx),
            "text": chunk.get("text", ""),
            "doc_title": chunk.get("doc_title"),
            "section": chunk.get("section"),
            **{k_extra: v_extra for k_extra, v_extra in chunk.items() if k_extra not in {"text", "doc_title", "section"}}
        })

    # fallback: if FAISS returns nothing, do simple keyword filtering
    if not results:
        query_keywords = query.lower().split()
        filtered_chunks = [
            (i, ch) for i, ch in enumerate(CHUNKS)
            if any(kw in ch["text"].lower() for kw in query_keywords)
        ]
        for idx, chunk in filtered_chunks[:k]:
            results.append({
                "score": 0.0,  # no FAISS score
                "idx": idx,
                "text": chunk.get("text", ""),
                "doc_title": chunk.get("doc_title"),
                "section": chunk.get("section"),
                **{k_extra: v_extra for k_extra, v_extra in chunk.items() if k_extra not in {"text", "doc_title", "section"}}
            })

    return results



# simple cli (for testing)

def main():
    print("\nType a question about the policies (or 'exit' to quit):\n")
    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        hits = retrieve_policies(q, k=TOP_K_DEFAULT)

        if not hits:
            print("No results found.\n")
            continue

        for i, h in enumerate(hits, start=1):
            print(f"\n=== RESULT {i} (score={h['score']:.3f}, idx={h['idx']}) ===")
            print(f"Title:   {h.get('doc_title')}")
            print(f"Section: {h.get('section')}")
            print(h["text"][:500].replace("\n", " "), "...")
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
