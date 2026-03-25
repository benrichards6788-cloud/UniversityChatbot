#!/usr/bin/env python3

"""
Hybrid retrieval module for the Strathclyde Policy Assistant.

Retrieval strategies:
    - "dense"  : FAISS cosine similarity only (all-MiniLM-L6-v2)
    - "sparse" : BM25 keyword ranking only (rank_bm25)
    - "hybrid" : Reciprocal Rank Fusion (RRF) of dense + sparse (default)

Reciprocal Rank Fusion reference:
    Cormack, G.V., Clarke, C.L.A. and Buettcher, S. (2009).
    Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
    Learning Methods. SIGIR 2009.
"""

import json
import math
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# configurations

VECTORSTORE_DIR = Path("vectorstore")
INDEX_PATH      = VECTORSTORE_DIR / "policy.index"
META_PATH       = VECTORSTORE_DIR / "meta.json"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DEFAULT   = 5

# RRF constant — standard value from Cormack et al. (2009)
RRF_K = 60

# load resources

print("[INFO] Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("[INFO] Loading metadata...")
with META_PATH.open("r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

print(f"[INFO] Index size:         {index.ntotal}")
print(f"[INFO] Metadata entries:   {len(CHUNKS)}")

if index.ntotal != len(CHUNKS):
    print("[WARN] Index size and metadata length differ — "
          "check your indexing step.")

print(f"[INFO] Loading embedding model: {MODEL_NAME}")
EMB = SentenceTransformer(MODEL_NAME)

# build BM25 index from the same chunk corpus

print("[INFO] Building BM25 index...")

def _tokenise(text):
    """lowercase whitespace tokenisation for BM25."""
    return text.lower().split()

_bm25_corpus = [_tokenise(ch.get("text", "")) for ch in CHUNKS]
BM25_INDEX   = BM25Okapi(_bm25_corpus)

print(f"[INFO] BM25 index built over {len(_bm25_corpus)} documents.")

# internal helpers

def _chunk_to_result(idx, score):
    """convert a chunk index + score into a result dictionary."""
    chunk = CHUNKS[idx]
    return {
        "score":     score,
        "idx":       idx,
        "text":      chunk.get("text", ""),
        "doc_title": chunk.get("doc_title"),
        "section":   chunk.get("section"),
        **{
            k: v for k, v in chunk.items()
            if k not in {"text", "doc_title", "section"}
        },
    }


def _dense_ranked(query, k):
    """
    return a ranked list of chunk indices using FAISS dense retrieval.
    queries all-MiniLM-L6-v2 embeddings with cosine similarity.
    """
    q_vec = (
        EMB.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        .astype("float32")
    )
    scores, idxs = index.search(q_vec, k)
    return [int(i) for i in idxs[0] if i != -1]


def _sparse_ranked(query, k):
    """
    return a ranked list of chunk indices using BM25Okapi sparse retrieval.
    """
    tokens      = _tokenise(query)
    bm25_scores = BM25_INDEX.get_scores(tokens)

    # argsort descending, take top-k
    ranked = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    return ranked[:k]


def _rrf_fusion(dense_ranking, sparse_ranking, k_rrf=RRF_K):
    """
    Reciprocal Rank Fusion of two ranked lists.

    RRF score for document d = sum over each list L of 1 / (k_rrf + rank_L(d))
    where rank is 1-based.

    Reference: Cormack et al. (2009) SIGIR.
    """
    rrf_scores = {}

    for rank, idx in enumerate(dense_ranking, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)

    for rank, idx in enumerate(sparse_ranking, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)

    # sort by descending RRF score
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# retrieval function — embed natural language query, search indexes and return top k chunks
# returns a list of dictionaries

def retrieve_policies(query, k=TOP_K_DEFAULT, retrieval_method="hybrid"):
    """
    Retrieve the top-k most relevant policy chunks for a natural language query.

    Parameters
    ----------
    query : str
        The user's natural language question.
    k : int
        Number of chunks to return.
    retrieval_method : str
        One of "dense", "sparse", or "hybrid".
        - "dense"  : FAISS cosine similarity (all-MiniLM-L6-v2)
        - "sparse" : BM25Okapi keyword ranking
        - "hybrid" : Reciprocal Rank Fusion of dense + sparse (default)

    Returns
    -------
    list of dicts, each containing: score, idx, text, doc_title, section,
    retrieval_method, and any additional chunk metadata.
    """
    retrieval_method = retrieval_method.lower().strip()

    if retrieval_method == "dense":
        # embed query and search FAISS
        q_vec = (
            EMB.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            .astype("float32")
        )
        scores, idxs = index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            r = _chunk_to_result(int(idx), float(score))
            r["retrieval_method"] = "dense"
            results.append(r)
        return results

    elif retrieval_method == "sparse":
        # tokenise query and score with BM25
        tokens      = _tokenise(query)
        bm25_scores = BM25_INDEX.get_scores(tokens)
        ranked      = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:k]
        results = []
        for idx in ranked:
            r = _chunk_to_result(idx, float(bm25_scores[idx]))
            r["retrieval_method"] = "sparse"
            results.append(r)
        return results

    elif retrieval_method == "hybrid":
        # retrieve a larger candidate pool before fusing, then trim to k
        candidate_k    = max(k * 3, 20)
        dense_ranking  = _dense_ranked(query, candidate_k)
        sparse_ranking = _sparse_ranked(query, candidate_k)
        fused          = _rrf_fusion(dense_ranking, sparse_ranking)

        results = []
        for idx, rrf_score in fused[:k]:
            r = _chunk_to_result(idx, rrf_score)
            r["retrieval_method"] = "hybrid"
            results.append(r)
        return results

    else:
        raise ValueError(
            f"Unknown retrieval_method '{retrieval_method}'. "
            "Choose from: 'dense', 'sparse', 'hybrid'."
        )


# simple cli for testing all three retrieval modes

def main():
    print("\nRetrieval method? [dense / sparse / hybrid] (default: hybrid)")
    method = input("Method> ").strip().lower() or "hybrid"
    if method not in {"dense", "sparse", "hybrid"}:
        print("Invalid method, defaulting to hybrid.")
        method = "hybrid"

    print(f"\nUsing: {method.upper()} retrieval")
    print("Type a question about the policies (or 'exit' to quit):\n")

    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        hits = retrieve_policies(q, k=TOP_K_DEFAULT, retrieval_method=method)

        if not hits:
            print("No results found.\n")
            continue

        for i, h in enumerate(hits, start=1):
            print(f"\n=== RESULT {i} "
                  f"(score={h['score']:.4f}, "
                  f"method={h['retrieval_method']}, "
                  f"idx={h['idx']}) ===")
            print(f"Title:   {h.get('doc_title')}")
            print(f"Section: {h.get('section')}")
            print(h["text"][:500].replace("\n", " "), "...")
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()