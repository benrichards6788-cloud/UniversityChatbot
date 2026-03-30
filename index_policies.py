import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# load embedding model
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# read JSONL chunks 

def load_chunks(jsonl_path: Path):
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

# encode chunks to embeddings
def embed_chunks(chunks):
    texts = [ch["text"] for ch in chunks]
    embs = EMB.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs

#build the FAISS index
def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  #cosine sim
    index.add(embs)
    return index

#save index and metadata
def save_index(index, embs, chunks, out_dir: Path):
    out_dir.mkdir(parents = True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "policy.index"))

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

#main
def main():
    input_jsonl = Path("data/chunks_semantic.jsonl")
    out_dir = Path("vectorstore")

    print("Loading chunks...")
    chunks = load_chunks(input_jsonl)

    print(f"Loaded {len(chunks)} chunks.")

    print("Encoding embeddings...")
    embs = embed_chunks(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embs)

    print("Saving index...")
    save_index(index, embs, chunks, out_dir)

    print("\nIndex built successfully!")

if __name__ == "__main__":
    main()