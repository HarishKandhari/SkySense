# build_index.py

import os
import pickle
import numpy as np
import faiss

from utils import load_sample_docs
from retriever import get_embedding

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 200
OUTPUT_PATH   = "data/vector_index.pkl"

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split `text` into chunks of `size` characters with `overlap`."""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

if __name__ == "__main__":
    print("👋 Hello from build_index.py")

    # 1️⃣ Load documents (we expect just one .txt)
    print("📄 Loading document(s)…")
    docs = load_sample_docs("data/sample_docs")
    if not docs:
        print("⚠️ No documents found in data/sample_docs/")
        exit(1)

    # We’ll concatenate them if there’s more than one, but typically it’s 1 file.
    full_text = "\n\n".join(docs)
    print(f"📄 Loaded document of length {len(full_text)} characters.")

    # 2️⃣ Chunk the single document
    print("✂️  Step 1: Chunking the single document into pieces…")
    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"📄 Created {len(chunks)} chunk(s).")

    # 3️⃣ Embed each chunk
    print("🔍 Step 2: Creating embeddings for each chunk…")
    vectors = [
        np.array(get_embedding(chunk), dtype="float32")
        for chunk in chunks
    ]
    print("✅ Embeddings generated.")

    # 4️⃣ Build a FAISS index
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.stack(vectors))
    print(f"⚙️ FAISS index created with chunked data.")

    # 5️⃣ Save the index + chunk texts
    print(f"💾 Saving index + chunks to {OUTPUT_PATH}…")
    with open(OUTPUT_PATH, "wb") as f:
        # we save a tuple of (index, chunks)
        pickle.dump((index, chunks), f)
    print("🎉 All done! Index saved successfully.")