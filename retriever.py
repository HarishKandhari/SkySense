import faiss
import numpy as np
from openai import OpenAI
from utils import OPENAI_API_KEY
from functools import lru_cache

# Initialize OpenAI client for embeddings
db_client = OpenAI(api_key=OPENAI_API_KEY)

@lru_cache(maxsize=256)
def get_embedding(text: str) -> tuple[float, ...]:
    """
    Return the embedding vector for a given text using OpenAI.
    """
    resp = db_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return tuple(resp.data[0].embedding)


def build_vector_store(text_list: list[str]) -> faiss.IndexFlatL2:
    """
    Build a FAISS vector store from a list of texts.
    Returns the FAISS index.
    """
    # Create embeddings
    vectors = [np.array(get_embedding(t), dtype="float32") for t in text_list]
    dim = vectors[0].shape[0]

    # Initialize FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(np.stack(vectors))
    return index


def search_similar_docs(query: str, index: faiss.IndexFlatL2, texts: list[str], k: int = 3) -> list[str]:
    """
    Search the FAISS index for the top-k documents most similar to the query.
    Returns a list of text chunks.
    """
    # Embed the query
    q_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)

    # Perform search
    distances, indices = index.search(q_vec, k)

    # Map to text chunks
    results = [texts[i] for i in indices[0] if i < len(texts)]
    return results
