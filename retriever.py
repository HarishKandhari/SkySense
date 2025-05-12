# retriever.py
import faiss
import numpy as np
import logging
from openai import OpenAI
from utils import OPENAI_API_KEY
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize OpenAI client for embeddings
db_client = OpenAI(api_key=OPENAI_API_KEY)

@lru_cache(maxsize=256)
def get_embedding(text: str) -> tuple[float, ...]:
    """
    Return the embedding vector for a given text using OpenAI.
    """
    try:
        resp = db_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return tuple(resp.data[0].embedding)
    except Exception as e:
        logging.error(f"Failed to generate embedding for text: {e}")
        return ()

def build_vector_store(text_list: list[str]) -> faiss.IndexFlatL2:
    """
    Build a FAISS vector store from a list of texts.
    Returns the FAISS index.
    """
    try:
        # Create embeddings
        vectors = [np.array(get_embedding(t), dtype="float32") for t in text_list]
        dim = vectors[0].shape[0]

        # Initialize FAISS index
        index = faiss.IndexFlatL2(dim)
        index.add(np.stack(vectors))
        return index
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}")
        raise

def search_similar_docs(query: str, index: faiss.IndexFlatL2, texts: list[str], k: int = 3) -> list[str]:
    """
    Search the FAISS index for the top-k documents most similar to the query.
    Returns a list of text chunks.
    """
    try:
        # Embed the query
        q_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
        # Perform search
        distances, indices = index.search(q_vec, k)
        # Map to text chunks
        results = [texts[i] for i in indices[0] if i < len(texts)]
        logging.info(f"RAG retrieved for query '{query}': {results}")
        return results
    except Exception as e:
        logging.error(f"RAG search failed: {e}")
        return []