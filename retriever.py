import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from joblib import Memory

# Load Embedding Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# FAISS Index Parameters
FAISS_INDEX_PATH = "faiss_hnsw_index.idx"
TEXT_CHUNKS_PATH = "text_chunks.pkl"
EF_SEARCH = 64  # Controls recall vs. speed

# Caching Mechanism to Speed Up Repeated Queries
memory = Memory(location="cache", verbose=0)


def load_faiss_index(path=FAISS_INDEX_PATH):
    """Loads FAISS index from disk."""
    if os.path.exists(path):
        index = faiss.read_index(path)
        index.hnsw.efSearch = EF_SEARCH  # Optimize retrieval speed
        return index
    raise FileNotFoundError("❌ FAISS index not found! Please run embeddings.py first.")


def load_text_chunks(path=TEXT_CHUNKS_PATH):
    """Loads text chunks corresponding to FAISS index."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("❌ Text chunks file not found!")


def encode_query(query):
    """Encodes a query into an embedding."""
    if not query.strip():  # Handle empty queries
        raise ValueError("⚠️ Query cannot be empty!")

    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return embedding


@memory.cache
def retrieve_documents(query, k=10):
    """Retrieves top-k most similar text chunks for a given query."""
    if isinstance(k, list):  # Ensure k is an integer
        k = k[0] if k else 10

    index = load_faiss_index()
    text_chunks = load_text_chunks()

    query_embedding = encode_query(query)
    _, indices = index.search(query_embedding, int(k))  # Ensure k is an integer

    results = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    return results



