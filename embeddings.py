import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Embedding Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# FAISS Index Parameters
FAISS_INDEX_PATH = "faiss_hnsw_index.idx"
HNSW_M = 32  # Higher M = better recall, slightly slower index time
EF_SEARCH = 64  # Higher = more accurate retrieval
D = 384  # Embedding dimension for MiniLM

# Text Chunking Parameters
CHUNK_SIZE = 256
OVERLAP = 50


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Splits text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def encode_text(texts):
    """Encodes a list of texts into embeddings."""
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def get_text_embeddings(text):
    """Encodes a single text input into an embedding (for external calls)."""
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]


def create_faiss_index(embeddings):
    """Creates an HNSW FAISS index from embeddings."""
    index = faiss.IndexHNSWFlat(D, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_SEARCH  # Better accuracy during indexing
    index.hnsw.efSearch = EF_SEARCH  # Better accuracy during search

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index


def save_faiss_index(index, path=FAISS_INDEX_PATH):
    """Saves FAISS index to disk."""
    faiss.write_index(index, path)


def load_faiss_index(path=FAISS_INDEX_PATH):
    """Loads FAISS index from disk if it exists."""
    if os.path.exists(path):
        return faiss.read_index(path)
    return None


def load_text_files(directory):
    """Loads all text files from a directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Only process .txt files
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                documents.append(file.read())  # Read file content
    return documents


def process_documents():
    """Loads, processes, and indexes text documents."""
    directory = "text_files"  # Change this to your folder name
    if not os.path.exists(directory):
        print(f"❌ Directory '{directory}' not found! Please check the path.")
        return

    print(f"📂 Loading text files from '{directory}'...")
    documents = load_text_files(directory)
    if not documents:
        print("❌ No text files found!")
        return

    print(f"📝 Processing {len(documents)} documents...")
    text_chunks = Parallel(n_jobs=-1)(delayed(chunk_text)(doc) for doc in documents)
    text_chunks = [chunk for sublist in text_chunks for chunk in sublist]  # Flatten

    print(f"🔢 Encoding {len(text_chunks)} text chunks...")
    embeddings = encode_text(text_chunks)

    print("📦 Creating FAISS index...")
    index = create_faiss_index(embeddings)

    print("💾 Saving FAISS index and text chunks...")
    save_faiss_index(index)

    with open("text_chunks.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

    print(f"✅ FAISS index saved successfully! Indexed {len(text_chunks)} chunks.")


if __name__ == "__main__":
    process_documents()
