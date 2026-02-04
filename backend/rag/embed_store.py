import json
import numpy as np
import faiss
from dotenv import load_dotenv
import os
import hashlib
from typing import List, Dict, Tuple, Any

# Load environment variables from .env file
load_dotenv()

EMBED_MODEL = "models/embedding-001"


def mock_embed_text(text: str) -> list:
    """Create mock embedding by hashing text (for quota-limited testing)"""
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to 768-dimensional vector (matches Gemini embedding size)
    embedding = []
    for i in range(768):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val - 128) / 128.0)
    
    return embedding


def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate embeddings  Uses mock for development due to API quotas"""
    embeddings = []
    for text in texts:
        # Use mock embeddings for development (avoids API quota limits)
        embedding = mock_embed_text(text)
        embeddings.append(embedding)
    
    arr = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_and_save_index(chunks: List[Dict[str, Any]], index_path: str, meta_path: str):
    """
    Build FAISS index from chunks and save metadata.
    
    Args:
        chunks: List of chunk dicts with 'text', 'doc_id', 'chunk_id', etc.
        index_path: Path to save FAISS index
        meta_path: Path to save metadata JSON
    """
    # Extract text for embeddings
    chunk_texts = [c["text"] for c in chunks]
    vectors = embed_texts(chunk_texts)
    dim = vectors.shape[1]

    # Build and save FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)

    # Save metadata with full chunk info
    metadata = {
        "total_chunks": len(chunks),
        "dimension": dim,
        "chunks": chunks
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index(index_path: str, meta_path: str) -> Tuple[Any, List[Dict]]:
    """
    Load FAISS index and metadata from disk.
    
    Returns:
        (faiss_index, chunks_with_metadata)
    """
    index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return index, meta.get("chunks", [])


def search_index(index: Any, query_text: str, chunks: List[Dict], 
                k: int = 4) -> List[Tuple[Dict, float]]:
    """
    Search FAISS index for similar chunks.
    
    Args:
        index: FAISS index
        query_text: Query text
        chunks: List of chunk metadata
        k: Number of results to return
    
    Returns:
        List of (chunk_dict, similarity_score)
    """
    query_vec = embed_texts([query_text])
    distances, indices = index.search(query_vec, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(distances[0][i])))
    
    return results
