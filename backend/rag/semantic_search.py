"""
Semantic Search Engine - Vector-based document retrieval
Uses real embeddings for semantic similarity search across document chunks
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .embedding_service import EmbeddingService


class VectorStore:
    """Vector store for embeddings with similarity search"""
    
    def __init__(self, dimension: int):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding vector dimension
        """
        self.dimension = dimension
        self.vectors = {}  # chunk_id -> embedding vector
        self.metadata = {}  # chunk_id -> metadata dict
        self.chunk_ids = []  # Ordered list of chunk IDs
    
    def add(self, chunk_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Add a vector with metadata"""
        self.vectors[chunk_id] = np.array(embedding)
        self.metadata[chunk_id] = metadata
        self.chunk_ids.append(chunk_id)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors
        
        Returns:
            List of (chunk_id, similarity_score, metadata) tuples
        """
        if not self.vectors:
            return []
        
        query_vec = np.array(query_embedding)
        similarities = []
        
        for chunk_id, vec in self.vectors.items():
            # Cosine similarity
            sim = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-10
            )
            similarities.append((chunk_id, float(sim), self.metadata[chunk_id]))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            "dimension": self.dimension,
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
            "metadata": self.metadata,
            "chunk_ids": self.chunk_ids
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.dimension = data["dimension"]
        self.metadata = data["metadata"]
        self.chunk_ids = data["chunk_ids"]
        
        for chunk_id, vec in data["vectors"].items():
            self.vectors[chunk_id] = np.array(vec)
    
    def size(self) -> int:
        """Return number of vectors"""
        return len(self.vectors)


class SemanticSearchEngine:
    """Semantic search engine using real embeddings"""
    
    def __init__(self, embedding_service: EmbeddingService, vector_store_path: str = "backend/data/vector_store.json"):
        """
        Initialize semantic search engine
        
        Args:
            embedding_service: Embedding service instance
            vector_store_path: Path to save/load vector store
        """
        self.embedding_service = embedding_service
        self.vector_store_path = vector_store_path
        self.vector_store = VectorStore(embedding_service.get_dimension())
        
        # Load existing vector store if available
        if Path(vector_store_path).exists():
            try:
                self.vector_store.load(vector_store_path)
                print(f"Loaded {self.vector_store.size()} vectors from {vector_store_path}")
            except Exception as e:
                print(f"Could not load vector store: {e}")
    
    def index_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Index a list of document chunks
        
        Args:
            chunks: List of dicts with 'chunk_id', 'text', and optionally 'doc_id', 'page'
        
        Returns:
            Number of chunks indexed
        """
        texts = [chunk.get("text", "") for chunk in chunks]
        chunk_ids = [chunk.get("chunk_id", f"chunk_{i}") for i, chunk in enumerate(chunks)]
        
        # Embed all texts
        embeddings = self.embedding_service.embed_batch(texts, use_cache=True)
        
        # Add to vector store
        indexed_count = 0
        for chunk_id, embedding, chunk in zip(chunk_ids, embeddings, chunks):
            metadata = {
                "text": chunk.get("text", ""),
                "doc_id": chunk.get("doc_id", "unknown"),
                "page": chunk.get("page", 0),
                "chunk_index": chunk.get("chunk_index", 0)
            }
            self.vector_store.add(chunk_id, embedding, metadata)
            indexed_count += 1
        
        # Save vector store
        self.vector_store.save(self.vector_store_path)
        print(f"Indexed {indexed_count} chunks, total: {self.vector_store.size()}")
        
        return indexed_count
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for semantically similar chunks
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            filters: Optional filters (e.g., {"doc_id": "doc123"})
        
        Returns:
            List of results with chunk info and similarity scores
        """
        # Embed query
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k*2)  # Get more to filter
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for chunk_id, sim, metadata in results:
                if all(metadata.get(k) == v for k, v in filters.items()):
                    filtered_results.append((chunk_id, sim, metadata))
            results = filtered_results[:top_k]
        else:
            results = results[:top_k]
        
        # Format output
        output = []
        for chunk_id, similarity, metadata in results:
            output.append({
                "chunk_id": chunk_id,
                "text": metadata["text"],
                "doc_id": metadata["doc_id"],
                "page": metadata["page"],
                "similarity": similarity
            })
        
        return output
    
    def search_by_doc(self, query: str, doc_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within a specific document"""
        return self.search(query, top_k=top_k, filters={"doc_id": doc_id})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            "total_chunks_indexed": self.vector_store.size(),
            "embedding_dimension": self.embedding_service.get_dimension(),
            "embedding_provider": self.embedding_service.provider_name,
            "cache_size": len(self.embedding_service.cache)
        }


class HybridSearchEngine:
    """
    Hybrid search combining semantic search with keyword matching
    """
    
    def __init__(self, semantic_engine: SemanticSearchEngine):
        """Initialize hybrid search engine"""
        self.semantic_engine = semantic_engine
    
    def _keyword_score(self, query: str, text: str) -> float:
        """
        Calculate keyword matching score
        
        Returns score between 0 and 1
        """
        query_words = query.lower().split()
        text_lower = text.lower()
        
        matches = sum(1 for word in query_words if word in text_lower)
        score = min(matches / len(query_words), 1.0) if query_words else 0.0
        
        return score
    
    def search(self, query: str, top_k: int = 5, 
               semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
        
        Returns:
            Combined results sorted by hybrid score
        """
        # Get semantic results
        semantic_results = self.semantic_engine.search(query, top_k=top_k*3)
        
        # Calculate hybrid scores
        results_with_scores = []
        for result in semantic_results:
            keyword_score = self._keyword_score(query, result["text"])
            hybrid_score = (semantic_weight * result["similarity"] + 
                          keyword_weight * keyword_score)
            
            result["keyword_score"] = keyword_score
            result["hybrid_score"] = hybrid_score
            results_with_scores.append(result)
        
        # Sort by hybrid score
        results_with_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return results_with_scores[:top_k]
