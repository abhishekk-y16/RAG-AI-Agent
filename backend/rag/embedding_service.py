"""
Embedding Service - Real Embeddings & Semantic Retrieval
Supports multiple embedding providers: OpenAI, Ollama (local), and Hugging Face
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import requests

load_dotenv()


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension"""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI Embedding API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding provider
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (text-embedding-3-small, text-embedding-3-large)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.dimension = 1536 if model == "text-embedding-3-small" else 3072
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": texts
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.text}")
        
        result = response.json()
        
        # Sort by index and extract embedding vectors
        embeddings = sorted(result["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class OllamaEmbedding(EmbeddingProvider):
    """Ollama local embedding provider (requires Ollama running locally)"""
    
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        """
        Initialize Ollama embedding provider
        
        Args:
            model: Ollama model to use (nomic-embed-text, all-minilm, etc.)
            host: Ollama server URL
        """
        self.model = model
        self.host = host
        self.embed_url = f"{host}/api/embeddings"
        
        # Test connection
        try:
            response = requests.get(f"{host}/api/tags")
            if response.status_code != 200:
                raise Exception("Ollama server not responding")
        except Exception as e:
            raise Exception(f"Cannot connect to Ollama at {host}: {e}")
        
        # Get dimension by embedding test text
        self.dimension = len(self._get_embedding("test"))
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        data = {
            "model": self.model,
            "prompt": text
        }
        
        response = requests.post(self.embed_url, json=data)
        
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")
        
        return response.json()["embedding"]
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        return self._get_embedding(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        embeddings = []
        for text in texts:
            embeddings.append(self._get_embedding(text))
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class HuggingFaceEmbedding(EmbeddingProvider):
    """Hugging Face local embedding provider using sentence-transformers"""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Hugging Face embedding provider
        
        Args:
            model: Hugging Face model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class EmbeddingService:
    """
    Unified embedding service supporting multiple providers
    """
    
    def __init__(self, provider: str = "openai", **kwargs):
        """
        Initialize embedding service
        
        Args:
            provider: "openai", "ollama", or "huggingface"
            **kwargs: Provider-specific arguments
        """
        self.provider_name = provider
        
        if provider.lower() == "openai":
            self.provider = OpenAIEmbedding(**kwargs)
        elif provider.lower() == "ollama":
            self.provider = OllamaEmbedding(**kwargs)
        elif provider.lower() == "huggingface":
            self.provider = HuggingFaceEmbedding(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.cache = {}  # Simple in-memory cache
    
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Embed a single text with optional caching
        """
        if use_cache and text in self.cache:
            return self.cache[text]
        
        embedding = self.provider.embed_text(text)
        
        if use_cache:
            self.cache[text] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Embed multiple texts with optional caching
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache
        for i, text in enumerate(texts):
            if use_cache and text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.provider.embed_batch(texts_to_embed)
            
            for text, embedding in zip(texts_to_embed, new_embeddings):
                if use_cache:
                    self.cache[text] = embedding
        
        # Reconstruct in original order
        result = [None] * len(texts)
        cache_idx = 0
        embed_idx = 0
        
        for i, text in enumerate(texts):
            if text in self.cache:
                result[i] = self.cache[text]
            else:
                result[i] = new_embeddings[embed_idx]
                embed_idx += 1
        
        return result
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.provider.get_embedding_dimension()
    
    def similarity_search(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """
        Find most similar candidates to query
        
        Returns:
            List of (candidate, similarity_score) tuples, sorted by similarity
        """
        # Embed query and candidates
        query_embedding = np.array(self.embed_text(query))
        candidate_embeddings = np.array(self.embed_batch(candidates))
        
        # Compute cosine similarity
        similarities = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            # Cosine similarity
            sim = np.dot(query_embedding, candidate_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_emb) + 1e-10
            )
            similarities.append((candidates[i], float(sim)))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_embeddings(self, embeddings_dict: Dict[str, List[float]], filepath: str):
        """Save embeddings to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(embeddings_dict, f)
    
    def load_embeddings(self, filepath: str) -> Dict[str, List[float]]:
        """Load embeddings from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)


def create_embedding_service(provider: Optional[str] = None, **kwargs) -> EmbeddingService:
    """
    Factory function to create embedding service
    
    If provider not specified, tries: OpenAI -> Ollama -> HuggingFace
    """
    if provider:
        return EmbeddingService(provider=provider, **kwargs)
    
    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        try:
            print("Using OpenAI embeddings...")
            return EmbeddingService(provider="openai")
        except Exception as e:
            print(f"OpenAI failed: {e}")
    
    # Try Ollama
    try:
        print("Trying Ollama...")
        service = EmbeddingService(provider="ollama")
        print("Using Ollama embeddings...")
        return service
    except Exception as e:
        print(f"Ollama failed: {e}")
    
    # Fall back to HuggingFace
    try:
        print("Using HuggingFace embeddings...")
        return EmbeddingService(provider="huggingface")
    except Exception as e:
        raise Exception("No embedding provider available. Install sentence-transformers or set OPENAI_API_KEY")
