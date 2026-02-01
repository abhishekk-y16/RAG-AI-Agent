"""
RAG Answer Module - Phase 4 Upgrade with Real Embeddings
Uses semantic search engine for intelligent retrieval
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq
from .semantic_search import SemanticSearchEngine, HybridSearchEngine
from .embedding_service import EmbeddingService


class SemanticRAG:
    """Retrieval-Augmented Generation with semantic search"""
    
    def __init__(self, 
                 embedding_service: EmbeddingService,
                 semantic_search_engine: Optional[SemanticSearchEngine] = None,
                 use_hybrid: bool = False):
        """
        Initialize Semantic RAG
        
        Args:
            embedding_service: Embedding service for text vectorization
            semantic_search_engine: Vector search engine
            use_hybrid: Use hybrid search (semantic + keyword)
        """
        self.embedding_service = embedding_service
        
        # Initialize search engine
        if semantic_search_engine is None:
            self.search_engine = SemanticSearchEngine(embedding_service)
        else:
            self.search_engine = semantic_search_engine
        
        # Setup hybrid search if requested
        if use_hybrid:
            self.search_engine = HybridSearchEngine(self.search_engine)
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.groq_client = Groq(api_key=api_key)
    
    def index_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Index document chunks for semantic search
        
        Args:
            chunks: List of chunk dicts with 'text', 'doc_id', etc.
        
        Returns:
            Number of chunks indexed
        """
        return self.search_engine.index_documents(chunks)
    
    def retrieve(self, query: str, doc_id: Optional[str] = None, 
                top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using semantic search
        
        Args:
            query: Query text
            doc_id: Optional document ID to limit search
            top_k: Number of chunks to retrieve
        
        Returns:
            List of retrieved chunks with metadata and similarity scores
        """
        if doc_id:
            results = self.search_engine.search_by_doc(query, doc_id, top_k=top_k)
        else:
            results = self.search_engine.search(query, top_k=top_k)
        
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]],
                       model: str = "mixtral-8x7b-32768", temperature: float = 0.3) -> str:
        """
        Generate answer using retrieved chunks as context
        
        Args:
            query: Original query
            retrieved_chunks: Chunks from retrieve()
            model: Groq model to use
            temperature: Model temperature
        
        Returns:
            Generated answer
        """
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Chunk from doc {c.get('doc_id', 'unknown')}]:\n{c.get('text', '')}"
            for c in retrieved_chunks
        ])
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant specializing in document analysis.
Answer questions based on the provided context. If the context doesn't contain enough information 
to answer the question, say so clearly. Be concise but thorough."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        # Call Groq API
        message = self.groq_client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return message.content[0].text
    
    def answer_question(self, query: str, doc_id: Optional[str] = None,
                       top_k: int = 5, model: str = "mixtral-8x7b-32768",
                       temperature: float = 0.3) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve then generate
        
        Args:
            query: User query
            doc_id: Optional document to search in
            top_k: Number of chunks to retrieve
            model: Groq model
            temperature: Model temperature
        
        Returns:
            Dict with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, doc_id=doc_id, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved, model=model, temperature=temperature)
        
        # Return structured response
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "chunk_id": r.get("chunk_id"),
                    "doc_id": r.get("doc_id"),
                    "page": r.get("page"),
                    "similarity": r.get("similarity"),
                    "preview": r.get("text", "")[:200] + "..."
                }
                for r in retrieved
            ],
            "metadata": {
                "model": model,
                "chunks_retrieved": len(retrieved),
                "embedding_provider": self.embedding_service.provider_name
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "search_engine": self.search_engine.semantic_engine.get_stats() 
                            if hasattr(self.search_engine, 'semantic_engine')
                            else self.search_engine.get_stats()
        }


def create_rag_system(provider: Optional[str] = None, 
                     use_hybrid: bool = False) -> SemanticRAG:
    """
    Factory function to create a complete RAG system
    
    Args:
        provider: Embedding provider ("openai", "ollama", "huggingface")
        use_hybrid: Use hybrid search
    
    Returns:
        Initialized SemanticRAG instance
    """
    # Create embedding service
    embedding_service = EmbeddingService(provider=provider) if provider else EmbeddingService()
    
    # Create semantic RAG
    return SemanticRAG(embedding_service, use_hybrid=use_hybrid)
