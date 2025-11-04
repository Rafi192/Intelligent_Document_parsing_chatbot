
from typing import List, Dict, Any
import logging
from src.ingestion.mongodb_indexer import MongoDBVectorIndexer
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MongoDBRetriever:
    """
    Retriever for MongoDB-based vector store.
    Integrates with your existing RAG pipeline's retriever.
    """
    
    def __init__(
        self,
        embedder,
        vector_store_path: str = "data/embeddings/mongodb_vectors"
    ):
        """
        Initialize retriever with embedder and vector store path.
        
        Args:
            embedder: Instance of your existing embedder
            vector_store_path: Path to saved FAISS index
        """
        self.indexer = MongoDBVectorIndexer(embedder, vector_store_path)
        self.indexer.load_index()
        logger.info("MongoDB retriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's search query
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters (e.g., {"category": "Electronics"})
            
        Returns:
            List of relevant documents with similarity scores
        """
        # Get search results
        results = self.indexer.search(query, top_k=top_k * 2)  # Get more initially
        
        # Apply metadata filters if provided
        if filter_metadata:
            results = [
                r for r in results
                if all(r['metadata'].get(k) == v for k, v in filter_metadata.items())
            ]
        
        # Return top_k after filtering
        return results[:top_k]
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve and format context for LLM prompt.
        
        Args:
            query: User's query
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string for LLM
        """
        results = self.retrieve(query, top_k=top_k)
        
        # Format context
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(result['text'])
            context_parts.append(f"(Relevance Score: {result['similarity_score']:.3f})")
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
