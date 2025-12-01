"""
Enhanced MongoDB Retriever for Multi-Collection Medical Database
Supports collection-specific filtering and better context formatting
"""

from typing import List, Dict, Any, Optional
import logging
from src.ingestion.mongodb_indexer import MongoDBVectorIndexer

logger = logging.getLogger(__name__)


class MongoDBRetriever:
    """
    Retrieves relevant documents from multi-collection vector index
    """
    
    def __init__(self, embedder, vector_store_path: str = "data/embeddings/medical_practice_vectors"):
        """
        Initialize retriever with pre-built FAISS index
        
        Args:
            embedder: Embedding model instance
            vector_store_path: Path to FAISS index directory
        """
        self.indexer = MongoDBVectorIndexer(embedder, vector_store_path)
        self.indexer.load_index()
        logger.info(f"MongoDB retriever initialized with {len(self.indexer.documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        collection_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector index
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Generic metadata filters (key-value pairs)
            collection_filter: Specific collection name to search in
                              (e.g., 'doctors', 'faqs', 'treatmentlists')
        
        Returns:
            List of relevant documents with similarity scores
        """
        # Retrieve more results initially for filtering
        results = self.indexer.search(query, top_k=top_k * 3)
        
        # Apply collection-specific filter
        if collection_filter:
            results = [
                r for r in results
                if r['metadata'].get('collection') == collection_filter
            ]
            logger.info(f"Filtered to {len(results)} results from '{collection_filter}' collection")
        
        # Apply generic metadata filters
        if filter_metadata:
            results = [
                r for r in results
                if all(r['metadata'].get(k) == v for k, v in filter_metadata.items())
            ]
            logger.info(f"Applied metadata filters, {len(results)} results remaining")
        
        # Return top_k after filtering
        return results[:top_k]
    
    def retrieve_by_collection(self, query: str, collection: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents from a specific collection only
        
        Args:
            query: Search query
            collection: Collection name ('doctors', 'faqs', 'treatmentlists', etc.)
            top_k: Number of results
            
        Returns:
            List of relevant documents from specified collection
        """
        return self.retrieve(query, top_k=top_k, collection_filter=collection)
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 3,
        include_sources: bool = True,
        collection_filter: Optional[str] = None
    ) -> str:
        """
        Retrieve and format context for LLM prompt
        
        Args:
            query: Search query
            top_k: Number of documents to include
            include_sources: Whether to include source collection names
            collection_filter: Optional collection filter
            
        Returns:
            Formatted context string ready for LLM
        """
        results = self.retrieve(query, top_k=top_k, collection_filter=collection_filter)
        
        if not results:
            return "No relevant information found in the database."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            # Header with source collection
            if include_sources:
                collection_name = result['metadata'].get('collection', 'unknown')
                context_parts.append(f"[Document {i} - Source: {collection_name}]")
            else:
                context_parts.append(f"[Document {i}]")
            
            # Document text
            context_parts.append(result['text'])
            
            # Relevance score
            context_parts.append(f"(Relevance: {result['similarity_score']:.3f})")
            context_parts.append("")  # Empty line for readability
        
        formatted_context = "\n".join(context_parts)
        logger.info(f"Retrieved context from {len(results)} documents")
        
        return formatted_context
    
    def retrieve_context_with_metadata(
        self, 
        query: str, 
        top_k: int = 3,
        collection_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve context along with structured metadata
        Useful for building rich UI responses
        
        Returns:
            Dictionary with 'context' string and 'sources' list
        """
        results = self.retrieve(query, top_k=top_k, collection_filter=collection_filter)
        
        # Build context string
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            collection_name = result['metadata'].get('collection', 'unknown')
            
            # Add to context
            context_parts.append(f"[Document {i} from {collection_name}]")
            context_parts.append(result['text'])
            context_parts.append("")
            
            # Add to sources list
            sources.append({
                'collection': collection_name,
                'text_preview': result['text'][:200] + "...",
                'score': result['similarity_score'],
                'metadata': result['metadata']
            })
        
        return {
            'context': "\n".join(context_parts),
            'sources': sources,
            'total_results': len(results)
        }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded index
        
        Returns:
            Dictionary with index statistics
        """
        # Count documents by collection
        collection_counts = {}
        for doc in self.indexer.documents:
            collection = doc['metadata'].get('collection', 'unknown')
            collection_counts[collection] = collection_counts.get(collection, 0) + 1
        
        return {
            'total_documents': len(self.indexer.documents),
            'collections': collection_counts,
            'available_collections': list(collection_counts.keys())
        }


# Factory function for easy initialization
def get_retriever(embedder, vector_store_path: str = "data/embeddings/medical_practice_vectors") -> MongoDBRetriever:
    """
    Create and initialize a MongoDB retriever
    
    Args:
        embedder: Embedding model instance
        vector_store_path: Path to vector store
        
    Returns:
        Initialized MongoDBRetriever
    """
    return MongoDBRetriever(embedder, vector_store_path)