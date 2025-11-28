
from typing import List, Dict, Any
import logging
from src.ingestion.mongodb_indexer import MongoDBVectorIndexer
# import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MongoDBRetriever:
    def __init__(self, embedder, vector_store_path: str = "data/embeddings/mongodb_vectors"):
        self.indexer = MongoDBVectorIndexer(embedder, vector_store_path)
        self.indexer.load_index()
        logger.info("MongoDB retriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        # takes the parameters and returns  List of relevant documents with similarity scores from the MongoDBVectorIndexer

        results = self.indexer.search(query, top_k=top_k * 2)         
        if filter_metadata:
            results = [
                r for r in results
                if all(r['metadata'].get(k) == v for k, v in filter_metadata.items())
            ]
        
        return results[:top_k]
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
      
        results = self.retrieve(query, top_k=top_k)
        
      
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(result['text'])
            context_parts.append(f"(Relevance Score: {result['similarity_score']:.3f})")
            context_parts.append("")  
        
        return "\n".join(context_parts) # returns the formatted context for the LLM 
