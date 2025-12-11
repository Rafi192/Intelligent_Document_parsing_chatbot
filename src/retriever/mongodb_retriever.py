
# from typing import List, Dict, Any
# import logging
# from src.ingestion.mongodb_indexer import MongoDBVectorIndexer
# # import os
# from pathlib import Path

# logger = logging.getLogger(__name__)


# class MongoDBRetriever:
#     def __init__(self, embedder, vector_store_path: str = "data/embeddings/mongodb_vectors"):
#         self.indexer = MongoDBVectorIndexer(embedder, vector_store_path)
#         self.indexer.load_index()
#         logger.info("MongoDB retriever initialized")
    
#     def retrieve(
#         self,
#         query: str,
#         top_k: int = 5,
#         filter_metadata: Dict[str, Any] = None
#     ) -> List[Dict[str, Any]]:
#         # takes the parameters and returns  List of relevant documents with similarity scores from the MongoDBVectorIndexer

#         results = self.indexer.search(query, top_k=top_k * 2)         
#         if filter_metadata:
#             results = [
#                 r for r in results
#                 if all(r['metadata'].get(k) == v for k, v in filter_metadata.items())
#             ]
        
#         return results[:top_k]
    
#     def retrieve_context(self, query: str, top_k: int = 3) -> str:
      
#         results = self.retrieve(query, top_k=top_k)
        
      
#         context_parts = []
#         for i, result in enumerate(results, 1):
#             context_parts.append(f"[Document {i}]")
#             context_parts.append(result['text'])
#             context_parts.append(f"(Relevance Score: {result['similarity_score']:.3f})")
#             context_parts.append("")  
        
#         return "\n".join(context_parts) # returns the formatted context for the LLM 


#---------------------------------


from typing import List, Dict, Any
import logging
from pathlib import Path

from src.ingestion.mongodb_indexer import MongoDBVectorIndexer
from src.retrievers.bm25_retriever import BM25Retriever  # <-- You must implement this lightweight class

logger = logging.getLogger(__name__)


class HybridMongoDBRetriever:
    def __init__(
        self,
        embedder,
        bm25_corpus: List[str],
        vector_store_path: str = "data/embeddings/mongodb_vectors",
        alpha: float = 0.5,  # weight between dense & sparse scores
    ):
        """
        alpha = 0.5 → equal weighting
        alpha > 0.5 → dense retrieval more important
        alpha < 0.5 → sparse retrieval more important
        """
        # Dense index
        self.indexer = MongoDBVectorIndexer(embedder, vector_store_path)
        self.indexer.load_index()

        # Sparse index (BM25)
        self.bm25 = BM25Retriever(bm25_corpus)

        self.alpha = alpha
        logger.info("Hybrid retriever initialized")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:

        # ------------------------------
        # 1. Dense results (embeddings)
        # ------------------------------
        dense_results = self.indexer.search(query, top_k=top_k * 2)

        # Convert to dict: doc_id → dense_score
        dense_map = {r["doc_id"]: r for r in dense_results}

        # ------------------------------
        # 2. Sparse results (BM25)
        # ------------------------------
        sparse_results = self.bm25.search(query, top_k=top_k * 2)

        # Convert BM25 → standardized format
        sparse_map = {r["doc_id"]: r for r in sparse_results}

        # ------------------------------
        # 3. Score merging
        # ------------------------------
        combined = {}

        all_ids = set(list(dense_map.keys()) + list(sparse_map.keys()))

        for doc_id in all_ids:
            dense_score = dense_map.get(doc_id, {}).get("similarity_score", 0)
            sparse_score = sparse_map.get(doc_id, {}).get("bm25_score", 0)

            final_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score

            combined[doc_id] = {
                "doc_id": doc_id,
                "text": dense_map.get(doc_id, sparse_map.get(doc_id)).get("text"),
                "metadata": dense_map.get(doc_id, sparse_map.get(doc_id)).get("metadata"),
                "score": final_score
            }

        # ------------------------------
        # 4. Filtering + Ranking
        # ------------------------------
        results = list(combined.values())
        results.sort(key=lambda x: x["score"], reverse=True)

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
            context_parts.append(f"(Hybrid Score: {result['score']:.3f})")
            context_parts.append("")

        return "\n".join(context_parts)
