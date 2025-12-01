"""
MongoDB Vector Indexer - Updated for Multi-Collection Medical Database
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any
import numpy as np
import faiss
import pickle
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class MongoDBVectorIndexer:
    def __init__(
        self,
        embedder, 
        vector_store_path: str = "data/embeddings/medical_practice_vectors"  # UPDATED PATH
    ):
        """
        Initialize Vector Indexer
        
        Args:
            embedder: Embedding model instance
            vector_store_path: Path to store FAISS index (NEW: medical_practice_vectors)
        """
        self.embedder = embedder
        project_root = Path(__file__).parent.parent.parent  
        self.vector_store_path = (project_root / vector_store_path).resolve()
        self.index = None
        self.documents = []
        
        # Create directory if not exists
        os.makedirs(self.vector_store_path, exist_ok=True)
        logger.info(f"Vector store path: {self.vector_store_path}")
    
    def create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create embeddings for documents
        
        Args:
            documents: List of documents with 'text' field
            
        Returns:
            Numpy array of embeddings
        """
        texts = [doc['text'] for doc in documents]
        
        logger.info(f"Creating embeddings for {len(texts)} documents...")
        
        # OPTIMIZED: Use batch embedding if available
        if hasattr(self.embedder, 'embed_documents'):
            # Batch embedding (faster)
            embeddings = self.embedder.embed_documents(texts)
        else:
            # Fallback: One by one
            embeddings = []
            for i, text in enumerate(texts):
                if i % 50 == 0:
                    logger.info(f"  Embedded {i}/{len(texts)} documents...")
                embedding = self.embedder.embed_query(text)
                embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')   
        logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Build FAISS index from documents
        
        Args:
            documents: List of documents with 'text' and 'metadata'
        """
        logger.info(f"Building index for {len(documents)} documents...")
        
        # Create embeddings
        embeddings = self.create_embeddings(documents)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store documents for retrieval
        self.documents = documents
        
        logger.info(f"✓ Built FAISS index with {self.index.ntotal} vectors (dimension: {dimension})")
    
    def save_index(self):
        """Save FAISS index and document metadata to disk"""
        index_path = self.vector_store_path / "faiss_index.bin"
        metadata_path = self.vector_store_path / "documents.pkl"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save documents metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"✓ Saved index to {index_path}")
        logger.info(f"✓ Saved metadata to {metadata_path}")
        logger.info(f"✓ Total size: {len(self.documents)} documents")
    
    def load_index(self):
        """Load FAISS index and document metadata from disk"""
        index_path = self.vector_store_path / "faiss_index.bin"
        metadata_path = self.vector_store_path / "documents.pkl"
        
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}\n"
                f"Please run ingestion first: python src/ingestion/ingest_multi_collection_mongodb.py"
            )
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load documents metadata
        with open(metadata_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        logger.info(f"✓ Loaded index with {self.index.ntotal} vectors")
        logger.info(f"✓ Loaded {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents given a query
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of top-k most similar documents with scores
        """
        if self.index is None:
            raise ValueError(
                "Index not built or loaded. "
                "Call build_index() or load_index() first."
            )
        
        # Create query embedding
        query_embedding = self.embedder.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search the index
        scores, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        logger.info(f"Found {len(results)} similar documents for query: '{query[:50]}...'")
        return results


#------------------------------------------

# #mongodb_indexer.py
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.her.')))
# from typing import List, Dict, Any
# import numpy as np
# import faiss
# import pickle
# import logging
# from pathlib import Path
# import os
# from src.ingestion.multi_collection_embedder import get_embedder
# logger = logging.getLogger(__name__)


# class MongoDBVectorIndexer:
#     def __init__(
#         self,
#         embedder, 
#         vector_store_path: str = "data/embeddings/medical_practice_vectors"
#     ):
       
#         self.embedder = embedder
#         project_root = Path(__file__).parent.parent.parent  
#         self.vector_store_path = (project_root / vector_store_path).resolve()
#         self.index = None
#         self.documents = []
#         # This is just to ensure if the creating a directory if not exists
#         os.makedirs(self.vector_store_path, exist_ok=True)
    
#     def create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
#         texts = [doc['text'] for doc in documents]
        
#         logger.info(f"Creating embeddings for {len(texts)} documents...")
#         embeddings = []
#         for text in texts:
#             embedding = self.embedder.embed_query(text)
#             embeddings.append(embedding)
        
#         embeddings_array = np.array(embeddings).astype('float32')   
#         logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
#         return embeddings_array
    

#     def build_index(self, documents: List[Dict[str, Any]]):
#         embeddings = self.create_embeddings(documents)
        
#         # Building the FAISS index
#         dimension = embeddings.shape[1]
#         self.index = faiss.IndexFlatIP(dimension)  
        
#         # Normalizing embeddings for cosine similarity
#         faiss.normalize_L2(embeddings)
        
#         # Adding embeddings to index
#         self.index.add(embeddings)
        
#         # Storing documents for retrieval
#         self.documents = documents
        
#         logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    

#     def save_index(self):
#         """Save FAISS index and document metadata to disk."""
#         index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
#         metadata_path = os.path.join(self.vector_store_path, "documents.pkl")
        
#         # Saving FAISS index
#         faiss.write_index(self.index, index_path)
        
#         # Saving documents metadata
#         with open(metadata_path, 'wb') as f:
#             pickle.dump(self.documents, f)
        
#         logger.info(f"Saved index to {index_path}")
#         logger.info(f"Saved metadata to {metadata_path}")
    
#     def load_index(self):
#         """Load FAISS index and document metadata from disk."""
#         index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
#         metadata_path = os.path.join(self.vector_store_path, "documents.pkl")
        
#         if not os.path.exists(index_path):
#             raise FileNotFoundError(f"Index not found at {index_path}")
        
#         # Loading FAISS index
#         self.index = faiss.read_index(index_path)
        
#         # Loading documents metadata
#         with open(metadata_path, 'rb') as f:
#             self.documents = pickle.load(f)
        
#         logger.info(f"Loaded index with {self.index.ntotal} vectors")
#         logger.info(f"Loaded {len(self.documents)} documents")
    
#     def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """
#         Search for similar documents given a query.
        
#         Args:
#             query: Search query text
#             top_k: Number of results to return
            
#         Returns:
#             List of top-k most similar documents with scores
#         """
#         if self.index is None:
#             raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
#         # Creating query embedding
#         query_embedding = self.embedder.embed_query(query)
#         query_vector = np.array([query_embedding]).astype('float32')
        
#         # Normalizing for cosine similarity
#         faiss.normalize_L2(query_vector)
        
#         # Searching the index
#         scores, indices = self.index.search(query_vector, top_k)
        
#         # Prepare results
#         results = []
#         for idx, score in zip(indices[0], scores[0]):
#             if idx < len(self.documents):
#                 result = self.documents[idx].copy()
#                 result['similarity_score'] = float(score)
#                 results.append(result)
        
#         logger.info(f"Found {len(results)} similar documents for query")
#         return results