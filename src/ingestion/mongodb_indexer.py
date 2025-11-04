# =============================================================================
# FILE 2: src/ingestion/mongodb_indexer.py
# Purpose: Create embeddings for MongoDB data and store in vector DB
# =============================================================================

import os
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
import faiss
import pickle
import logging
from pathlib import Path
import os
logger = logging.getLogger(__name__)


class MongoDBVectorIndexer:
    """
    Creates embeddings for MongoDB documents and stores them in FAISS vector DB.
    Integrates with existing RAG pipeline's embedder.
    """
    
    def __init__(
        self,
        embedder,  # Your existing embedder from src/ingestion/embedder.py
        vector_store_path: str = "data/embeddings/mongodb_vectors"
    ):
        """
        Initialize indexer with embedder and vector store path.
        
        Args:
            embedder: Instance of your existing embedder class
            vector_store_path: Path to save FAISS index and metadata
        """
        self.embedder = embedder
        project_root = Path(__file__).parent.parent.parent  # adjust if your folder depth differs
        self.vector_store_path = (project_root / vector_store_path).resolve()
        # self.vector_store_path = vector_store_path
        self.index = None
        self.documents = []
        
        # Create directory if it doesn't exist
        os.makedirs(vector_store_path, exist_ok=True)
    
    def create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create embeddings for a list of documents.
        
        Args:
            documents: List of formatted documents with 'text' field
            
        Returns:
            Numpy array of embeddings
        """
        texts = [doc['text'] for doc in documents]
        
        logger.info(f"Creating embeddings for {len(texts)} documents...")
        
        # Use your existing embedder
        embeddings = []
        for text in texts:
            embedding = self.embedder.embed_query(text)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')
        
        logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of formatted documents
        """
        # Create embeddings
        embeddings = self.create_embeddings(documents)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store documents for retrieval
        self.documents = documents
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def save_index(self):
        """Save FAISS index and document metadata to disk."""
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        metadata_path = os.path.join(self.vector_store_path, "documents.pkl")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save documents metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved index to {index_path}")
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_index(self):
        """Load FAISS index and document metadata from disk."""
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        metadata_path = os.path.join(self.vector_store_path, "documents.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents metadata
        with open(metadata_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents given a query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of top-k most similar documents with scores
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        # Create query embedding
        query_embedding = self.embedder.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        logger.info(f"Found {len(results)} similar documents for query")
        return results