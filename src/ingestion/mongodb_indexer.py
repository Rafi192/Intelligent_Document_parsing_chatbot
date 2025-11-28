import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.her.')))
from typing import List, Dict, Any
import numpy as np
import faiss
import pickle
import logging
from pathlib import Path
import os
from src.ingestion.embedder_bge import get_embedder
logger = logging.getLogger(__name__)


class MongoDBVectorIndexer:
    def __init__(
        self,
        embedder, 
        vector_store_path: str = "data/embeddings/mongodb_vectors"
    ):
       
        self.embedder = embedder
        project_root = Path(__file__).parent.parent.parent  
        self.vector_store_path = (project_root / vector_store_path).resolve()
        self.index = None
        self.documents = []
        # This is just to ensure if the creating a directory if not exists
        os.makedirs(self.vector_store_path, exist_ok=True)
    
    def create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        texts = [doc['text'] for doc in documents]
        
        logger.info(f"Creating embeddings for {len(texts)} documents...")
        embeddings = []
        for text in texts:
            embedding = self.embedder.embed_query(text)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')   
        logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
    

    def build_index(self, documents: List[Dict[str, Any]]):
        embeddings = self.create_embeddings(documents)
        
        # Building the FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  
        
        # Normalizing embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Adding embeddings to index
        self.index.add(embeddings)
        
        # Storing documents for retrieval
        self.documents = documents
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    

    def save_index(self):
        """Save FAISS index and document metadata to disk."""
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        metadata_path = os.path.join(self.vector_store_path, "documents.pkl")
        
        # Saving FAISS index
        faiss.write_index(self.index, index_path)
        
        # Saving documents metadata
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
        
        # Loading FAISS index
        self.index = faiss.read_index(index_path)
        
        # Loading documents metadata
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
        
        # Creating query embedding
        query_embedding = self.embedder.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalizing for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Searching the index
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