"""
Enhanced embedder for multi-collection RAG pipeline
Supports both BERT and BGE-M3 models with improved batching and chunking
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from langchain.embeddings.base import Embeddings
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class MultiCollectionEmbedder(Embeddings):
    """
    Flexible embedder that can handle documents from multiple collections
    with different text lengths and structures
    """
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-base-en-v1.5",  # Default to BGE-M3 for multilingual
        device: str = None,
        max_length: int = 512,
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        """
        Initialize embedder
        
        Args:
            model_name: HuggingFace model name
                - "bert-base-uncased" (English only)
                - "BAAI/bge-m3" (Multilingual, recommended)
                - "BAAI/bge-base-en-v1.5" (English, high quality)
            device: "cuda", "cpu", or None (auto-detect)
            max_length: Maximum token length
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings
        """
        logger.info(f"Loading embedding model: {model_name}")
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"✓ Model loaded on {self.device}")
        logger.info(f"✓ Embedding dimension: {self.get_embedding_dimension()}")
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings
        """
        token_embeddings = model_output[0]  # First element contains token embeddings
        
        # Expand attention mask to match token embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings and divide by number of tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """
        Split long text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Characters per chunk
            overlap: Overlapping characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.7:  # Only break if not too early
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def embed_documents(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed a list of documents
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        if len(texts) == 0:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Pool and optionally normalize
            embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
            
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Log progress for large batches
            if len(texts) > 100 and (i // self.batch_size) % 10 == 0:
                logger.info(f"Embedded {i + len(batch_texts)}/{len(texts)} documents")
        
        # Combine all batches
        return np.vstack(all_embeddings)
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query
        
        Args:
            text: Query text
            
        Returns:
            1D numpy array of embeddings
        """
        return self.embed_documents([text])[0]
    
    def embed_documents_with_chunking(
        self, 
        texts: List[str],
        chunk_size: int = 400,
        overlap: int = 50
    ) -> tuple[np.ndarray, List[List[int]]]:
        """
        Embed documents with automatic chunking for long texts
        
        Args:
            texts: List of texts (can be very long)
            chunk_size: Characters per chunk
            overlap: Overlapping characters
            
        Returns:
            Tuple of:
                - Embeddings array (n_chunks, embedding_dim)
                - Chunk mapping (which chunks belong to which document)
        """
        all_chunks = []
        chunk_mapping = []
        
        for doc_idx, text in enumerate(texts):
            chunks = self.chunk_text(text, chunk_size, overlap)
            all_chunks.extend(chunks)
            chunk_mapping.append(list(range(len(all_chunks) - len(chunks), len(all_chunks))))
        
        logger.info(f"Split {len(texts)} documents into {len(all_chunks)} chunks")
        
        # Embed all chunks
        embeddings = self.embed_documents(all_chunks)
        
        return embeddings, chunk_mapping
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        dummy_embedding = self.embed_query("test")
        return len(dummy_embedding)


def get_embedder(model_name: str = "BAAI/bge-base-en-v1.5", **kwargs) -> MultiCollectionEmbedder:
    """
    Factory function to create embedder
    
    Args:
        model_name: HuggingFace model name
            Recommended options:
            - "BAAI/bge-m3" - Best for multilingual
            - "BAAI/bge-base-en-v1.5" - Best for English
            - "bert-base-uncased" - Lightweight option
        **kwargs: Additional arguments for MultiCollectionEmbedder
        
    Returns:
        Configured embedder instance
    """
    return MultiCollectionEmbedder(model_name=model_name, **kwargs)


# Backward compatibility with your existing code
# class BERTEmbeddings(MultiCollectionEmbedder):
#     """Alias for backward compatibility"""
#     def __init__(self, model_name="BAAI/bge-base-en-v1.5", device=None):
#         super().__init__(model_name=model_name, device=device)