from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class BGE_M3_Embedder:
    def __init__(self, model_name="BAAI/bge-m3", batch_size=32, chunk_size=300, verbose=True):
        """
        Args:
            model_name (str): Name of the SentenceTransformer model
            batch_size (int): Number of texts to process at once
            chunk_size (int): Maximum tokens per chunk (approximate for long texts)
            verbose (bool): Print debug info
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.verbose = verbose

    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunks.append(" ".join(words[i:i+self.chunk_size]))
        return chunks

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches with normalization."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            embeddings = self.model.encode(batch, normalize_embeddings=True)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents, handling long text chunking.
        Returns a list of embeddings (one per document or chunk).
        """
        if not texts:
            return []

        # Chunk long texts
        all_chunks = []
        chunk_map = []  # map to reconstruct document embeddings if needed
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                continue
            chunks = self._chunk_text(text)
            all_chunks.extend(chunks)
            chunk_map.append((idx, len(chunks)))

        # Embed all chunks
        embeddings = self._embed_batch(all_chunks)

        if self.verbose:
            print(f"Embedded {len(all_chunks)} chunks from {len(texts)} documents.")
            print(f"Vector shape: {embeddings.shape}")

        # Aggregate chunk embeddings per document (mean pooling)
        doc_embeddings = []
        start = 0
        for _, n_chunks in chunk_map:
            doc_vec = embeddings[start:start+n_chunks].mean(axis=0)
            doc_embeddings.append(doc_vec)
            start += n_chunks

        return [vec.tolist() for vec in doc_embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        """
        if not text or not text.strip():
            return []

        chunks = self._chunk_text(text)
        embeddings = self._embed_batch(chunks)
        query_embedding = embeddings.mean(axis=0)

        if self.verbose:
            print(f"Embedded query (length {len(text.split())} words).")
            print(f"Vector shape: {query_embedding.shape}")

        return query_embedding.tolist()


def get_embedder(verbose=True):
    return BGE_M3_Embedder(verbose=verbose)