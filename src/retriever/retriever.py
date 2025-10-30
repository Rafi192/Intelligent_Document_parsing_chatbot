#Now comse the retrieving part of the RAG model

# src/retrieval/retriever.py
from src.ingestion.indexer import load_faiss_index

def get_retriever(index_path="data/embeddings/faiss_index", k=5):
    vectorstore = load_faiss_index(index_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever
