# src/ingestion/indexer.py
from langchain_community.vectorstores import FAISS
from .embedder_bge import get_embedder
import os


def create_faiss_index(docs, index_path="data/embeddings/faiss_index"):
    embedder = get_embedder()
    vectorstore = FAISS.from_documents(docs, embedder)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vectorstore.save_local(index_path)
    
    print(f"FAISS index saved to {index_path}")
    # print("Number of vectors:", len(vectorstore.index_to_docstore_id))
    # print("Example vector IDs:", list(vectorstore.index_to_docstore_id.items())[:5])
    # print("Vector dimension:", vectorstore.index.d)
   
    return vectorstore

def load_faiss_index(index_path="data/embeddings/faiss_index"):
    embedder = get_embedder()
    vectorstore = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
    print(f"Loaded FAISS index from {index_path}")
    return vectorstore
