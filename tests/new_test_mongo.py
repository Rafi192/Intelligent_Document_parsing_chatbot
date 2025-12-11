"""
Simple Test Script for Multi-Collection Medical RAG Chatbot
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever.mongodb_retriever import MongoDBRetriever
from src.ingestion.multi_collection_embedder import get_embedder
from src.llm.enhanced_generator import generate_llm_response


# Initialize components
print("Initializing RAG system...")
embedder = get_embedder()

mongodb_retriever = MongoDBRetriever(
    embedder=embedder,
    vector_store_path="data/embeddings/medical_practice_vectors"
)

print("✓ System ready!\n")


def query_mongodb_rag(user_query: str, session_id: str = "default") -> str:
    """
    Process user query using MongoDB RAG pipeline.
    
    Args:
        user_query: User's question
        session_id: Session ID for conversation history
        
    Returns:
        Generated response from LLM
    """
    # Retrieve relevant documents
    retrieved_docs = mongodb_retriever.retrieve(user_query, top_k=3)
    
    # Show what was retrieved
    print(f"\n[Retrieved {len(retrieved_docs)} documents]")
    for i, doc in enumerate(retrieved_docs, 1):
        collection = doc['metadata'].get('collection', 'unknown')
        score = doc['similarity_score']
        print(f"  {i}. {collection} (score: {score:.3f})")
    
    # Generate response using LLM
    response = generate_llm_response(user_query, retrieved_docs, session_id=session_id)
    
    return response


# Main chatbot loop
if __name__ == "__main__":
    print("="*70)
    print("Medical Practice Chatbot - Type 'exit' to quit")
    print("="*70)
    print()
    
    session_id = "test_session"
    
    while True:
        query = input("You: ").strip()
        
        if not query:
            continue
            
        if query.lower() in ["exit", "quit", "q"]:
            print("\n Goodbye!")
            break
        
        try:
            response = query_mongodb_rag(query, session_id=session_id)
            print(f"\nAI: {response}\n")
            print("-" * 70)
            
        except Exception as e:
            print(f"\n Error: {e}\n")