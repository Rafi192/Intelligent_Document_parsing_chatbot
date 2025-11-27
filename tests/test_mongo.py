import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retriever.mongodb_retriever import MongoDBRetriever
from src.ingestion.embedder_bge import get_embedder
from src.llm.generator import generate_llm_response
# from src.llm.augmented_prompt import augmented_prompt

# Initialize components
embedder = get_embedder()
mongodb_retriever = MongoDBRetriever(embedder)
# prompt_builder = augmented_prompt()


def query_mongodb_rag(user_query: str) -> str:
    """
    Process user query using MongoDB RAG pipeline.

    Args:
        user_query: User's question

    Returns:
        Generated response from LLM
    """
    # Step 1: Retrieve relevant documents from MongoDB
    retrieved_docs = mongodb_retriever.retrieve(user_query, top_k=3)

    # Step 2: Build prompt with context
    
    # context = prompt_builder.build_prompt(
    #     query=user_query,
    #     context="\n\n".join([doc['text'] for doc in retrieved_docs]),
    #     system_message="You are a helpful e-commerce assistant."
    # )
    

    # Step 3: Generate response
    response = generate_llm_response(user_query, retrieved_docs)
    
    return response


# Example usage
if __name__ == "__main__":
    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = query_mongodb_rag(query)
        print(f"Query: {query}")
        print("AI:", response)
