# test_retrieval.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingestion.indexer import load_faiss_index
from src.retriever.retriever import get_retriever
from src.llm.generator import generate_llm_response
from src.retriever.mongodb_retriever import MongoDBRetriever
from src.ingestion.embedder import get_embedder


import time
# Load existing FAISS index
start = time.time()
vectorstore = load_faiss_index(r"C:\Users\hasan\Rafi_SAA\practice_project_1\Intelligent_Document_parsing_chatbot\data\embeddings\faiss_index")

# Now you can test queries directly
retriever = get_retriever()
# print("This is the retriever ----",len(retriever))

#taking the user queryw
# query = str(input("Ask your questions to AI: "))



# #retrieve  top k chunks 
# retrieved_docs = retriever.invoke(query)
# print(f"Retreived {len(retrieved_docs)} relevant chunks")


# # building the augmented prompt:

# # prompt = augmented_propmt(query, retrieved_docs)
# # print(f"Augmented prompt : ", prompt)

# response = generate_llm_response(query, retrieved_docs)

# print("LLM response : ", response)
# session_id = "user1"
print("Chatbot ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    retrieved_docs = retriever.invoke(query)
    response = generate_llm_response(query, retrieved_docs)
    print("AI:", response)

