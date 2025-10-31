# test_retrieval.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingestion.indexer import load_faiss_index

import time
# Load existing FAISS index
start = time.time()
vectorstore = load_faiss_index(r"C:\Users\hasan\Rafi_SAA\practice_project_1\Intelligent_Document_parsing_chatbot\data\embeddings\faiss_index")

# Now you can test queries directly
query = "What are the top 5 reasons for the market crash?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke(query)
end = time.time()

print( f"total time taken for laoding and querying the doc --{end-start:.2f} seconds")
print("Retrieved Chunks:")
for r in results:
    print(r.page_content)


