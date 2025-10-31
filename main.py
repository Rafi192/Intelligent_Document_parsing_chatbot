from src.ingestion.loader import load_document
from src.ingestion.splitter import split_documents

from src.ingestion.indexer import create_faiss_index
from src.retriever.retriever import get_retriever
from src.llm.augmented_prompt import augmented_propmt
from src.llm.generator import generate_llm_response
import time
# from src.llm.generator import 

file_path = r"data/stock_market_analysis_one.pdf"
start  = time.time()
docs = load_document(file_path)
chunks = split_documents(docs)
index = create_faiss_index(chunks)
end = time.time()

print(f"Indexed {len(chunks)} chunks in {end-start:.2f} seconds")

# print("document in format .......",docs)
# print("Chunk in format..........",chunks)

# print("Number of vectors0..........",len( index.index_to_docstore_id))

# print("example vector Ids", list(index.index_to_docstore_id.items()))

# print("Vector dimension", index.index.d)

# print("------------------------------------------")

retriever = get_retriever()
# print("This is the retriever ----",len(retriever))

#taking the user query
query = str(input("Ask your questions to AI: "))

#retrieve  top k chunks 
retrieved_docs = retriever.invoke(query)
print(f"Retreived {len(retrieved_docs)} relevant chunks")


# building the augmented prompt:

# prompt = augmented_propmt(query, retrieved_docs)
# print(f"Augmented prompt : ", prompt)

response = generate_llm_response(query, retrieved_docs)

print("LLM response : ", response)



# result = retriever.invoke(query)
# # print(result)

# for r in result:
#     print("-----------")
#     print(r.page_content)
