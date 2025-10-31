from src.ingestion.loader import load_document
from src.ingestion.splitter import split_documents

from src.ingestion.indexer import create_faiss_index
from src.retriever.retriever import get_retriever
import time

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

query = " give me the summary of why the 2008 market crashed?"

result = retriever.invoke(query)
# print(result)

for r in result:
    print("-----------")
    print(r.page_content)
