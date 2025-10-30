from src.ingestion.loader import load_document
from src.ingestion.splitter import split_documents

from src.ingestion.indexer import create_faiss_index
from src.retriever.retriever import get_retriever

file_path = r"C:\Users\hasan\Rafi_SAA\practice_project_1\Intelligent_Document_parsing_chatbot\data\work17.pdf"

docs = load_document(file_path)
chunks = split_documents(docs)
index = create_faiss_index(chunks)

print("document in format .......",docs)
print("Chunk in format..........",chunks)

print("Number of vectors0..........",len( index.index_to_docstore_id))

print("example vector Ids", list(index.index_to_docstore_id.items()))

print("Vector dimension", index.index.d)

print("------------------------------------------")

retriever = get_retriever()

query = "what did rafi work on? "

result = retriever._get_relevant_documents(query)

for r in result:
    print("-----------")
    print(r.page_content)
