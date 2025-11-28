#this is used fro loading different documents
# src/ingestion/loader.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from pathlib import Path

def load_document(file_path: str):
    """Loads a document into LangChain Document objects."""
    file_ext = Path(file_path).suffix.lower()

    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_ext in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {file_path}")
    return documents
