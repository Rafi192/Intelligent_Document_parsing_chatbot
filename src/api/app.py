from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_classic.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from langchain_classic.docstore.document import Document
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.generator import generate_llm_response, get_session_history

import tempfile


from PyPDF2 import PdfReader

from langchain_classic.document_loaders import(
    TextLoader, # for .txt files
    PyPDFLoader, # for .pdf files
    Docx2txtLoader, # for .docx file
    UnstructuredMarkdownLoader # for .md
)

from pathlib import Path
from sentence_transformers import SentenceTransformer



app = Flask(__name__)

CORS(app)

def process_files(file_path: str):
    file_ext = Path(file_path).suffix.lower()

    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_ext in [".doc", ".docx"]:
        loader = Docx2txtLoader(file_path)
    elif file_ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip() != ""] # filtering empyt
    return documents


from langchain_core.embeddings import Embeddings
from typing import List

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


embeddings = SentenceTransformerEmbeddings('bert-base-uncased')


def create_temp_vectorestore(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap= 100)

    docs = [Document(page_content = t) for t in texts]
    docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    return vector_store



@app.route('/ask_AI', methods = ["POST"])

def ask_AI():
    """
    Expects multipart/form-data
    - files : uploads files (PDFs, wordc, text)
    - query : question string
    - session id : optional session id for memory
    """

    query = request.form.get("query")

    session_id  = request.form.get("session_id", "default_session")
    files = request.files.getlist("file")

    if not query:
        return jsonify({"eror": "query is required"})
    
    if not files:
        return jsonify({"error": "No files given, file required"})
    
    texts = []

    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # temp_path = tmp.name
            file.save(tmp.name)
            tmp_path = tmp.name

        
        docs = process_files(tmp_path)
        texts.extend([doc.page_content for doc in docs])
        os.unlink(tmp_path)

        
    #creating the temporary vector store
    temp_store = create_temp_vectorestore(texts)
    # Retrieve top docs
    retrieved_docs = temp_store.similarity_search(query, k=4)

    # Generate LLM response with memory
    response = generate_llm_response(query, retrieved_docs, session_id=session_id)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



