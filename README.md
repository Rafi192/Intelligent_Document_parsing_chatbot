#  Intelligent Document Parsing Chatbot

A **RAG-based LLM chatbot** that can understand structured and unstructured data — including **PDF, Word, Excel, and text files** — and generate accurate answers to user queries based on document context.

---



## 📁 Project Folder Structure

```bash
rag_chatbot/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .env                         # API keys, DB credentials
│
├── 📁 data/
│   ├── raw/                        # Original uploaded documents (PDF, DOCX, etc.)
│   ├── processed/                  # Cleaned or extracted text files
│   └── embeddings/                 # Local vector store or FAISS index
│
├── 📁 src/
│   ├── __init__.py
│   │
│   ├── 📁 ingestion/               # Handles document ingestion and embedding
│   │   ├── loader.py               # Load PDFs, DOCX, etc.
│   │   ├── splitter.py             # Chunking logic
│   │   ├── embedder.py             # Embedding model (OpenAI, SentenceTransformer)
│   │   └── indexer.py              # Saves embeddings to vector DB
│   │
│   ├── 📁 retriever/               # Retrieves relevant docs from vector DB
│   │   ├── retriever.py
│   │   └── ranker.py               # (Optional) Re-ranks results by similarity
│   │
│   ├── 📁 llm/                     # LLM prompt construction & response generation
│   │   ├── generator.py            # Calls LLM to generate answer
│   │   └── prompt_builder.py       # Combines context + query into final prompt
│   │
│   ├── 📁 api/                     # FastAPI / Flask backend
│   │   ├── app.py                  # API entry point
│   │   └── routes.py               # Endpoints for chat, ingestion, etc.
│   │
│   ├── 📁 utils/                   # Helper functions and configs
│   │   ├── logger.py
│   │   ├── config.py
│   │   └── helpers.py
│   │
│   └── main.py                     # Entry point for chatbot pipeline
│
├── 📁 notebooks/                   # Jupyter notebooks for experiments
│
└── 📁 tests/                       # Unit tests
    ├── test_ingestion.py
    ├── test_retriever.py
    └── test_llm.py
