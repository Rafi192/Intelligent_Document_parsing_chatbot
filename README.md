# Intelligent_Document_parsing_chatbot
A RAG based LLM chatbot that understands data such as pdf, word, excel, .txt and can generate answers based on query from user.

# 📁 Project Folder Structure
rag_chatbot/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .env                    # API keys, DB credentials
│
├── 📁 data/
│   ├── raw/                   # Original uploaded documents (PDF, DOCX, etc.)
│   ├── processed/             # Cleaned text files or JSON after preprocessing
│   └── embeddings/            # (Optional) Local vector store or FAISS index
│
├── 📁 src/
│   ├── __init__.py
│   │
│   ├── 📁 ingestion/          # Handles document ingestion and embedding
│   │   ├── loader.py          # Load PDFs, DOCX, etc.
│   │   ├── splitter.py        # Chunking logic
│   │   ├── embedder.py        # Embedding model (e.g. OpenAI, SentenceTransformer)
│   │   └── indexer.py         # Saves embeddings to vector DB
│   │
│   ├── 📁 retriever/          # Retrieves relevant docs from vector DB
│   │   ├── retriever.py
│   │   └── ranker.py          # (Optional) rerank results by similarity
│   │
│   ├── 📁 llm/
│   │   ├── generator.py       # Calls LLM to generate answer
│   │   └── prompt_builder.py  # Combines context + query into final prompt
│   │
│   ├── 📁 api/
│   │   ├── app.py             # FastAPI or Flask backend entry
│   │   └── routes.py          # API endpoints for chat, ingestion, etc.
│   │
│   ├── 📁 utils/
│   │   ├── logger.py
│   │   ├── config.py
│   │   └── helpers.py
│   │
│   └── main.py                # Entry point for chatbot pipeline
│
├── 📁 notebooks/              # Jupyter notebooks for testing & experiments
│
└── 📁 tests/                  # Unit tests
    ├── test_ingestion.py
    ├── test_retriever.py
    └── test_llm.py
