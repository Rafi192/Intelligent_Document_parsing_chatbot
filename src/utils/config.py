# src/utils/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DATABASE_NAME = os.getenv("MONGODB_DATABASE", "ecommerce")
    COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "products")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/embeddings/mongodb_vectors")
