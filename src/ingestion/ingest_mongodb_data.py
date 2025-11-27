
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.mongodb_loader import MongoDBLoader
from src.ingestion.mongodb_indexer import MongoDBVectorIndexer
from src.ingestion.embedder_bge import get_embedder  #bge-m3 embedder
from src.utils.config import Config  
from src.utils.logger import setup_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger(__name__)


def main():
    """
    Main function to ingest MongoDB data into vector database.
    
    Steps:
    1. Connect to MongoDB and load product data
    2. Format documents for RAG pipeline
    3. Create embeddings using existing embedder
    4. Build FAISS index
    5. Save index for retrieval
    """

    
    print("=" * 70)
    print("MONGODB DATA INGESTION INTO RAG VECTOR DATABASE")
    print("=" * 70)
    
    # Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DATABASE_NAME = os.getenv("MONGODB_DATABASE", "ecommerce")
    COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "products")
    
    print(f"\n[1] Connecting to MongoDB...")
    print(f"    Database: {DATABASE_NAME}")
    print(f"    Collection: {COLLECTION_NAME}")
    
    # Initialize MongoDB loader
    mongo_loader = MongoDBLoader(
        connection_string=MONGODB_URI,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME
    )
    
    # Load and format documents
    print(f"\n[2] Loading documents from MongoDB...")
    
    # Optional: Add filter query if needed
    # filter_query = {"category": "Electronics", "stock_status": "In Stock"}
    filter_query = None  # Load all documents
    
    documents = mongo_loader.load_and_format(
        filter_query=filter_query,
        limit=None  # Set limit if you want to test with subset
    )
    
    print(f"    Loaded {len(documents)} documents")
    
    if len(documents) == 0:
        print("\n  No documents found in MongoDB collection!")
        print("    Please ensure your MongoDB collection has data.")
        return
    
    # Show sample document
    print(f"\n[3] Sample document:")
    print(f"    ID: {documents[0]['id']}")
    print(f"    Text preview: {documents[0]['text'][:200]}...")
    

    print(f"\n[4] Initializing embedder...")
    embedder = get_embedder()  
    
    # Initialize vector indexer
    print(f"\n[5] Creating vector index...")
    indexer = MongoDBVectorIndexer(
        embedder=embedder,
        vector_store_path="data/embeddings/mongodb_vectors"
    )
    
    # Build index
    print(f"\n[6] Building FAISS index (this may take a while)...")
    indexer.build_index(documents)
    
    # Save index
    print(f"\n[7] Saving index to disk...")
    indexer.save_index()
    
    print("\n" + "=" * 70)
    print(" INGESTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nVector database created with {len(documents)} documents")
    print(f"Location: data/embeddings/mongodb_vectors/")
    print(f"\nYou can now use the retriever to search this data!")
    
    # Close MongoDB connection
    mongo_loader.close()
    
    # Optional: Test search
    test_search = input("\n\nWould you like to test a search query? (yes/no): ")
    if test_search.lower() == 'yes':
        query = input("Enter your search query: ")
        print(f"\nSearching for: '{query}'")
        results = indexer.search(query, top_k=3)
        
        print(f"\n Top {len(results)} Results:")
        print("-" * 70)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['metadata']['name']}")
            print(f"   Score: {result['similarity_score']:.4f}")
            print(f"   Category: {result['metadata']['category']}")
            print(f"   Price: ${result['metadata']['price']}")


if __name__ == "__main__":
    main()