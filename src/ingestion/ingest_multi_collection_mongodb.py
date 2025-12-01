"""
Multi-Collection MongoDB RAG Ingestion Pipeline
Processes multiple collections from medical practice database
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.multi_collection_mongodb_loader import MultiCollectionMongoDBLoader
from src.ingestion.mongodb_indexer import MongoDBVectorIndexer
from src.ingestion.multi_collection_embedder import get_embedder
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def display_collection_preview(formatted_docs: dict, max_per_collection: int = 2):
    """Display sample documents from each collection"""
    print("\n" + "="*70)
    print("📄 DOCUMENT PREVIEW")
    print("="*70)
    
    for collection_name, docs in formatted_docs.items():
        if not docs:
            continue
            
        print(f"\n📚 Collection: {collection_name}")
        print("-" * 70)
        
        for i, doc in enumerate(docs[:max_per_collection], 1):
            print(f"\n  Document {i}:")
            print(f"  ID: {doc['id']}")
            print(f"  Text Preview:")
            # Show first 200 chars of text
            preview = doc['text'][:200].replace('\n', ' ')
            print(f"  {preview}...")
            print(f"  Metadata: {doc['metadata']['collection']}")


def main():
    """
    Main ingestion pipeline for multi-collection medical database
    
    Steps:
    1. Connect to MongoDB
    2. Load and format documents from multiple collections
    3. Create embeddings using BGE-M3 (multilingual)
    4. Build unified FAISS index
    5. Save index for retrieval
    """
    
    print("\n" + "="*70)
    print("🏥 MULTI-COLLECTION MEDICAL DATABASE RAG INGESTION")
    print("="*70)
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # MongoDB connection
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("MONGODB_DATABASE", "test")
    
    # Vector store path
    VECTOR_STORE_PATH = "data/embeddings/medical_practice_vectors"
    
    # Embedding model
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Multilingual, high quality
    # Alternatives:
    # - "BAAI/bge-base-en-v1.5" (English only, faster)
    # - "bert-base-uncased" (lightweight)
    
    # Collections to process (None = all available)
    COLLECTIONS_TO_PROCESS = None  # Will auto-detect
    # Or specify: ['doctors', 'faqs', 'treatmentlists', 'treatmentfees']
    
    # Limit documents per collection (None = all)
    LIMIT_PER_COLLECTION = None  # Set to 50 for testing
    
    if not MONGODB_URI:
        print("\n❌ ERROR: MONGODB_URI not found in environment variables")
        print("Please set it in your .env file")
        return
    
    print(f"\n[1] Configuration:")
    print(f"    📊 Database: {DATABASE_NAME}")
    print(f"    🧠 Embedding Model: {EMBEDDING_MODEL}")
    print(f"    💾 Vector Store: {VECTOR_STORE_PATH}")
    
    # ============================================================
    # STEP 1: CONNECT TO MONGODB
    # ============================================================
    
    print(f"\n[2] Connecting to MongoDB...")
    
    loader = MultiCollectionMongoDBLoader(
        connection_string=MONGODB_URI,
        database_name=DATABASE_NAME
    )
    
    # Show available collections
    available_collections = loader.get_available_collections()
    print(f"    ✓ Found {len(available_collections)} RAG-compatible collections")
    print(f"    Collections: {', '.join(available_collections)}")
    
    if not available_collections:
        print("\n❌ No compatible collections found!")
        print("   Check your collection schemas in MultiCollectionMongoDBLoader")
        loader.close()
        return
    
    # ============================================================
    # STEP 2: LOAD AND FORMAT DOCUMENTS
    # ============================================================
    
    print(f"\n[3] Loading and formatting documents...")
    
    # Option A: Load as dictionary (organized by collection)
    formatted_by_collection = loader.load_and_format_all_collections(
        collections=COLLECTIONS_TO_PROCESS,
        limit_per_collection=LIMIT_PER_COLLECTION
    )
    
    # Option B: Load as flat list (for single unified index)
    all_documents = loader.load_all_formatted_flat(
        collections=COLLECTIONS_TO_PROCESS,
        limit_per_collection=LIMIT_PER_COLLECTION
    )
    
    if not all_documents:
        print("\n❌ No documents loaded!")
        print("   Check if your collections have data")
        loader.close()
        return
    
    print(f"\n    ✓ Total documents loaded: {len(all_documents)}")
    
    # Show collection breakdown
    collection_counts = {}
    for doc in all_documents:
        coll = doc['metadata']['collection']
        collection_counts[coll] = collection_counts.get(coll, 0) + 1
    
    print(f"\n    📊 Breakdown by collection:")
    for coll, count in sorted(collection_counts.items()):
        print(f"       - {coll}: {count} documents")
    
    # Display preview
    display_collection_preview(formatted_by_collection, max_per_collection=1)
    
    # ============================================================
    # STEP 3: INITIALIZE EMBEDDER
    # ============================================================
    
    print(f"\n[4] Initializing embedding model...")
    print(f"    Model: {EMBEDDING_MODEL}")
    
    embedder = get_embedder(model_name=EMBEDDING_MODEL)
    
    print(f"    ✓ Model loaded")
    print(f"    ✓ Embedding dimension: {embedder.get_embedding_dimension()}")
    
    # ============================================================
    # STEP 4: BUILD VECTOR INDEX
    # ============================================================
    
    print(f"\n[5] Creating vector indexer...")
    
    indexer = MongoDBVectorIndexer(
        embedder=embedder,
        vector_store_path=VECTOR_STORE_PATH
    )
    
    print(f"\n[6] Building FAISS index...")
    print(f"    ⏳ This may take several minutes for large collections...")
    print(f"    Processing {len(all_documents)} documents...")
    
    try:
        indexer.build_index(all_documents)
        print(f"    ✓ Index built successfully!")
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        logger.exception("Index building failed")
        loader.close()
        return
    
    # ============================================================
    # STEP 5: SAVE INDEX
    # ============================================================
    
    print(f"\n[7] Saving index to disk...")
    
    try:
        indexer.save_index()
        print(f"    ✓ Index saved to: {VECTOR_STORE_PATH}")
    except Exception as e:
        print(f"\n❌ Error saving index: {e}")
        logger.exception("Index saving failed")
        loader.close()
        return
    
    # ============================================================
    # COMPLETION
    # ============================================================
    
    print("\n" + "="*70)
    print("✅ INGESTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   - Total Documents: {len(all_documents)}")
    print(f"   - Collections Processed: {len(collection_counts)}")
    print(f"   - Vector Store Location: {VECTOR_STORE_PATH}/")
    print(f"   - Embedding Model: {EMBEDDING_MODEL}")
    print(f"   - Embedding Dimension: {embedder.get_embedding_dimension()}")
    
    print(f"\n✨ Your multi-collection RAG system is ready!")
    print(f"   You can now use the retriever to search across all collections.")
    
    # Close MongoDB connection
    loader.close()
    
    # ============================================================
    # OPTIONAL: TEST SEARCH
    # ============================================================
    
    test_search = input("\n\n🔍 Would you like to test a search query? (yes/no): ").strip().lower()
    
    if test_search == 'yes':
        while True:
            query = input("\nEnter your search query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\n🔎 Searching for: '{query}'")
            print("-" * 70)
            
            try:
                results = indexer.search(query, top_k=5)
                
                if not results:
                    print("No results found.")
                    continue
                
                print(f"\n📄 Top {len(results)} Results:\n")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. [{result['metadata']['collection']}]")
                    print(f"   Score: {result['similarity_score']:.4f}")
                    
                    # Show relevant fields based on collection
                    metadata = result['metadata']
                    if 'name' in metadata:
                        print(f"   Name: {metadata['name']}")
                    if 'question' in metadata and metadata['collection'] == 'faqs':
                        print(f"   Question: {metadata['question'][:100]}...")
                    
                    # Show text preview
                    text_preview = result['text'][:200].replace('\n', ' ')
                    print(f"   Preview: {text_preview}...")
                    print()
                
            except Exception as e:
                print(f"❌ Search error: {e}")
                logger.exception("Search failed")
    
    print("\n👋 Ingestion pipeline finished!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.exception("Pipeline failed")
        sys.exit(1)