from typing import List, Dict, Any, Optional
from pymongo import MongoClient
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MongoDBLoader:
    """
    Loads product data from MongoDB for RAG pipeline ingestion.
    Connects to MongoDB and fetches documents based on filters.
    """
    
    def __init__(
        self,
        connection_string: str,
        database_name: str,
        collection_name: str
    ):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB URI (e.g., mongodb://localhost:27017/)
            database_name: Name of the database
            collection_name: Name of the collection to load from
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        logger.info(f"Connected to MongoDB: {database_name}.{collection_name}")
    
    def load_documents(
        self,
        filter_query: Optional[Dict] = None,
        projection: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load documents from MongoDB collection.
        
        Args:
            filter_query: MongoDB filter query (e.g., {"category": "Electronics"})
            projection: Fields to include/exclude (e.g., {"_id": 0, "name": 1})
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of documents from MongoDB
        """
        filter_query = filter_query or {}
        
        query = self.collection.find(filter_query, projection)
        
        if limit:
            query = query.limit(limit)
        
        documents = list(query)
        logger.info(f"Loaded {len(documents)} documents from MongoDB")
        
        return documents
    
    def format_product_for_rag(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format MongoDB product document for RAG pipeline.
        Converts product data into a structured format with text representation.
        
        Args:
            product: Raw product document from MongoDB
            
        Returns:
            Formatted document ready for embedding
        """
        # Create a comprehensive text representation
        text_parts = []
        
        # Product name and basic info
        if 'name' in product:
            text_parts.append(f"Product: {product['name']}")
        
        if 'category' in product:
            text_parts.append(f"Category: {product['category']}")
        
        if 'brand' in product:
            text_parts.append(f"Brand: {product['brand']}")
        
        if 'price' in product:
            text_parts.append(f"Price: ${product['price']}")
        
        # Description
        if 'description' in product:
            text_parts.append(f"Description: {product['description']}")
        
        # Features
        if 'features' in product and isinstance(product['features'], list):
            features_text = ', '.join(product['features'])
            text_parts.append(f"Features: {features_text}")
        
        # Specifications (if nested dict)
        if 'specifications' in product and isinstance(product['specifications'], dict):
            specs_text = ', '.join([f"{k}: {v}" for k, v in product['specifications'].items()])
            text_parts.append(f"Specifications: {specs_text}")
        
        # Stock status
        if 'stock_status' in product:
            text_parts.append(f"Availability: {product['stock_status']}")
        
        # Combine all parts
        combined_text = '\n'.join(text_parts)
        
        # Return formatted document
        return {
            'id': str(product.get('_id', product.get('product_id', ''))),
            'text': combined_text,
            'metadata': {
                'source': 'mongodb',
                'collection': self.collection.name,
                'product_id': product.get('product_id', ''),
                'name': product.get('name', ''),
                'category': product.get('category', ''),
                'price': product.get('price', 0),
                'loaded_at': datetime.now().isoformat()
            }
        }
    
    def load_and_format(
        self,
        filter_query: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load and format documents in one step.
        
        Args:
            filter_query: MongoDB filter query
            limit: Maximum number of documents
            
        Returns:
            List of formatted documents ready for embedding
        """
        raw_documents = self.load_documents(filter_query=filter_query, limit=limit)
        formatted_documents = [self.format_product_for_rag(doc) for doc in raw_documents]
        
        logger.info(f"Formatted {len(formatted_documents)} documents for RAG pipeline")
        return formatted_documents
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")
