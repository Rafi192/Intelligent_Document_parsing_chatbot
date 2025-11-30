from typing import List, Dict, Any, Optional
from pymongo import MongoClient
import logging
from datetime import datetime
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class MultiCollectionMongoDBLoader:
    """
    Enhanced MongoDB loader for handling multiple collections with different schemas.
    Designed for medical practice database with doctors, treatments, FAQs, etc.
    """
    
    # Define schema for each collection - which fields to extract and how to format them
    COLLECTION_SCHEMAS = {
    "doctors": {
        "fields": ["name", "title", "description"],
        "required_fields": ["name", "title", "description"],
        "template": """Doctor Information:
Name: {name}
Title: {title}
Description: {description}
"""
    },

    "faqs": {
        "fields": ["question", "answer"],
        "required_fields": ["question", "answer"],
        "template": """FAQ:
Question: {question}
Answer: {answer}
"""
    },

    "treatmentlists": {
        "fields": ["serviceName", "description"],
        "required_fields": ["serviceName", "description"],
        "template": """Treatment Information:
Service Name: {serviceName}
Description: {description}
"""
    },

    "treatmentcategories": {
        "fields": ["name"],
        "required_fields": ["name"],
        "template": """Treatment Category:
Name: {name}
"""
    },

    "treatmentfees": {
        "fields": ["serviceName", "currency"],
        "required_fields": ["serviceName", "currency"],
        "template": """Treatment Pricing:
Treatment Name: {serviceName}
Price: {currency}
"""
    },

    "contactinfos": {
        "fields": ["address", "email", "openingHours", "phoneNumbers"],
        "required_fields": ["address", "email", "openingHours", "phoneNumbers"],
        "template": """Contact Information:
Address: {address}
Phone Numbers: {phoneNumbers}
Email: {email}
Opening Hours: {openingHours}
"""
    },

    "privacypolicies": {
        "fields": ["title", "content", "section", "last_updated"],
        "required_fields": ["content"],
        "template": """Privacy Policy:
Title: {title}
Section: {section}
Last Updated: {last_updated}
Content: {content}
"""
    },

    "termsofservices": {
        "fields": ["policyContent"],
        "required_fields": ["policyContent"],
        "template": """Terms of Service:
Content: {policyContent}
"""
    },

    "gdprs": {
        "fields": ["gdprContent"],
        "required_fields": ["gdprContent"],
        "template": """GDPR Policy:
Content: {gdprContent}
"""
    }
}

    # Collections to EXCLUDE from RAG (privacy/security concerns)
    EXCLUDED_COLLECTIONS = ['users', 'bookings', 'referrals', 'galleries', 'contacts']
    
    def __init__(self, connection_string: str, database_name: str):
        """
        Initialize loader for multiple collections
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of the database
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.database_name = database_name
        logger.info(f"Connected to MongoDB: {database_name}")
    
    def get_available_collections(self) -> List[str]:
        """Get all collections that can be used for RAG"""
        all_collections = self.db.list_collection_names()
        rag_collections = [
            col for col in all_collections 
            if col in self.COLLECTION_SCHEMAS and col not in self.EXCLUDED_COLLECTIONS
        ]
        logger.info(f"Found {len(rag_collections)} RAG-compatible collections: {rag_collections}")
        return rag_collections
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and clean text"""
        if not text or not isinstance(text, str):
            return ""
        soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text(separator=' ', strip=True)
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = str(text)
        # Clean HTML if present
        if '<' in text and '>' in text:
            text = self.clean_html(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def load_collection_documents(
        self,
        collection_name: str,
        filter_query: Optional[Dict] = None,
        projection: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load documents from a single collection
        
        Args:
            collection_name: Name of the collection
            filter_query: MongoDB filter query
            projection: Fields to include/exclude
            limit: Maximum number of documents
            
        Returns:
            List of raw MongoDB documents
        """
        if collection_name not in self.COLLECTION_SCHEMAS:
            logger.warning(f"No schema defined for collection: {collection_name}")
            return []
        
        if collection_name in self.EXCLUDED_COLLECTIONS:
            logger.warning(f"Collection {collection_name} is excluded from RAG")
            return []
        
        collection = self.db[collection_name]
        filter_query = filter_query or {}
        
        query = collection.find(filter_query, projection)
        if limit:
            query = query.limit(limit)
        
        documents = list(query)
        logger.info(f"Loaded {len(documents)} documents from {collection_name}")
        return documents
    
    def format_document_for_rag(
        self, 
        document: Dict[str, Any], 
        collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Format a single document for RAG pipeline based on its collection schema
        
        Args:
            document: Raw MongoDB document
            collection_name: Source collection name
            
        Returns:
            Formatted document with text and metadata, or None if invalid
        """
        schema = self.COLLECTION_SCHEMAS.get(collection_name)
        if not schema:
            return None
        
        # Check required fields
        for required_field in schema['required_fields']:
            if required_field not in document or not document[required_field]:
                logger.warning(f"Missing required field '{required_field}' in {collection_name}")
                return None
        
        # Extract and clean fields
        extracted_data = {}
        for field in schema['fields']:
            value = document.get(field, '')
            
            # Handle different data types
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            elif isinstance(value, dict):
                value = ', '.join(f"{k}: {v}" for k, v in value.items())
            
            # Clean the text
            cleaned_value = self.clean_text(value) if value else ''
            extracted_data[field] = cleaned_value
        
        # Format using template
        try:
            formatted_text = schema['template'].format(**extracted_data)
        except KeyError as e:
            logger.error(f"Template formatting error for {collection_name}: {e}")
            return None
        
        # Skip if text is too short
        if len(formatted_text.strip()) < 20:
            logger.warning(f"Formatted text too short for {collection_name}")
            return None
        
        # Create formatted document
        return {
            'id': str(document.get('_id', '')),
            'text': formatted_text,
            'metadata': {
                'source': 'mongodb',
                'database': self.database_name,
                'collection': collection_name,
                'document_id': str(document.get('_id', '')),
                'loaded_at': datetime.now().isoformat(),
                # Include key fields in metadata for filtering
                **{k: v for k, v in extracted_data.items() if k in schema['required_fields']}
            }
        }
    
    def load_and_format_collection(
        self,
        collection_name: str,
        filter_query: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load and format documents from a single collection
        
        Args:
            collection_name: Name of collection to process
            filter_query: MongoDB filter query
            limit: Maximum number of documents
            
        Returns:
            List of formatted documents ready for embedding
        """
        raw_documents = self.load_collection_documents(
            collection_name=collection_name,
            filter_query=filter_query,
            limit=limit
        )
        
        formatted_documents = []
        for doc in raw_documents:
            formatted = self.format_document_for_rag(doc, collection_name)
            if formatted:
                formatted_documents.append(formatted)
        
        logger.info(f"Formatted {len(formatted_documents)}/{len(raw_documents)} documents from {collection_name}")
        return formatted_documents
    
    def load_and_format_all_collections(
        self,
        collections: Optional[List[str]] = None,
        limit_per_collection: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load and format documents from multiple collections
        
        Args:
            collections: List of collection names (None = all available)
            limit_per_collection: Max documents per collection
            
        Returns:
            Dictionary mapping collection names to formatted documents
        """
        if collections is None:
            collections = self.get_available_collections()
        
        all_formatted = {}
        total_docs = 0
        
        print(f"\n{'='*70}")
        print(f"Loading from {len(collections)} collections...")
        print(f"{'='*70}\n")
        
        for collection_name in collections:
            print(f"📚 Processing: {collection_name}...", end=' ')
            formatted = self.load_and_format_collection(
                collection_name=collection_name,
                limit=limit_per_collection
            )
            all_formatted[collection_name] = formatted
            total_docs += len(formatted)
            print(f"✓ {len(formatted)} documents")
        
        print(f"\n{'='*70}")
        print(f"✅ Total: {total_docs} documents from {len(collections)} collections")
        print(f"{'='*70}\n")
        
        return all_formatted
    
    def load_all_formatted_flat(
        self,
        collections: Optional[List[str]] = None,
        limit_per_collection: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all collections and return as a flat list (for single index)
        
        Args:
            collections: List of collection names
            limit_per_collection: Max documents per collection
            
        Returns:
            Flat list of all formatted documents
        """
        collection_data = self.load_and_format_all_collections(
            collections=collections,
            limit_per_collection=limit_per_collection
        )
        
        # Flatten to single list
        all_documents = []
        for collection_name, docs in collection_data.items():
            all_documents.extend(docs)
        
        return all_documents
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("MongoDB connection closed")