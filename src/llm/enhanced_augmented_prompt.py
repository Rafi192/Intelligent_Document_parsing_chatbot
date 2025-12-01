"""
Enhanced Augmented Prompt Builder for Multi-Collection Medical RAG
Creates well-structured prompts with retrieved context
"""

from typing import List, Dict, Any


def augmented_prompt(
    query: str, 
    retrieved_docs: List[Dict[str, Any]], 
    max_docs: int = 4,
    include_scores: bool = True,
    include_sources: bool = True
) -> str:
    """
    Build an augmented prompt with retrieved context for LLM
    
    Args:
        query: User's question
        retrieved_docs: List of retrieved documents from vector search
                       Each doc should have: 'text', 'metadata', 'similarity_score'
        max_docs: Maximum number of documents to include
        include_scores: Whether to show relevance scores
        include_sources: Whether to show source collections
        
    Returns:
        Formatted prompt string ready for LLM
    """
    if not retrieved_docs:
        return f"""
You are a helpful medical practice assistant. The user asked:

"{query}"

However, no relevant information was found in the database. Please provide a helpful response 
based on general knowledge, or suggest that the user contact the practice directly for specific information.
"""
    
    prompt_parts = []
    
    # Header
    prompt_parts.append("=== CONTEXT FROM DATABASE ===\n")
    
    # Add each retrieved document
    for i, doc in enumerate(retrieved_docs[:max_docs], 1):
        # Extract metadata
        collection = doc.get('metadata', {}).get('collection', 'unknown')
        score = doc.get('similarity_score', 0.0)
        text = doc.get('text', '')
        
        # Build document header
        header_parts = [f"Document {i}"]
        if include_sources:
            header_parts.append(f"from {collection.upper()}")
        if include_scores:
            header_parts.append(f"(Relevance: {score:.3f})")
        
        prompt_parts.append(f"[{' '.join(header_parts)}]")
        prompt_parts.append(text)
        prompt_parts.append("")  # Blank line for readability
    
    # Add user query
    prompt_parts.append("=== USER QUESTION ===")
    prompt_parts.append(query)
    
    return "\n".join(prompt_parts)


def augmented_prompt_medical(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    max_docs: int = 4
) -> str:
    """
    Specialized prompt builder for medical practice queries
    Includes specific instructions for medical context
    
    Args:
        query: User's question
        retrieved_docs: Retrieved documents
        max_docs: Maximum documents to include
        
    Returns:
        Formatted prompt with medical-specific instructions
    """
    prompt_parts = []
    
    # Context section
    prompt_parts.append("=== MEDICAL PRACTICE INFORMATION ===\n")
    
    if not retrieved_docs:
        prompt_parts.append("No specific information found in the database for this query.\n")
    else:
        for i, doc in enumerate(retrieved_docs[:max_docs], 1):
            collection = doc.get('metadata', {}).get('collection', 'unknown')
            score = doc.get('similarity_score', 0.0)
            
            # Collection-specific formatting
            if collection == 'doctors':
                prompt_parts.append(f"[Healthcare Provider Information - Relevance: {score:.3f}]")
            elif collection == 'treatmentlists':
                prompt_parts.append(f"[Treatment/Procedure Information - Relevance: {score:.3f}]")
            elif collection == 'treatmentfees':
                prompt_parts.append(f"[Pricing Information - Relevance: {score:.3f}]")
            elif collection == 'faqs':
                prompt_parts.append(f"[Frequently Asked Question - Relevance: {score:.3f}]")
            elif collection == 'contactinfos':
                prompt_parts.append(f"[Contact Information - Relevance: {score:.3f}]")
            else:
                prompt_parts.append(f"[{collection.title()} Information - Relevance: {score:.3f}]")
            
            prompt_parts.append(doc['text'])
            prompt_parts.append("")
    
    # User question
    prompt_parts.append("=== PATIENT INQUIRY ===")
    prompt_parts.append(query)
    prompt_parts.append("")
    
    # Instructions
    prompt_parts.append("=== RESPONSE GUIDELINES ===")
    prompt_parts.append(
        "Provide a helpful, professional response based on the information above. "
        "Remember:\n"
        "- Be clear and empathetic\n"
        "- If discussing treatments, mention that consultation with a healthcare provider is recommended\n"
        "- For booking requests, direct to contact information or booking procedures\n"
        "- If information is incomplete, acknowledge limitations and suggest contacting the practice\n"
        "- Maintain patient confidentiality and professionalism"
    )
    
    return "\n".join(prompt_parts)


def augmented_prompt_with_intent(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    intent: str = "general",
    max_docs: int = 4
) -> str:
    """
    Build prompt that adapts based on detected query intent
    
    Args:
        query: User's question
        retrieved_docs: Retrieved documents
        intent: Detected intent ('booking', 'pricing', 'doctor_info', 'treatment_info', 'general')
        max_docs: Maximum documents
        
    Returns:
        Intent-adapted prompt
    """
    prompt_parts = []
    
    # Intent-specific headers
    intent_headers = {
        'booking': "=== APPOINTMENT BOOKING INFORMATION ===",
        'pricing': "=== PRICING AND FEE INFORMATION ===",
        'doctor_info': "=== HEALTHCARE PROVIDER INFORMATION ===",
        'treatment_info': "=== TREATMENT AND PROCEDURE INFORMATION ===",
        'contact': "=== CONTACT INFORMATION ===",
        'general': "=== RELEVANT INFORMATION ==="
    }
    
    prompt_parts.append(intent_headers.get(intent, intent_headers['general']))
    prompt_parts.append("")
    
    # Add retrieved documents
    for i, doc in enumerate(retrieved_docs[:max_docs], 1):
        collection = doc.get('metadata', {}).get('collection', 'unknown')
        score = doc.get('similarity_score', 0.0)
        
        prompt_parts.append(f"[Source {i}: {collection} - Relevance: {score:.3f}]")
        prompt_parts.append(doc['text'])
        prompt_parts.append("")
    
    # User query
    prompt_parts.append(f"=== {intent.upper()} QUERY ===")
    prompt_parts.append(query)
    prompt_parts.append("")
    
    # Intent-specific instructions
    intent_instructions = {
        'booking': "Guide the user through the appointment booking process based on the information provided.",
        'pricing': "Provide clear pricing information. Mention if additional costs may apply and suggest contacting for a detailed quote.",
        'doctor_info': "Provide information about the healthcare provider, including specializations and experience.",
        'treatment_info': "Explain the treatment clearly, including benefits, duration, and any relevant considerations.",
        'contact': "Provide complete contact information and office hours.",
        'general': "Provide a helpful and accurate response based on the available information."
    }
    
    prompt_parts.append("=== INSTRUCTIONS ===")
    prompt_parts.append(intent_instructions.get(intent, intent_instructions['general']))
    
    return "\n".join(prompt_parts)


def build_context_summary(retrieved_docs: List[Dict[str, Any]], max_docs: int = 4) -> str:
    """
    Build a summary of what information was retrieved
    Useful for debugging or showing users what sources were used
    
    Returns:
        Summary string
    """
    if not retrieved_docs:
        return "No relevant documents found."
    
    summary_parts = []
    summary_parts.append(f"Retrieved {min(len(retrieved_docs), max_docs)} relevant document(s):")
    
    collection_counts = {}
    for doc in retrieved_docs[:max_docs]:
        collection = doc.get('metadata', {}).get('collection', 'unknown')
        collection_counts[collection] = collection_counts.get(collection, 0) + 1
    
    for collection, count in collection_counts.items():
        summary_parts.append(f"  - {count} from {collection}")
    
    return "\n".join(summary_parts)