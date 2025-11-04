

# Restricted words to check in queries
RESTRICTED_WORDS = [
    "kill", "murder", "suicide", "bomb", "weapon", "drug", "illegal",
    "hack", "crack", "pirate", "steal", "fraud", "scam", "password", 
    "credit card", "ssn", "cvv"
]


def augmented_prompt(query, retrieved_docs, max_docs=4):
    """
    Builds a context-aware prompt for the LLM using retrieved MongoDB documents.

    Args:
        query (str): User's search query
        retrieved_docs (list): List of dict objects from MongoDB retriever.
                               Each dict should have 'text' and 'metadata' fields
        max_docs (int): Number of top documents to include in context.

    Returns:
        str: A complete prompt ready for LLM 
    """
    
    # Check for restricted words in query
    query_lower = query.lower()
    if any(word in query_lower for word in RESTRICTED_WORDS):
        return """I apologize, but I can only assist with product-related shopping questions. 
Is there a product I can help you find?"""
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:max_docs], 1):
        # Get metadata
        metadata = doc.get('metadata', {})
        
        # Format product info
        product_text = f"Product {i}:\n"
        if metadata.get('name'):
            product_text += f"Name: {metadata['name']}\n"
        if metadata.get('category'):
            product_text += f"Category: {metadata['category']}\n"
        if metadata.get('price'):
            product_text += f"Price: ${metadata['price']}\n"
        
        # Add full text content
        product_text += f"\n{doc.get('text', '')}\n"
        
        context_parts.append(product_text)
    
    context = "\n" + "="*70 + "\n".join(context_parts)
    
    prompt = f"""You are a helpful e-commerce shopping assistant with access to product information.

Use the following product information to answer the customer's question accurately and concisely.
Focus on key features, prices, and availability. Compare products when relevant.

If the answer is not in the provided product information, say "I don't have information about that in our current catalog."

Product Information:
{context}

Customer Question:
{query}

Answer:

"""
    
    return prompt



# def augmented_propmt(query, retrieved_docs, max_docs =4):
#     """
#     Builds a context-aware prompt for the LLM using retrieved document chunks.

#     Args:
#         query (str):
#         retrived_docs (list): List of Document objects from retriever.
#         max_docs(int): Number of top documents to include in context.

#     returns:
#         str: A complete prompt ready for LLM 

#     """

#     context = "\n\n".join(
#         [doc.page_content for doc in retrieved_docs[:max_docs]]
#         )
    
#     prompt =  f"""
# You are an intelligent assistant with access to retrieved knowledge and chat history.

# use the following context to answer the user query cocisely and factually.
# if the answer is not in the context, say "I'm not sure based on the provided data"

# Context:
# {context}

# user Query:
# {query}

# Answer:

# """
#     return prompt


