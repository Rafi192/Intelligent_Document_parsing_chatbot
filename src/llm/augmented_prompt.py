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
                               Each dict should have 'text' and 'metadata' fields.
        max_docs (int): Number of top documents to include in context.

    Returns:
        str: A complete prompt ready for LLM 
    """

    # 🔒 Step 1: Restricted content check
    query_lower = query.lower()
    if any(word in query_lower for word in RESTRICTED_WORDS):
        return (
            "I apologize, but I can only assist with product-related shopping questions. "
            "Is there a product I can help you find?"
        )

    # 🧠 Step 2: Build context if docs exist
    context_parts = []
    if retrieved_docs and len(retrieved_docs) > 0:
        for i, doc in enumerate(retrieved_docs[:max_docs], 1):
            metadata = doc.get('metadata', {})

            product_text = f"Product {i}:\n"
            if metadata.get('name'):
                product_text += f"Name: {metadata['name']}\n"
            if metadata.get('category'):
                product_text += f"Category: {metadata['category']}\n"
            if metadata.get('price'):
                product_text += f"Price: ${metadata['price']}\n"

            # Add main product description or text
            product_text += f"\n{doc.get('text', '')}\n"
            context_parts.append(product_text)

        context = "\n" + "=" * 70 + "\n".join(context_parts)

        # 🧩 Case 1: RAG context available
        prompt = f"""
You are a helpful e-commerce shopping assistant with access to product information.

Use the following product information to answer the customer's question accurately and concisely.
Focus on key features, prices, and availability. Compare products when relevant.

If the answer is not in the provided product information, you may also use your general knowledge or chat history.

Product Information:
{context}

Customer Question:
{query}

Answer:
"""

    else:
        # 🧩 Case 2: No context available → fallback to memory or reasoning
        prompt = f"""
You are a helpful e-commerce shopping assistant.

The following question may not be directly related to product data.
Use your general knowledge and the ongoing chat history to answer naturally and helpfully.

Customer Question:
{query}

Answer:
"""

    return prompt
