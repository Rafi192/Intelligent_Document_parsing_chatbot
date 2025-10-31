
def augmented_propmt(query, retrieved_docs, max_docs =4):
    """
    Builds a context-aware prompt for the LLM using retrieved document chunks.

    Args:
        query (str):
        retrived_docs (list): List of Document objects from retriever.
        max_docs(int): Number of top documents to include in context.

    returns:
        str: A complete prompt ready for LLM 

    """

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs[:max_docs]]
        )
    
    prompt =  f"""
You are an intelligent assistant with access to retrieved knowledge.

use the following context to answer the user query cocisely and factually.
if the answer is not in the context, say "I'm not sure based on the provided data"

Context:
{context}

user Query:
{query}

Answer:

"""
    return prompt