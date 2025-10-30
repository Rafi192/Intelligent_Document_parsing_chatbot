# now comes the data embedding part of the RAG model to store that in the vector DB

# converting text into vectors

# src/ingestion/embedder.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def get_embedder(model_name="bert-base-uncased"):
    """
    Returns a BERT-based embedding model and tokenizer.
    Model examples:
      - 'bert-base-uncased' (standard BERT)
      - 'bert-large-uncased' (larger, more accurate)
      - 'distilbert-base-uncased' (smaller & faster)
      - 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract' (domain-specific)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    return tokenizer, model

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings to get sentence embeddings.
    Takes into account the attention mask for proper averaging.
    """
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_documents(docs, model_name="bert-base-uncased", batch_size=32):
    """
    Creates embeddings for a list of LangChain Document objects using BERT.
    
    Args:
        docs: List of LangChain Document objects
        model_name: Name of the BERT model to use
        batch_size: Number of documents to process at once
    
    Returns:
        numpy array of embeddings
    """
    tokenizer, model = get_embedder(model_name)
    texts = [doc.page_content for doc in docs]
    
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Apply mean pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings (optional but recommended for similarity search)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        all_embeddings.append(embeddings.cpu().numpy())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} documents")
    
    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)
    
    print(f"Generated {len(embeddings)} embeddings using {model_name}.")
    return embeddings