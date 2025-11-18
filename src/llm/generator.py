import sys
import os
from pathlib import Path

# Add project root to path - go up TWO levels from generator.py
# generator.py is in src/llm/, so we need to go up to project root

# ---------------------------
# Setup path and environment
# ---------------------------
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------
# Imports
# ---------------------------

from dotenv import load_dotenv
load_dotenv(dotenv_path=r"C:\Users\hasan\Rafi_SAA\practice_project_1\Intelligent_Document_parsing_chatbot\.env")
from openai import OpenAI

from src.llm.augmented_prompt import augmented_prompt

# Initializing the OpenAI client with Hugging Face endpoint for LLaMA
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN_READ"],
)

# In-memory conversation store (session_id -> list of messages)
store = {}


def get_session_history(session_id: str):
    """
    Get or create conversation history for a session.
    Returns list of message dicts compatible with OpenAI format.
    """
    if session_id not in store:
        store[session_id] = []
    
    return store[session_id]


def generate_llm_response(query, retrieved_docs, session_id="default_session", max_docs=4):
    """
    Generate response using Meta LLaMA with retrieved docs + conversation memory.
    
    Args:
        query: User's question
        retrieved_docs: Documents retrieved from vector store
        session_id: Session identifier for conversation history
        max_docs: Maximum number of documents to include in context
    
    Returns:
        str: Generated response from LLaMA
    """
    
    # Build augmented context prompt
    user_input_text = augmented_prompt(query, retrieved_docs, max_docs)
    
    # Get conversation history for this session
    history = get_session_history(session_id)
    
    # System prompt
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant that analyzes documents and answers questions based on retrieved information and conversation context. Provide accurate, concise responses."
    }
    
    # Add current user message to history
    history.append({
        "role": "user",
        "content": user_input_text
    })
    
    # Prepare messages: system prompt + conversation history
    # Keep last 10 messages to avoid token limits
    max_history_length = 10
    if len(history) > max_history_length:
        messages = [system_prompt] + history[-max_history_length:]
    else:
        messages = [system_prompt] + history
    
    try:
        # Call LLaMA model via HuggingFace
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=messages,
            max_tokens=1024,
            temperature=0.3,  # Lower temperature for more factual responses
        )
        
        # Extract response
        assistant_response = completion.choices[0].message.content
        
        # Add assistant response to history
        history.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # Update store
        store[session_id] = history
        
        return assistant_response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


def clear_session_history(session_id: str):
    """
    Clear conversation history for a specific session.
    """
    if session_id in store:
        store[session_id] = []
        print(f"Session {session_id} history cleared.")


def get_all_sessions():
    """
    Get list of all active session IDs.
    """
    return list(store.keys())