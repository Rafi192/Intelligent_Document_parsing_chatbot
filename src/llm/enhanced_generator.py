"""
Enhanced LLM Generator for Multi-Collection Medical Chatbot
Uses Google Gemini with improved prompting for medical practice queries
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=str(project_root / ".env"))

import google.generativeai as genai
from typing import List, Dict, Any

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# In-memory conversation store
store = {}


def get_session_history(session_id: str) -> List[Dict]:
    """Get or create session history"""
    if session_id not in store:
        store[session_id] = []
    return store[session_id]


def _convert_to_gemini_messages(system_prompt: Dict, history: List[Dict]) -> List[Dict]:
    """Convert messages to Gemini format"""
    gemini_messages = []
    
    # Add system prompt as first user message
    gemini_messages.append({
        "role": "user",
        "parts": [system_prompt["content"]]
    })
    
    # Add conversation history
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_messages.append({
            "role": role,
            "parts": [msg["content"]]
        })
    
    return gemini_messages


def generate_llm_response(
    query: str, 
    retrieved_docs: List[Dict], 
    session_id: str = "default_session", 
    max_docs: int = 4
) -> str:
    """
    Generate LLM response using Gemini with retrieved context
    
    Args:
        query: User's question
        retrieved_docs: Documents retrieved from vector search
        session_id: Session identifier for conversation history
        max_docs: Maximum documents to include in context
        
    Returns:
        Generated response text
    """
    # Build context from retrieved documents
    context_parts = []
    context_parts.append("=== RELEVANT INFORMATION ===\n")
    
    for i, doc in enumerate(retrieved_docs[:max_docs], 1):
        collection = doc.get('metadata', {}).get('collection', 'unknown')
        score = doc.get('similarity_score', 0.0)
        
        context_parts.append(f"[Source {i}: {collection} - Relevance: {score:.3f}]")
        context_parts.append(doc['text'])
        context_parts.append("")
    
    context_parts.append("=== USER QUESTION ===")
    context_parts.append(query)
    
    user_input_text = "\n".join(context_parts)
    
    # Get conversation history
    history = get_session_history(session_id)
    
    # System prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant for a medical/dental practice. "
            "Answer questions based on the provided information about doctors, treatments, "
            "pricing, and practice policies. Be professional, clear, and helpful. "
            "If information is not available, be honest about it."
        )
    }
    
    # Add user message to history
    history.append({"role": "user", "content": user_input_text})
    
    # Keep only last 10 messages
    if len(history) > 10:
        history = history[-10:]
    
    # Convert to Gemini format
    gemini_messages = _convert_to_gemini_messages(system_prompt, history)
    
    try:
        # Generate response
        response = model.generate_content(
            gemini_messages,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
            }
        )
        
        assistant_response = response.text.strip() if hasattr(response, "text") else "[Empty response]"
        
        # Save to history
        history.append({"role": "assistant", "content": assistant_response})
        store[session_id] = history
        
        return assistant_response
    
    except Exception as e:
        print(f"Gemini error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


def clear_session_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in store:
        store[session_id] = []
        print(f"Session {session_id} history cleared.")


def get_all_sessions():
    """List all active sessions"""
    return list(store.keys())