# import sys
# import os
# from pathlib import Path

# # Add project root to path - go up TWO levels from generator.py
# # generator.py is in src/llm/, so we need to go up to project root


# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))


# from dotenv import load_dotenv
# load_dotenv(dotenv_path=r"C:\Users\hasan\Rafi_SAA\practice_project_1\Intelligent_Document_parsing_chatbot\.env")
# from openai import OpenAI

# from src.llm.augmented_prompt import augmented_prompt

# # Initializing the OpenAI client with Hugging Face endpoint for LLaMA
# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.environ["HF_TOKEN_READ"],
# )

# # In-memory conversation store (session_id -> list of messages)
# store = {}


# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = []
    
#     return store[session_id]


# def generate_llm_response(query, retrieved_docs, session_id="default_session", max_docs=4):
    
#     # Build augmented context prompt
#     user_input_text = augmented_prompt(query, retrieved_docs, max_docs)
    
#     # Get conversation history for this session
#     history = get_session_history(session_id)
    
#     # System prompt
#     system_prompt = {
#         "role": "system",
#         "content": "You are a helpful assistant that analyzes documents and answers questions based on retrieved information and conversation context. Provide accurate, concise responses. If the user query is not related to the documents, just respond with your general knowledge base. And before generating an asnwer, always ask the user which kind of information they are looking for, then responsd accordingly."
#     }
    
#     # Add current user message to history
#     history.append({
#         "role": "user",
#         "content": user_input_text
#     })
    

#     max_history_length = 10
#     if len(history) > max_history_length:
#         messages = [system_prompt] + history[-max_history_length:]
#     else:
#         messages = [system_prompt] + history
    
#     try:
#         # Call LLaMA model via HuggingFace
#         completion = client.chat.completions.create(
#             model="meta-llama/Llama-3.1-8B-Instruct:novita",
#             messages=messages,
#             max_tokens=1024,
#             temperature=0.3,  # Lower temperature for more factual responses
#         )
        
#         # Extract response
#         assistant_response = completion.choices[0].message.content
        
#         # Add assistant response to history
#         history.append({
#             "role": "assistant",
#             "content": assistant_response
#         })
        
#         # Update store
#         store[session_id] = history
        
#         return assistant_response
        
#     except Exception as e:
#         print(f"Error generating response: {e}")
#         return f"Sorry, I encountered an error: {str(e)}"


# def clear_session_history(session_id: str):

#     if session_id in store:
#         store[session_id] = []
#         print(f"Session {session_id} history cleared.")


# def get_all_sessions():
#returing the list of all active session ids
#     return list(store.keys())

#----------------------------------------------------------------

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=str(project_root / ".env"))
import google.generativeai as genai
from src.llm.augmented_prompt import augmented_prompt
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# In-memory conversation store
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = []
    return store[session_id]


def _convert_to_gemini_messages(system_prompt, history):
    gemini_messages = []
    gemini_messages.append({
        "role": "user",  # Gemini does not have 'system' role
        "parts": [system_prompt["content"]]
    })

    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_messages.append({
            "role": role,
            "parts": [msg["content"]]
        })

    return gemini_messages


def generate_llm_response(query, retrieved_docs, session_id="default_session", max_docs=4):
    user_input_text = augmented_prompt(query, retrieved_docs, max_docs)
    history = get_session_history(session_id)

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant that analyzes documents and answers questions "
            "based on retrieved information and conversation context. Provide accurate, "
            "concise responses. If the user query is not related to the documents, "
            "respond using general knowledge. Always ask the user which type of information "
            "they are looking for before giving the final answer."
        )
    }

    history.append({"role": "user", "content": user_input_text})
    max_history_length = 10
    if len(history) > max_history_length:
        trimmed_history = history[-max_history_length:]
    else:
        trimmed_history = history

    # Convert to Gemini format
    gemini_messages = _convert_to_gemini_messages(system_prompt, trimmed_history)

    try:
        # Generate response using Gemini
        response = model.generate_content(
            gemini_messages,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
            }
        )

        assistant_response = response.text.strip() if hasattr(response, "text") else "[Empty response]"

        # Save assistant response in session history
        history.append({
            "role": "assistant",
            "content": assistant_response
        })

        store[session_id] = history
        return assistant_response

    except Exception as e:
        print(f"Gemini error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


def clear_session_history(session_id: str):
    """Clear only the requested session's history."""
    if session_id in store:
        store[session_id] = []
        print(f"Session {session_id} history cleared.")


def get_all_sessions():
    """List all active sessions."""
    return list(store.keys())
