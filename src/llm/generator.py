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
from langchain_community.chat_models import ChatOpenAI
# from langchain_community.chat_message_histories import ChatMessageHistory
from src.llm.augmented_prompt import augmented_prompt
# from langchain.chains import ConcersationChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

API_KEY = os.getenv("GPT_github_access_token")

doc_parser_llm = ChatOpenAI(
    model="openai/gpt-4o",
    openai_api_key=API_KEY,
    openai_api_base="https://models.github.ai/inference"
)

# memory = InMemoryChatMessageHistory

store = {}



def get_session_history( session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
    return store[session_id]

# def generate_llm_response(query, retrieved_docs, session_id="default_session", max_docs=4):
   
#     # Build augmented context prompt
#     prompt_text = augmented_propmt(query, retrieved_docs, max_docs)

#     # Create a new conversation chain (with history)
#     # conversation = RunnableWithMessageHistory(
#     #     doc_parser_llm,
#     #     get_session_history,
#     #     input_messages_key='input',
#     #     history_messages_key='history',
#     #     verbose=True
#     # )

#     # Invoke the LLM with the prompt
#     # response = conversation.invoke(
#     #     {"input": prompt_text},
#     #     config={"configurable": {"session_id": session_id}}
#     # )
#     response = doc_parser_llm.invoke(prompt_text)


#     # Return only the message text
#     return response.content if hasattr(response, "content") else str(response)

def generate_llm_response(query, retrieved_docs, session_id="default_session", max_docs=4):
    """
    Generate response using retrieved docs + conversation memory.
    """

    # Static system prompt
    # system_prompt = (
    #     "You are an intelligent assistant that uses retrieved context and chat history. "
    #     "If context doesn't contain the answer, say 'I'm not sure about the  data!!'."
    # )
    # Get user input with augmented context
    user_input_text = augmented_prompt(query, retrieved_docs, max_docs)

    # Build prompt template with memory placeholder
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides factual and concise answers using the given information and chat history."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # LLM chain
    chain = prompt_template | doc_parser_llm

    # Wrap chain in memory
    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    # Invoke LLM
    response = conversation.invoke(
        {"input": user_input_text},
        config={"configurable": {"session_id": session_id}}
    )

    return response.content if hasattr(response, "content") else str(response)


# print("hello")


# response = converstaion.predict(input = augmented_propmt)

# print(f"\nLLM Response: {response}")
# if __name__ == "__main__":
#     user_query = input("Enter your question: ")
#     response = conversation.invoke(
#         {"input": user_query},
#         config={"configurable": {"session_id": "user1"}}
#     )
#     print("Response:", response)