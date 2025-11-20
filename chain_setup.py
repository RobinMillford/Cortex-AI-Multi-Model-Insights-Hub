from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Rebuild ChatGroq model to resolve Pydantic issue
ChatGroq.model_rebuild()

def get_chain(model_name, temperature=0.7):
    """
    Creates a chain with the specified model and temperature.
    """
    print("Initializing ChatGroq with model:", model_name)  # Debug
    # Initialize ChatGroq
    lama = ChatGroq(
        temperature=temperature,
        groq_api_key=groq_api_key,
        model_name=model_name,
    )
    print("ChatGroq initialized successfully")  # Debug

    # Output parser
    parser = StrOutputParser()

    # ChatPromptTemplate
    template = """
    Answer the question based ONLY on the context and chat history below.
    If you can't answer the question from the context, reply "I cannot answer this question based on the provided documents."
    Do not use outside knowledge.
    Cite the source documents by referencing their numbered content, e.g., [Source 1: ...].

    Chat History: {chat_history}

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define chain
    chain = prompt | lama | parser
    return chain

def ask_question(chain, question, context, chat_history="", final_docs=None):
    """
    Generate a response to the user's question using the provided chain.
    """
    formatted_input = {"chat_history": chat_history, "context": context, "question": question}
    response = chain.invoke(formatted_input)
    return response, final_docs # Return response and final_docs

def get_query_transform_chain(model_name, temperature=0.0):
    """
    Creates a chain for query transformation.
    """
    llm = ChatGroq(
        temperature=temperature,
        groq_api_key=groq_api_key,
        model_name=model_name,
    )
    
    template = """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}

    Follow Up Question: {question}

    Standalone question:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm | StrOutputParser()
    return chain
