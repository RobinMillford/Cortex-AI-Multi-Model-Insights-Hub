__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
import hashlib
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from helpers import extract_text_from_pdf, extract_text_from_url, process_content, create_vector_store, keyword_search
from langchain.schema import Document
from chain_setup import get_chain, ask_question, get_query_transform_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# --- Custom CSS for Dark Theme ---
st.markdown('''
<style>
    body {
        font-family: 'monospace';
        color: #40e723;
    }
    .stApp {
        background-color: #000000;
    }
    h1, h2, h3 {
        color: #40e723;
    }
    .st-emotion-cache-16txtl3 {
        background-color: #0d0d0d;
        border: 1px solid #40e723;
        border-radius: 10px;
    }
    .st-emotion-cache-163ttbj {
        background-color: #1a1a1a;
    }
    .st-emotion-cache-6q9sum.ef3psqc4 {
        background-color: #40e723;
        color: #000000;
    }
    .stChatMessage {
        background-color: #1a1a1a;
        border: 1px solid #40e723;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .st-expander {
        border-color: #40e723 !important;
    }
</style>
''', unsafe_allow_html=True)

# --- Model & Settings Data ---
models = {
    "llama-3.3-70b-versatile": {"advantages": "High accuracy in diverse scenarios.", "disadvantages": "Lower throughput.", "provider": "Meta"},
    "llama-3.1-8b-instant": {"advantages": "High-speed for real-time apps.", "disadvantages": "Less accurate for complex tasks.", "provider": "Meta"},
    "deepseek-r1-distill-llama-70b": {"advantages": "Low latency, no token limits.", "disadvantages": "Limited daily requests.", "provider": "DeepSeek"},
    "qwen/qwen3-32b": {"advantages": "Powerful 32B model for long-context.", "disadvantages": "Computationally intensive.", "provider": "Alibaba Cloud"},
    "openai/gpt-oss-120b": {"advantages": "120B params, browser search, code execution.", "disadvantages": "Slower speed.", "provider": "OpenAI"},
    "openai/gpt-oss-20b": {"advantages": "20B params, browser search, code execution.", "disadvantages": "Smaller model.", "provider": "OpenAI"},
}

# --- Sidebar UI ---
with st.sidebar:
    st.header("RAG Configuration")

    st.subheader("Content Source")
    input_method = st.radio("Input Method", ["PDF File", "URL"], label_visibility="collapsed")
    content = None
    collection_name = None
    if input_method == "PDF File":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            content = extract_text_from_pdf(uploaded_file)
            content_id = f"{uploaded_file.name}-{uploaded_file.size}"
            collection_name = hashlib.sha256(content_id.encode()).hexdigest()
    else:
        url = st.text_input("Article URL")
        if url:
            with st.spinner("Extracting text..."):
                content = extract_text_from_url(url)
                collection_name = hashlib.sha256(url.encode()).hexdigest()

    st.subheader("AI Models")
    selected_models = st.multiselect("Select models", options=list(models.keys()), default=["llama-3.3-70b-versatile"])
    if not selected_models:
        selected_models = ["llama-3.3-70b-versatile"]

    with st.expander("Model Details"):
        for model_name in selected_models:
            st.markdown(f"**{model_name}** (*{models[model_name]['provider']}*)")
            st.markdown(f"- **Pros**: {models[model_name]['advantages']}")
            st.markdown(f"- **Cons**: {models[model_name]['disadvantages']}")

    st.subheader("Parameters")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    chunk_size = st.slider("Chunk Size", 100, 3000, 300, 50)
    chunk_overlap = st.slider("Chunk Overlap", 10, 300, 50, 10)

    if st.button("Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main Page UI ---
st.title("📄 RAG Chatbot")
st.markdown("Chat with your documents.")

# --- Session State Initialization ---
if "last_collection_name" not in st.session_state:
    st.session_state.last_collection_name = None
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [{"role": "assistant", "content": "Upload a document to begin."}]
if "rag_conversation_history" not in st.session_state:
    st.session_state.rag_conversation_history = ""
if "rag_vector_store" not in st.session_state:
    st.session_state.rag_vector_store = None

# --- Chat Logic ---
if content and "Error" not in content:
    # Reset chat if new content is loaded
    if st.session_state.last_collection_name != collection_name:
        st.session_state.last_collection_name = collection_name
        st.session_state.rag_messages = [{"role": "assistant", "content": "Content processed. You can now ask questions."}]
        st.session_state.rag_conversation_history = ""
        st.session_state.rag_vector_store = None

    with st.sidebar:
        with st.expander("Content Preview", expanded=True):
            st.success("Content Extracted!")
            st.markdown(f"**Word Count:** {len(content.split())}")
            st.text(content[:250] + "...")
    
    # --- Vector Store Logic ---
    if st.session_state.rag_vector_store is None:
        with st.spinner("Processing content and creating vector store..."):
            client = chromadb.PersistentClient(path="./chroma_db")
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                st.success(f"Loading existing vector store: {collection_name}")
                st.session_state.rag_vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                    persist_directory="./chroma_db"
                )
            else:
                st.info(f"Creating new vector store: {collection_name}")
                chunks = process_content(content, chunk_size, chunk_overlap)
                st.session_state.rag_vector_store = create_vector_store(chunks, collection_name)

    if st.session_state.rag_vector_store:
        retriever = st.session_state.rag_vector_store.as_retriever()

        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask your document a question..."):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # --- Query Transformation ---
            with st.spinner("Transforming query..."):
                query_transform_chain = get_query_transform_chain(selected_models[0]) # Use the first selected model for transformation
                transformed_query = query_transform_chain.invoke({"chat_history": st.session_state.rag_conversation_history, "question": prompt})
            
            st.markdown(f"*Transformed Query:* `{transformed_query}`")

            # Get all documents from the vector store for keyword search
            # This might be inefficient for very large documents/collections
            all_docs_in_store = st.session_state.rag_vector_store.get(
                include=['documents']
            )['documents']
            # Convert list of strings to list of Document objects
            all_docs_in_store_obj = [Document(page_content=d) for d in all_docs_in_store]


            # Perform semantic search
            semantic_docs = retriever.get_relevant_documents(transformed_query)

            # Perform keyword search
            keyword_docs = keyword_search(transformed_query, all_docs_in_store_obj)

            # Combine results and remove duplicates
            combined_docs = {}
            for doc in semantic_docs + keyword_docs:
                combined_docs[doc.page_content] = doc # Use page_content as key for uniqueness

            final_docs = list(combined_docs.values())

            # Format context with numbered sources for LLM
            formatted_context_parts = []
            for i, doc in enumerate(final_docs):
                formatted_context_parts.append(f"[Source {i+1}: {doc.page_content}]")
            context_text = "\n\n".join(formatted_context_parts)

            with st.expander("Retrieved Context (Hybrid Search)"):
                if final_docs:
                    for i, doc in enumerate(final_docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:200] + "...")
                else:
                    st.info("No relevant context found.")

            for model in selected_models:
                with st.chat_message("assistant"):
                    st.markdown(f"*Response from {model}:*")
                    with st.spinner(f"Asking {model}..."):
                        chain = get_chain(model, temperature)
                        response, sources = ask_question(chain, prompt, context_text, chat_history=st.session_state.rag_conversation_history, final_docs=final_docs)
                        st.write(response)
                
                st.session_state.rag_messages.append({"role": "assistant", "content": response})
                st.session_state.rag_conversation_history += f"\nUser: {prompt}\nAssistant ({model}): {response}"

                if sources:
                    with st.expander(f"Sources for {model}"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(doc.page_content[:200] + "...")
else:
    st.info("Upload a document or provide a URL to get started.")
    if st.session_state.last_collection_name is not None:
        st.session_state.last_collection_name = None
        st.session_state.rag_messages = [{"role": "assistant", "content": "Upload a document to begin."}]