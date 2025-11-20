import streamlit as st
import chromadb
import hashlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from helpers import extract_text_from_pdf, extract_text_from_url, process_content, create_vector_store, keyword_search
from langchain.schema import Document
from chain_setup import get_chain, ask_question

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# --- Custom CSS for Premium Dark Theme ---
st.markdown('''
<style>
    /* General App Styling */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #00ff9d !important; /* Neon Green */
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    p, div, label {
        font-family: 'Segoe UI', sans-serif;
        color: #e0e0e0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #333;
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333;
        border-radius: 8px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00ff9d;
        box-shadow: 0 0 5px rgba(0, 255, 157, 0.5);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00ff9d, #00cc7a);
        color: #000000;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 157, 0.3);
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: #111;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    [data-testid="stChatMessageContent"] {
        color: #e0e0e0;
    }
    
    /* User Message Accent */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 4px solid #00ff9d;
    }
    
    /* Assistant Message Accent */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        border-left: 4px solid #00b8ff;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border-radius: 8px;
        color: #00ff9d;
    }
    
    /* Success/Info Messages */
    .stSuccess, .stInfo {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border-left: 4px solid #00ff9d;
    }
</style>
''', unsafe_allow_html=True)

# --- Session State Initialization ---
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [{"role": "assistant", "content": "👋 Hello! Upload a document or provide a URL to get started."}]
if "rag_conversation_history" not in st.session_state:
    st.session_state.rag_conversation_history = ""
if "rag_vector_store" not in st.session_state:
    st.session_state.rag_vector_store = None
if "last_collection_name" not in st.session_state:
    st.session_state.last_collection_name = None

# --- Sidebar: Configuration & Uploads ---
with st.sidebar:
    st.title("🛠️ Configuration")
    
    st.subheader("1. Data Sources")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    url_input = st.text_input("Or enter a URL", placeholder="https://example.com/article")
    
    st.subheader("2. Model Settings")
    selected_models = st.multiselect(
        "Select LLM Models",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "meta-llama/llama-guard-4-12b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        default=["llama-3.3-70b-versatile"]
    )
    
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7)
    
    process_btn = st.button("🚀 Process Content")

# --- Main Content Area ---
st.title("🧠 RAG Chatbot")
st.caption("Powered by Groq, LangChain & ChromaDB")

# --- Content Processing Logic ---
if process_btn:
    content_list = []
    
    # 1. Process Uploaded Files
    if uploaded_files:
        with st.status("Processing uploaded files...", expanded=True) as status:
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        text = extract_text_from_pdf(file)
                        if text:
                            content_list.append((text, file.name))
                            st.write(f"✅ Loaded: {file.name}")
                    elif file.type == "text/plain":
                        text = file.getvalue().decode("utf-8")
                        content_list.append((text, file.name))
                        st.write(f"✅ Loaded: {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
            status.update(label="Files processed!", state="complete", expanded=False)

    # 2. Process URL
    if url_input:
        with st.spinner(f"Scraping {url_input}..."):
            try:
                text = extract_text_from_url(url_input)
                if text:
                    content_list.append((text, url_input))
                    st.success(f"✅ Loaded URL: {url_input}")
            except Exception as e:
                st.error(f"Failed to load URL: {e}")

    if not content_list:
        st.warning("⚠️ Please upload a file or enter a valid URL.")
    else:
        # Create unique collection name based on content hash
        content_str = "".join([t[0] for t in content_list])
        collection_name = f"collection_{hashlib.md5(content_str.encode()).hexdigest()}"
        
        # Reset chat if content changed
        if st.session_state.last_collection_name != collection_name:
            st.session_state.last_collection_name = collection_name
            st.session_state.rag_messages = [{"role": "assistant", "content": "✅ Content processed! I'm ready to answer your questions."}]
            st.session_state.rag_conversation_history = ""
            st.session_state.rag_vector_store = None
            
            # Create/Load Vector Store
            with st.spinner("Creating vector embeddings..."):
                try:
                    chunks = process_content(content_list)
                    st.session_state.rag_vector_store = create_vector_store(chunks, collection_name)
                    st.success(f"🎉 Knowledge base ready! ({len(chunks)} chunks created)")
                except Exception as e:
                    st.error(f"Vector store creation failed: {e}")

# --- Chat Interface ---
# Display Chat History
for message in st.session_state.rag_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Logic
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.rag_vector_store:
        st.error("⚠️ Please upload and process documents first!")
    else:
        # Add User Message
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieval & Generation
        retriever = st.session_state.rag_vector_store.as_retriever()
        
        # 1. Retrieve Documents (Hybrid Search)
        # Note: For simplicity in this view, we use the standard retriever + keyword search from helpers
        # Ideally, we should fetch all docs for keyword search, but that can be heavy. 
        # We'll use the helper's logic if possible, or just standard retrieval for now to ensure speed.
        
        # Fetching all docs for keyword search (optimization: cache this if possible)
        all_docs_data = st.session_state.rag_vector_store.get(include=['documents', 'metadatas'])
        all_docs_obj = [Document(page_content=d, metadata=m) for d, m in zip(all_docs_data['documents'], all_docs_data['metadatas'])]
        
        semantic_docs = retriever.get_relevant_documents(prompt)
        keyword_docs = keyword_search(prompt, all_docs_obj, top_n=3)
        
        # Combine & Deduplicate
        combined_docs = {doc.page_content: doc for doc in semantic_docs + keyword_docs}
        final_docs = list(combined_docs.values())[:5] # Limit to top 5
        
        # Format Context
        context_text = "\n\n".join([f"[Source {i+1} ({d.metadata.get('source', 'Unknown')}): {d.page_content[:400]}...]" for i, d in enumerate(final_docs)])
        
        # Display Retrieved Context (Optional, for transparency)
        with st.expander("🔍 Retrieved Context"):
            for i, doc in enumerate(final_docs):
                st.markdown(f"**Source {i+1} ({doc.metadata.get('source', 'Unknown')}):**")
                st.caption(doc.page_content[:300] + "...")

        # 2. Generate Responses
        if not selected_models:
            st.error("Please select at least one model in the sidebar.")
        else:
            for model in selected_models:
                with st.chat_message("assistant"):
                    st.markdown(f"**🤖 {model}**")
                    placeholder = st.empty()
                    
                    with st.spinner(f"{model} is thinking..."):
                        try:
                            chain = get_chain(model, temperature)
                            response, sources = ask_question(
                                chain, 
                                prompt, 
                                context_text, 
                                chat_history=st.session_state.rag_conversation_history, 
                                final_docs=final_docs
                            )
                            placeholder.markdown(response)
                            
                            # Append to history
                            st.session_state.rag_messages.append({"role": "assistant", "content": f"**{model}:** {response}"})
                            st.session_state.rag_conversation_history += f"\nUser: {prompt}\nAssistant ({model}): {response}"
                            
                            # Show Sources specifically for this answer if needed
                            # (Already shown in context expander, but can be specific here)
                            
                        except Exception as e:
                            placeholder.error(f"Error generating response: {e}")