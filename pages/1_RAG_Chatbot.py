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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* General App Styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #050505 50%, #0d0d0d 100%);
        color: #e8e8e8;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Typography */
    h1 {
        background: linear-gradient(135deg, #00ff9d 0%, #00d4aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 0 30px rgba(0, 255, 157, 0.3);
    }
    
    h2, h3 {
        color: #00ff9d !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    p, div, label {
        font-family: 'Inter', sans-serif;
        color: #e8e8e8;
        font-weight: 400;
    }

    /* Sidebar Styling with Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 10, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 255, 157, 0.1);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(26, 26, 26, 0.8);
        color: #ffffff;
        border: 1.5px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00ff9d;
        box-shadow: 0 0 0 3px rgba(0, 255, 157, 0.15),
                    0 0 20px rgba(0, 255, 157, 0.2);
        background: rgba(26, 26, 26, 1);
        transform: translateY(-1px);
    }

    /* Buttons with Enhanced Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #00ff9d 0%, #00cc7a 100%);
        color: #000000;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.2);
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 255, 157, 0.4);
        background: linear-gradient(135deg, #00ffaa 0%, #00d47a 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Chat Messages with Glassmorphism */
    .stChatMessage {
        background: rgba(17, 17, 17, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        background: rgba(17, 17, 17, 0.8);
        border-color: rgba(0, 255, 157, 0.2);
        transform: translateX(4px);
    }
    
    [data-testid="stChatMessageContent"] {
        color: #e8e8e8;
        line-height: 1.6;
    }
    
    /* User Message with Neon Accent */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 3px solid #00ff9d;
        background: linear-gradient(90deg, rgba(0, 255, 157, 0.05) 0%, rgba(17, 17, 17, 0.6) 100%);
    }
    
    /* Assistant Message with Blue Accent */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid #00b8ff;
        background: linear-gradient(90deg, rgba(0, 184, 255, 0.05) 0%, rgba(17, 17, 17, 0.6) 100%);
    }

    /* Chat Input */
    .stChatInputContainer {
        border-top: 1px solid rgba(0, 255, 157, 0.1);
        background: rgba(10, 10, 10, 0.8);
        backdrop-filter: blur(10px);
    }

    /* Expanders with Enhanced Style */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.8) 0%, rgba(20, 20, 20, 0.8) 100%);
        border: 1px solid rgba(0, 255, 157, 0.2);
        border-radius: 12px;
        color: #00ff9d;
        font-weight: 500;
        padding: 12px 16px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(0, 255, 157, 0.1) 0%, rgba(20, 20, 20, 0.9) 100%);
        border-color: #00ff9d;
        box-shadow: 0 4px 12px rgba(0, 255, 157, 0.15);
    }
    
    .streamlit-expanderContent {
        background: rgba(15, 15, 15, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: none;
        border-radius: 0 0 12px 12px;
    }
    
    /* Success/Info/Warning Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 255, 157, 0.1) 0%, rgba(0, 204, 122, 0.05) 100%);
        color: #e8e8e8;
        border-left: 4px solid #00ff9d;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0, 255, 157, 0.1);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 184, 255, 0.1) 0%, rgba(0, 150, 255, 0.05) 100%);
        color: #e8e8e8;
        border-left: 4px solid #00b8ff;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0, 184, 255, 0.1);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.05) 100%);
        color: #e8e8e8;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 12px 16px;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.1) 0%, rgba(255, 23, 68, 0.05) 100%);
        color: #e8e8e8;
        border-left: 4px solid #ff5252;
        border-radius: 8px;
        padding: 12px 16px;
    }
    
    /* Select Boxes and Dropdowns */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(26, 26, 26, 0.8);
        border: 1.5px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: rgba(0, 255, 157, 0.5);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: rgba(0, 255, 157, 0.2);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #00ff9d 0%, #00cc7a 100%);
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(26, 26, 26, 0.6);
        border: 2px dashed rgba(0, 255, 157, 0.3);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #00ff9d;
        background: rgba(26, 26, 26, 0.8);
        box-shadow: 0 4px 20px rgba(0, 255, 157, 0.1);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00ff9d 0%, #00cc7a 100%);
        box-shadow: 0 0 15px rgba(0, 255, 157, 0.5);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00ff9d !important;
    }
    
    /* Status Container */
    [data-testid="stStatusWidget"] {
        background: rgba(26, 26, 26, 0.8);
        border: 1px solid rgba(0, 255, 157, 0.2);
        border-radius: 12px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(10, 10, 10, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00ff9d 0%, #00cc7a 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00ffaa 0%, #00d47a 100%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.3s ease-out;
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
                            
                        except Exception as e:
                            placeholder.error(f"Error generating response: {e}")