import streamlit as st
import chromadb
import hashlib
import os
import pickle
from PIL import Image
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from helpers import extract_text_from_pdf, extract_text_from_url, process_content, create_vector_store, hybrid_multimodal_search, format_multimodal_results
from multimodal_helpers import FreeMultimodalEmbedder, save_multimodal_store, load_multimodal_store, get_collection_hash
from langchain.schema import Document
from chain_setup import get_chain, ask_question
import base64
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🖼️",
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
    
    /* Image Container */
    .image-container {
        border: 2px solid #00ff9d;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #111;
    }
</style>
''', unsafe_allow_html=True)

# --- Session State Initialization ---
if "multimodal_last_collection_name" not in st.session_state:
    st.session_state.multimodal_last_collection_name = None
if "multimodal_messages" not in st.session_state:
    st.session_state.multimodal_messages = [{"role": "assistant", "content": "Upload a document to begin multimodal analysis."}]
if "multimodal_conversation_history" not in st.session_state:
    st.session_state.multimodal_conversation_history = ""
if "multimodal_vector_store" not in st.session_state:
    st.session_state.multimodal_vector_store = None
if "multimodal_store" not in st.session_state:
    st.session_state.multimodal_store = None
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
if "multimodal_embedder" not in st.session_state:
    st.session_state.multimodal_embedder = None
if "current_vision_model" not in st.session_state:
    st.session_state.current_vision_model = None

# --- Model & Settings Data ---
models = {
    "llama-3.3-70b-versatile": {"advantages": "High accuracy in diverse scenarios.", "disadvantages": "Lower throughput.", "provider": "Meta"},
    "llama-3.1-8b-instant": {"advantages": "High-speed for real-time apps.", "disadvantages": "Less accurate for complex tasks.", "provider": "Meta"},
    "meta-llama/llama-guard-4-12b": {"advantages": "Optimized for safety and guardrailing.", "disadvantages": "Specialized for safety, not general chat.", "provider": "Meta"},
    "openai/gpt-oss-120b": {"advantages": "120B params, browser search, code execution.", "disadvantages": "Slower speed.", "provider": "OpenAI"},
    "openai/gpt-oss-20b": {"advantages": "20B params, browser search, code execution.", "disadvantages": "Smaller model.", "provider": "OpenAI"},
}

# Vision model options
vision_models = {
    "blip": {"name": "BLIP (Fast)", "description": "Fast image captioning, good for basic analysis"},
    "blip2": {"name": "BLIP-2 (Advanced)", "description": "Advanced understanding, slower but more detailed"},
    "git": {"name": "GIT (Detailed)", "description": "Detailed image descriptions, best for complex images"}
}

# --- Sidebar UI ---
with st.sidebar:
    st.header("🖼️ Multimodal RAG Configuration")

    st.subheader("Content Source")
    input_method = st.radio("Input Method", ["PDF File", "URL"], label_visibility="collapsed")
    
    st.subheader("Vision Settings")
    vision_model = st.selectbox("Vision Model", options=list(vision_models.keys()), 
                               format_func=lambda x: vision_models[x]["name"])
    st.caption(vision_models[vision_model]["description"])
    
    # Processing options
    extract_images = st.checkbox("Extract and analyze images", value=True)
    max_images = st.slider("Max images to process", 1, 10, 5, 1)
    
    content = ""
    content_for_processing = [] # List of (text, source) tuples
    images = []
    collection_name = None
    
    if input_method == "PDF File":
        uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            content_text_list = []
            all_images = []
            
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    if extract_images:
                        file_content, file_images = extract_text_from_pdf(uploaded_file, extract_images=True)
                        if "Error" not in file_content:
                            content_text_list.append(file_content)
                            content_for_processing.append((file_content, uploaded_file.name))
                            all_images.extend(file_images)
                        else:
                            st.error(f"Error processing {uploaded_file.name}: {file_content}")
                    else:
                        file_content = extract_text_from_pdf(uploaded_file)
                        if "Error" not in file_content:
                            content_text_list.append(file_content)
                            content_for_processing.append((file_content, uploaded_file.name))
                        else:
                            st.error(f"Error processing {uploaded_file.name}: {file_content}")
            
            if content_text_list:
                content = "\n\n".join(content_text_list)
                images = all_images[:max_images]  # Limit total images
                
                # Create hash from all file names and sizes
                files_id = "".join([f"{f.name}-{f.size}" for f in uploaded_files])
                collection_name = get_collection_hash(files_id, has_images=extract_images and len(images) > 0)
            
    else:
        url = st.text_input("Article URL")
        if url:
            with st.spinner("Extracting text..."):
                content = extract_text_from_url(url)
                if "Error" not in content:
                    content_for_processing = [(content, url)]
                    collection_name = get_collection_hash(url, has_images=False)
                else:
                    st.error(f"Error loading URL: {content}")
                    content = ""

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
    num_retrieved_chunks = st.slider("Number of Retrieved Chunks", 1, 10, 5, 1)
    max_chars_per_chunk = st.slider("Max Characters per Chunk (for LLM)", 100, 2000, 500, 100)
    include_images_in_search = st.checkbox("Include images in search", value=True)

    if st.button("Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # --- Information Section in Sidebar ---
    st.markdown("---")
    st.markdown("### 🖼️ About Multimodal RAG")
    st.markdown("""
    This enhanced RAG system can analyze both **text and visual content** in your documents:

    **🎯 What it analyzes:**
    - 📊 Charts and graphs
    - 📈 Data visualizations  
    - 🖼️ Images and diagrams
    - 📋 Infographics
    - 🏗️ Technical drawings
    - 📸 Screenshots

    **🔧 How it works:**
    1. Extracts images from PDFs
    2. Analyzes visual content using AI vision models
    3. Creates unified embeddings for text and images
    4. Searches across both modalities simultaneously
    5. Provides context-aware responses with visual citations
    
    **💡 Free & Local Processing:**
    - Uses open-source vision models (BLIP, BLIP-2, GIT)
    - No paid API dependencies
    - All processing done locally for privacy
    """)

# --- Main Page UI ---
st.title("🖼️ Multimodal RAG Chatbot")
st.markdown("Chat with your documents **including images, charts, and infographics**!")

# --- Chat Logic ---
if content and "Error" not in content:
    # Reset chat if new content is loaded
    if st.session_state.multimodal_last_collection_name != collection_name:
        st.session_state.multimodal_last_collection_name = collection_name
        st.session_state.multimodal_messages = [{"role": "assistant", "content": "Content processed. You can now ask questions about text and images."}]
        st.session_state.multimodal_conversation_history = ""
        st.session_state.multimodal_vector_store = None
        st.session_state.multimodal_store = None
        st.session_state.processed_images = []

    with st.sidebar:
        with st.expander("Content Preview", expanded=True):
            st.success("Content Extracted!")
            st.markdown(f"**Word Count:** {len(content.split())}")
            if images:
                st.markdown(f"**Images Found:** {len(images)}")
            st.text(content[:250] + "...")
            
        # Show extracted images
        if images:
            with st.expander(f"📷 Extracted Images ({len(images)})", expanded=False):
                for i, img_data in enumerate(images[:3]):  # Show first 3
                    st.markdown(f"**Image {i+1}** (Page {img_data['page']})")
                    st.image(img_data['image'], width=200)
                if len(images) > 3:
                    st.caption(f"... and {len(images) - 3} more images")
    
    # --- Vector Store Logic ---
    if st.session_state.multimodal_vector_store is None:
        with st.spinner("Processing content and creating multimodal vector store..."):
            # Create traditional text vector store
            client = chromadb.PersistentClient(path="./chroma_db")
            collections = client.list_collections()
            # ChromaDB v0.6.0+ returns collection names directly
            try:
                # Try new API (v0.6.0+) - returns strings directly
                collection_names = collections
            except:
                # Fallback to old API (< v0.6.0) - returns collection objects
                collection_names = [c.name for c in collections]
            
            text_collection_name = collection_name.replace("_multimodal", "")
            
            if text_collection_name in collection_names:
                st.success(f"Loading existing text vector store: {text_collection_name}")
                st.session_state.multimodal_vector_store = Chroma(
                    collection_name=text_collection_name,
                    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                    persist_directory="./chroma_db"
                )
            else:
                st.info(f"Creating new text vector store: {text_collection_name}")
                chunks = process_content(content_for_processing)
                st.session_state.multimodal_vector_store = create_vector_store(chunks, text_collection_name)
            
            # Create multimodal store if images are present
            if images and extract_images:
                multimodal_store_path = f"./multimodal_stores/{collection_name}.pkl"
                os.makedirs("./multimodal_stores", exist_ok=True)
                
                if os.path.exists(multimodal_store_path):
                    st.success("Loading existing multimodal store...")
                    st.session_state.multimodal_store = load_multimodal_store(multimodal_store_path)
                    st.session_state.processed_images = images
                else:
                    st.info("Creating new multimodal store...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Initialize or reuse embedder
                        if (st.session_state.multimodal_embedder is None or 
                            st.session_state.current_vision_model != vision_model):
                            status_text.text("Initializing vision models...")
                            st.session_state.multimodal_embedder = FreeMultimodalEmbedder(vision_model=vision_model)
                            st.session_state.current_vision_model = vision_model
                        
                        embedder = st.session_state.multimodal_embedder
                        progress_bar.progress(0.2)
                        
                        # Get text chunks
                        chunks = process_content(content_for_processing)
                        text_contents = [chunk.page_content for chunk in chunks]
                        progress_bar.progress(0.4)
                        
                        # Create multimodal embeddings
                        status_text.text("Creating unified embeddings...")
                        st.session_state.multimodal_store = embedder.create_unified_embeddings(text_contents, images)
                        progress_bar.progress(0.8)
                        
                        # Save multimodal store
                        status_text.text("Saving multimodal store...")
                        save_multimodal_store(st.session_state.multimodal_store, multimodal_store_path)
                        st.session_state.processed_images = images
                        progress_bar.progress(1.0)
                        
                        status_text.empty()
                        progress_bar.empty()
                        st.success("Multimodal store created successfully!")
                        
                    except Exception as e:
                        st.error(f"Error creating multimodal store: {e}")
                        st.session_state.multimodal_store = None
            else:
                st.session_state.multimodal_store = None

    # --- Chat Interface ---
    for message in st.session_state.multimodal_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.multimodal_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieval & Generation
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            # 1. Retrieve Context
            context_text = ""
            retrieved_images = []
            
            # Text Retrieval
            if st.session_state.multimodal_vector_store:
                retriever = st.session_state.multimodal_vector_store.as_retriever(search_kwargs={"k": num_retrieved_chunks})
                docs = retriever.get_relevant_documents(prompt)
                context_text = "\n\n".join([d.page_content for d in docs])
            
            # Image Retrieval (if enabled)
            if st.session_state.multimodal_store and include_images_in_search:
                try:
                    embedder = st.session_state.multimodal_embedder
                    results = hybrid_multimodal_search(
                        st.session_state.multimodal_store,
                        embedder,
                        prompt,
                        top_k=3
                    )
                    formatted_results = format_multimodal_results(results)
                    
                    # Add image descriptions to context
                    for res in formatted_results:
                        if res['type'] == 'image':
                            context_text += f"\n[Image Context: {res['content']}]"
                            retrieved_images.append(res['image_obj'])
                except Exception as e:
                    st.warning(f"Multimodal search failed: {e}")

            # 2. Generate Response
            if not selected_models:
                st.error("Please select at least one model.")
            else:
                # Use the first selected model for now (or loop through them)
                model = selected_models[0] 
                st.markdown(f"**🤖 {model}**")
                
                try:
                    chain = get_chain(model, temperature)
                    response, _ = ask_question(
                        chain, 
                        prompt, 
                        context_text, 
                        chat_history=st.session_state.multimodal_conversation_history
                    )
                    
                    placeholder.markdown(response)
                    
                    # Display retrieved images if relevant
                    if retrieved_images:
                        with st.expander("Related Images"):
                            for img in retrieved_images:
                                st.image(img, width=300)
                    
                    st.session_state.multimodal_messages.append({"role": "assistant", "content": response})
                    st.session_state.multimodal_conversation_history += f"\nUser: {prompt}\nAssistant: {response}"
                    
                except Exception as e:
                    placeholder.error(f"Error generating response: {e}")