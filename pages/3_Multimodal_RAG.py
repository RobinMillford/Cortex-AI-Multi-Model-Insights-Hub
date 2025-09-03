__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
    .image-container {
        border: 2px solid #40e723;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
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
    
    content = None
    images = []
    collection_name = None
    
    if input_method == "PDF File":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            if extract_images:
                content, images = extract_text_from_pdf(uploaded_file, extract_images=True)
                images = images[:max_images]  # Limit number of images
            else:
                content = extract_text_from_pdf(uploaded_file)
            content_id = f"{uploaded_file.name}-{uploaded_file.size}"
            collection_name = get_collection_hash(content_id, has_images=extract_images and len(images) > 0)
    else:
        url = st.text_input("Article URL")
        if url:
            with st.spinner("Extracting text..."):
                content = extract_text_from_url(url)
                collection_name = get_collection_hash(url, has_images=False)

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
                chunks = process_content(content)
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
                        chunks = process_content(content)
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
                        progress_bar.empty()
                        status_text.empty()

    # --- Chat Interface ---
    if st.session_state.multimodal_vector_store:
        # Display chat messages
        for message in st.session_state.multimodal_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask about your document (text and images)..."):
            st.session_state.multimodal_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Get all documents from the vector store for keyword search
            all_docs_in_store = st.session_state.multimodal_vector_store.get(
                include=['documents']
            )['documents']
            all_docs_in_store_obj = [Document(page_content=d) for d in all_docs_in_store]

            # Perform hybrid multimodal search
            final_docs = hybrid_multimodal_search(
                query=prompt,
                vector_store=st.session_state.multimodal_vector_store,
                chunks=all_docs_in_store_obj,
                multimodal_store=st.session_state.multimodal_store,
                top_n=num_retrieved_chunks,
                include_images=include_images_in_search
            )

            # Format context for LLM
            formatted_context_parts = []
            for i, doc in enumerate(final_docs):
                truncated_content = doc.page_content[:max_chars_per_chunk]
                if len(doc.page_content) > max_chars_per_chunk:
                    truncated_content += "..."
                
                # Add metadata for image sources
                if hasattr(doc, 'metadata') and doc.metadata.get('type') == 'image':
                    truncated_content = f"[IMAGE CONTENT: {truncated_content}]"
                
                formatted_context_parts.append(f"[Source {i+1}: {truncated_content}]")
            
            context_text = "\n\n".join(formatted_context_parts)

            # Show retrieved context
            with st.expander("Retrieved Context (Multimodal Hybrid Search)"):
                if final_docs:
                    for i, doc in enumerate(final_docs):
                        if hasattr(doc, 'metadata') and doc.metadata.get('type') == 'image':
                            st.markdown(f"**🖼️ Image Source {i+1}:**")
                            image_data = doc.metadata.get('image_data')
                            if image_data:
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(image_data['image'], width=150)
                                with col2:
                                    st.text(doc.page_content[:300] + "...")
                        else:
                            st.markdown(f"**📄 Text Source {i+1}:**")
                            st.text(doc.page_content[:500] + "...")
                else:
                    st.info("No relevant context found.")

            # Generate responses from selected models
            for model in selected_models:
                with st.chat_message("assistant"):
                    st.markdown(f"*Response from {model}:*")
                    with st.spinner(f"Asking {model}..."):
                        chain = get_chain(model, temperature)
                        response, sources = ask_question(
                            chain, prompt, context_text, 
                            chat_history=st.session_state.multimodal_conversation_history, 
                            final_docs=final_docs
                        )
                        st.write(response)
                
                st.session_state.multimodal_messages.append({"role": "assistant", "content": response})
                
                # Update conversation history
                MAX_HISTORY_CHARS = 1500
                st.session_state.multimodal_conversation_history += f"\nUser: {prompt}\nAssistant ({model}): {response}"
                if len(st.session_state.multimodal_conversation_history) > MAX_HISTORY_CHARS:
                    st.session_state.multimodal_conversation_history = st.session_state.multimodal_conversation_history[-(MAX_HISTORY_CHARS // 2):]

                # Show sources with images
                if sources:
                    with st.expander(f"Sources for {model}"):
                        for i, doc in enumerate(sources):
                            if hasattr(doc, 'metadata') and doc.metadata.get('type') == 'image':
                                st.markdown(f"**🖼️ Image Source {i+1}:**")
                                image_data = doc.metadata.get('image_data')
                                if image_data:
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.image(image_data['image'], width=100)
                                    with col2:
                                        st.text(doc.page_content[:200] + "...")
                            else:
                                st.markdown(f"**📄 Text Source {i+1}:**")
                                st.text(doc.page_content[:200] + "...")

else:
    st.info("Upload a document or provide a URL to get started with multimodal analysis.")
    if st.session_state.multimodal_last_collection_name is not None:
        st.session_state.multimodal_last_collection_name = None
        st.session_state.multimodal_messages = [{"role": "assistant", "content": "Upload a document to begin multimodal analysis."}]