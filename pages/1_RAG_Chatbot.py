import streamlit as st
from helpers import extract_text_from_pdf, extract_text_from_url, process_content, create_vector_store
from chain_setup import get_chain, ask_question

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
    "deepseek-r1-distill-llama-70b": {"advantages": "Low latency, no token limits.", "disadvantages": "Limited daily requests.", "provider": "DeepSeek"},
    "qwen-2.5-32b": {"advantages": "Long-context comprehension.", "disadvantages": "Computationally intensive.", "provider": "Alibaba Cloud"},
    "gemma2-9b-it": {"advantages": "High throughput, fast inference.", "disadvantages": "Limited versatility.", "provider": "Google"},
    "llama-3.1-8b-instant": {"advantages": "High-speed, for real-time apps.", "disadvantages": "Less accurate for complex tasks.", "provider": "Meta"},
    "llama-3.3-70b-versatile": {"advantages": "High accuracy in diverse scenarios.", "disadvantages": "Lower throughput.", "provider": "Meta"},
    "llama3-70b-8192": {"advantages": "Ideal for detailed research.", "disadvantages": "Moderate speed.", "provider": "Meta"},
    "llama3-8b-8192": {"advantages": "High-speed with long-context.", "disadvantages": "Less accurate for complex reasoning.", "provider": "Meta"},
    "mistral-saba-24b": {"advantages": "Strong multi-turn conversation.", "disadvantages": "Limited token capacity.", "provider": "Mistral AI"},
    "mixtral-8x7b-32768": {"advantages": "Supports long documents.", "disadvantages": "Lower token throughput.", "provider": "Mistral AI"},
}

# --- Sidebar UI ---
with st.sidebar:
    st.header("RAG Configuration")

    st.subheader("Content Source")
    input_method = st.radio("Input Method", ["PDF File", "URL"], label_visibility="collapsed")
    content = None
    content_id = None
    if input_method == "PDF File":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            content = extract_text_from_pdf(uploaded_file)
            content_id = f"{uploaded_file.name}-{uploaded_file.size}"
    else:
        url = st.text_input("Article URL")
        if url:
            with st.spinner("Extracting text..."):
                content = extract_text_from_url(url)
                content_id = url

    st.subheader("AI Models")
    selected_models = st.multiselect("Select models", options=list(models.keys()), default=["llama3-70b-8192"])
    if not selected_models:
        selected_models = ["llama3-70b-8192"]

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
if "last_content_id" not in st.session_state:
    st.session_state.last_content_id = None
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [{"role": "assistant", "content": "Upload a document to begin."}]
if "rag_conversation_history" not in st.session_state:
    st.session_state.rag_conversation_history = ""
if "rag_vector_store" not in st.session_state:
    st.session_state.rag_vector_store = None

# --- Chat Logic ---
if content and "Error" not in content:
    # Reset chat if new content is loaded
    if st.session_state.last_content_id != content_id:
        st.session_state.last_content_id = content_id
        st.session_state.rag_messages = [{"role": "assistant", "content": "Content processed. You can now ask questions."}]
        st.session_state.rag_conversation_history = ""
        st.session_state.rag_vector_store = None

    with st.sidebar:
        with st.expander("Content Preview", expanded=True):
            st.success("Content Extracted!")
            st.markdown(f"**Word Count:** {len(content.split())}")
            st.text(content[:250] + "...")
    
    chunks = process_content(content, chunk_size, chunk_overlap)
    if chunks:
        if st.session_state.rag_vector_store is None:
            with st.spinner("Creating vector store..."):
                st.session_state.rag_vector_store = create_vector_store(chunks)
        
        retriever = st.session_state.rag_vector_store.as_retriever()

        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask your document a question..."):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            context_text = " ".join([doc.page_content for doc in retriever.get_relevant_documents(prompt)])
            with st.expander("Retrieved Context"):
                st.info(context_text or "No relevant context found.")

            for model in selected_models:
                with st.chat_message("assistant"):
                    st.markdown(f"*Response from {model}:*", unsafe_allow_html=True)
                    with st.spinner(f"Asking {model}..."):
                        chain = get_chain(model, temperature)
                        response = ask_question(chain, prompt, context_text, chat_history=st.session_state.rag_conversation_history)
                        st.write(response)
                
                st.session_state.rag_messages.append({"role": "assistant", "content": response})
                st.session_state.rag_conversation_history += f"\nUser: {prompt}\nAssistant ({model}): {response}"
else:
    st.info("Upload a document or provide a URL to get started.")
    # Reset if content is cleared
    if st.session_state.last_content_id is not None:
        st.session_state.last_content_id = None
        st.session_state.rag_messages = [{"role": "assistant", "content": "Upload a document to begin."}]
