import streamlit as st
from helpers import extract_text_from_pdf, extract_text_from_url, process_content, create_vector_store
from chain_setup import get_chain, ask_question
from streamlit import cache_data, cache_resource

# Streamlit App Configuration
st.set_page_config(page_title="Multi-Model RAG Chatbot 📄🔍", page_icon="🤖", layout="wide")
st.title("Multi-Model RAG-Powered Article Chatbot 📄🔍")

# Sidebar Settings
st.sidebar.title("Settings")

# Optimized Model List for PDF & Research Article RAG
models = {
    # ✅ DeepSeek Models (Efficient for RAG, Optimized for Speed & Scalability)
    "deepseek-r1-distill-llama-70b": {
        "requests_per_minute": 30,
        "requests_per_day": 1_000,
        "tokens_per_minute": 6_000,
        "tokens_per_day": None,  # Unlimited token capacity
        "advantages": "Highly optimized for low latency with no token limits, making it ideal for large-scale deployments.",
        "disadvantages": "Limited daily requests compared to other models.",
    },

    # ✅ Alibaba Cloud Qwen Models (Top-tier Open-Source Models for RAG)
    "qwen-2.5-32b": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 10_000,
        "tokens_per_day": 500_000,
        "advantages": "Powerful 32B model optimized for long-context comprehension and reasoning.",
        "disadvantages": "Requires more computational resources.",
    },

    # ✅ Google’s Gemma Model (High Throughput & Fast Inference)
    "gemma2-9b-it": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 15_000,
        "tokens_per_day": 500_000,
        "advantages": "Higher token throughput, suitable for large-scale, fast inference.",
        "disadvantages": "Limited versatility compared to larger LLaMA3 models.",
    },

    # ✅ Meta’s LLaMA 3 Models (Best for Long-Context RAG Applications)
    "llama-3.1-8b-instant": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 20_000,
        "tokens_per_day": 500_000,
        "advantages": "High-speed processing with large token capacity, great for real-time applications.",
        "disadvantages": "Less accurate for complex reasoning tasks compared to larger models.",
    },
    "llama-3.3-70b-versatile": {
        "requests_per_minute": 30,
        "requests_per_day": 1_000,
        "tokens_per_minute": 6_000,
        "tokens_per_day": 100_000,
        "advantages": "Versatile model optimized for high accuracy in diverse scenarios.",
        "disadvantages": "Lower throughput compared to some smaller models.",
    },
    "llama3-70b-8192": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 6_000,
        "tokens_per_day": 500_000,
        "advantages": "Long-context capabilities, ideal for handling detailed research papers and articles.",
        "disadvantages": "Moderate speed and accuracy for shorter tasks.",
    },
    "llama3-8b-8192": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 20_000,
        "tokens_per_day": 500_000,
        "advantages": "Supports high-speed inference with long-context support.",
        "disadvantages": "Slightly less accurate for complex reasoning compared to larger models.",
    },

    # ✅ Mistral AI (Best for Multi-Turn Chat & Retrieval-Based Augmentation)
    "mistral-saba-24b": {
        "requests_per_minute": 30,
        "requests_per_day": 7_000,
        "tokens_per_minute": 7_000,
        "tokens_per_day": 250_000,
        "advantages": "Strong multi-turn conversation capabilities and effective retrieval augmentation.",
        "disadvantages": "Limited token capacity compared to LLaMA-70B.",
    },
    "mixtral-8x7b-32768": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 5_000,
        "tokens_per_day": 500_000,
        "advantages": "Supports long document processing for better contextual understanding.",
        "disadvantages": "Lower token throughput compared to some other models.",
    },
}

# Allow users to select multiple models
selected_models = st.sidebar.multiselect(
    "Choose Models",
    options=list(models.keys()),
    default=["llama3-70b-8192"],  # Default model selection
)

# If no model is selected, set a fallback default model
if not selected_models:
    selected_models = ["llama3-70b-8192"]  # Default model to prevent errors

# Display details for all selected models
st.sidebar.write("## Selected Model Details")
for model_name in selected_models:
    selected_model = models[model_name]
    st.sidebar.write(f"### {model_name}")
    
    for key, value in selected_model.items():
        if key not in ["advantages", "disadvantages"]:
            st.sidebar.write(f"- **{key.replace('_', ' ').title()}**: {value}")
    
    st.sidebar.write(f"- **Advantages**: {selected_model['advantages']}")
    st.sidebar.write(f"- **Disadvantages**: {selected_model['disadvantages']}")
    st.sidebar.write("---")  # Add a separator for clarity

# Temperature slider with explanation
st.sidebar.markdown(
    """
    ### Temperature:
    Controls the randomness of the model's output. 
    - **Low Values (e.g., 0.1–0.3):** Makes the responses more deterministic and focused.  
    - **Medium Values (e.g., 0.7–1.0):** Balanced creativity and focus.  
    - **High Values (e.g., 1.5–2.0):** Increases creativity and variability, but may lead to less accurate or unpredictable responses.
    """
)
temperature = st.sidebar.slider(
    label="Temperature",
    min_value=0.0,
    max_value=2.0,  # Updated to extend the range to 2.0
    value=0.7,
    step=0.1,
    help="Adjust the randomness of the model's responses. Lower values = more focused; higher values = more creative."
)

# Chunk size slider with explanation
st.sidebar.markdown(
    """
    ### Chunk Size:
    Defines the number of words in each content chunk. 
    - **Small Chunks (e.g., 100–300):** Ideal for highly specific queries or short articles.  
    - **Large Chunks (e.g., 1000–3000):** Better for summarization or broader context but may lose finer details.
    """
)
chunk_size = st.sidebar.slider(
    label="Chunk Size",
    min_value=100,
    max_value=3000,
    value=300,
    step=50,
    help="Set the number of words in each content chunk. Choose smaller values for better specificity or larger for broader context."
)

# Chunk overlap slider with explanation
st.sidebar.markdown(
    """
    ### Chunk Overlap:
    Specifies the overlap between consecutive content chunks.  
    - **Smaller Overlap (e.g., 10–50):** Reduces redundancy but may miss context in some queries.  
    - **Larger Overlap (e.g., 200–300):** Ensures more context but may increase processing time.
    """
)
chunk_overlap = st.sidebar.slider(
    label="Chunk Overlap",
    min_value=10,
    max_value=300,
    value=50,
    step=10,
    help="Control the overlap of consecutive content chunks. Larger values improve context but may slow down processing."
)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How may I assist you today?"}]

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = ""

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "chain" not in st.session_state or st.session_state.get("selected_model") != model_name:
    st.session_state["chain"] = get_chain(model_name, temperature)
    st.session_state["selected_model"] = model_name

# Reset App Function
def reset_app():
    """Set a reset flag to True and trigger app re-run."""
    st.session_state["reset"] = True
    st.rerun()

# Add Reset App button in the sidebar
st.sidebar.button("Reset App", on_click=reset_app)

# Check if reset is flagged
if st.session_state.get("reset", False):
    # Clear all session state variables and cache
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["reset"] = False
    st.rerun()

# Content Upload Section
st.sidebar.title("Upload Content")
input_method = st.sidebar.radio("Input Method", ["PDF File", "URL"])
content = None

# Cache extracted text from PDF or URL
@cache_data
def cached_extract_text(input_method, uploaded_file=None, url=None):
    if input_method == "PDF File" and uploaded_file:
        return extract_text_from_pdf(uploaded_file)
    elif input_method == "URL" and url:
        return extract_text_from_url(url)
    return None

# Cache processed content chunks
@cache_data
def cached_process_content(content, chunk_size, chunk_overlap):
    return process_content(content, chunk_size, chunk_overlap)

# Cache vector store creation
@cache_resource
def cached_create_vector_store(_chunks):
    return create_vector_store(_chunks)

# Cache chain setup
@cache_resource
def cached_get_chain(model_name, temperature):
    return get_chain(model_name, temperature)

# --- Logic with Cached Functions ---
if input_method == "PDF File":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF File")
    if uploaded_file:
        content = cached_extract_text(input_method, uploaded_file=uploaded_file)
elif input_method == "URL":
    url = st.sidebar.text_input("Enter a News URL")
    if url:
        content = cached_extract_text(input_method, url=url)

if content and "Error" not in content:
    st.sidebar.markdown(
        f"<div style='color:green; font-weight:bold;'>Content Extracted:</div>"
        f"<div style='color:green;'>{content[:500]}...</div>",
        unsafe_allow_html=True,
    )

    total_word_count = len(content.split())
    st.sidebar.markdown(
        f"<div style='color:green; font-weight:bold;'>Total Word Count:</div>"
        f"<div style='color:green;'>{total_word_count}</div>",
        unsafe_allow_html=True,
    )

    chunks = cached_process_content(content, chunk_size, chunk_overlap)
    if chunks:
        st.sidebar.markdown(
            f"<div style='color:green; font-weight:bold;'>Total Chunks Created:</div>"
            f"<div style='color:green;'>{len(chunks)}</div>",
            unsafe_allow_html=True,
        )

        if st.session_state["vector_store"] is None:
            st.session_state["vector_store"] = cached_create_vector_store(chunks)

        retriever = st.session_state["vector_store"].as_retriever()

        st.write("---")
        st.header("Chat with the Content 🤖")

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask a question about the content:"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            context_docs = retriever.get_relevant_documents(prompt) if retriever else []
            context = " ".join([doc.page_content for doc in context_docs]) if context_docs else ""

            # Store responses from different models
            responses = {}

            for model in selected_models:
                chain = get_chain(model, temperature)  # Get the chain for each model
                response, _ = ask_question(chain, prompt, context, st.session_state["conversation_history"])
                responses[model] = response  # Store responses

            # Display responses from all selected models
            for model, response in responses.items():
                with st.chat_message("assistant"):
                    st.markdown(f"### Response from {model}:")
                    with st.spinner("Thinking..."):
                        st.write(response)
                st.session_state["messages"].append({"role": "assistant", "content": f"**{model}**: {response}"})