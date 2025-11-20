import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Search Agent",
    page_icon="🔍",
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

# --- Environment & API Setup ---
load_dotenv()
if not os.getenv("TAVILY_API_KEY") or not os.getenv("GROQ_API_KEY"):
    st.error("API keys for Tavily and Groq are not set. Please add them to your .env file.")
    st.stop()

ChatGroq.model_rebuild()

# --- Tool Initialization ---
tools = [
    ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)),
    TavilySearchResults(max_results=5, search_depth="advanced")
]

# --- Model & Settings Data ---
models = {
    "llama-3.3-70b-versatile": {"advantages": "High accuracy in diverse scenarios.", "disadvantages": "Lower throughput.", "provider": "Meta"},
    "llama-3.1-8b-instant": {"advantages": "High-speed for real-time apps.", "disadvantages": "Less accurate for complex tasks.", "provider": "Meta"},
    "meta-llama/llama-guard-4-12b": {"advantages": "Optimized for safety and guardrailing.", "disadvantages": "Specialized for safety, not general chat.", "provider": "Meta"},
    "openai/gpt-oss-120b": {"advantages": "120B params, browser search, code execution.", "disadvantages": "Slower speed.", "provider": "OpenAI"},
    "openai/gpt-oss-20b": {"advantages": "20B params, browser search, code execution.", "disadvantages": "Smaller model.", "provider": "OpenAI"},
}

# --- Sidebar UI ---
with st.sidebar:
    st.title("🔍 Search Configuration")
    
    st.subheader("AI Model")
    selected_model = st.selectbox("Select a model", options=list(models.keys()), index=0)
    
    with st.expander("Model Details"):
        st.markdown(f"**{selected_model}** (*{models[selected_model]['provider']}*)")
        st.markdown(f"- **Pros**: {models[selected_model]['advantages']}")
        st.markdown(f"- **Cons**: {models[selected_model]['disadvantages']}")

    st.subheader("Parameters")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.05)

    if st.button("🗑️ Clear Chat"):
        st.session_state.search_state = {"messages": [AIMessage(content="Hi! I'm your search agent. How can I help?")]}
        st.rerun()
    
    # --- Information Section ---
    st.markdown("---")
    st.markdown("### 🔍 About Search Agent")
    st.markdown("""
    This intelligent agent can search across multiple sources:

    **📚 Available Tools:**
    - 🌐 **Web Search** (Tavily): Real-time web search
    - 📖 **Wikipedia**: Encyclopedia knowledge
    - 📄 **ArXiv**: Academic papers and research

    **💡 How it works:**
    1. Analyzes your question
    2. Selects appropriate tools
    3. Searches multiple sources
    4. Synthesizes comprehensive answers
    """)

# --- LangGraph State & Graph Definition ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

@st.cache_resource
def create_graph(selected_model, temperature):
    llm = ChatGroq(model=selected_model, temperature=temperature)
    llm_with_tools = llm.bind_tools(tools=tools)
    
    def tool_calling_llm(state):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")
    return builder.compile()

search_graph = create_graph(selected_model, temperature)

# --- Session State Initialization ---
if "search_state" not in st.session_state:
    st.session_state.search_state = {"messages": [AIMessage(content="Hi! I'm your search agent. How can I help?")]}

# --- Main Page UI & Chat Logic ---
st.title("🔍 Intelligent Search Agent")
st.caption("Your gateway to real-time information and research")

for message in st.session_state.search_state["messages"]:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("Ask anything..."):
    st.session_state.search_state["messages"].append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.spinner("🔍 Searching the web..."):
        try:
            output = search_graph.invoke(st.session_state.search_state)
            st.session_state.search_state = output
            
            if output["messages"]:
                latest_message = output["messages"][-1]
                with st.chat_message(latest_message.type):
                    st.markdown(latest_message.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")