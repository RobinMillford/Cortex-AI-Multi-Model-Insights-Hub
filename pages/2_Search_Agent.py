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
    "deepseek-r1-distill-llama-70b": {"advantages": "Low latency, no token limits.", "disadvantages": "Limited daily requests.", "provider": "DeepSeek"},
    "qwen/qwen3-32b": {"advantages": "Powerful 32B model for long-context.", "disadvantages": "Computationally intensive.", "provider": "Alibaba Cloud"},
    "openai/gpt-oss-120b": {"advantages": "120B params, browser search, code execution.", "disadvantages": "Slower speed.", "provider": "OpenAI"},
    "openai/gpt-oss-20b": {"advantages": "20B params, browser search, code execution.", "disadvantages": "Smaller model.", "provider": "OpenAI"},
}

# --- Sidebar UI ---
with st.sidebar:
    st.header("Search Configuration")
    
    st.subheader("AI Model")
    selected_model = st.selectbox("Select a model", options=list(models.keys()), index=0)
    
    with st.expander("Model Details"):
        st.markdown(f"**{selected_model}** (*{models[selected_model]['provider']}*)")
        st.markdown(f"- **Pros**: {models[selected_model]['advantages']}")
        st.markdown(f"- **Cons**: {models[selected_model]['disadvantages']}")

    st.subheader("Temperature")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.05)

    if st.button("Clear Chat"):
        st.session_state.search_state = {"messages": [AIMessage(content="Hi! I'm your search agent. How can I help?")]}
        st.rerun()

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
st.markdown("Your gateway to real-time information and research.")

for message in st.session_state.search_state["messages"]:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("Ask anything..."):
    st.session_state.search_state["messages"].append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.spinner("Searching the web..."):
        try:
            output = search_graph.invoke(st.session_state.search_state)
            st.session_state.search_state = output
            
            if output["messages"]:
                latest_message = output["messages"][-1]
                with st.chat_message(latest_message.type):
                    st.markdown(latest_message.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")