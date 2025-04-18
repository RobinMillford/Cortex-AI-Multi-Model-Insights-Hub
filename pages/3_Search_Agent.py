import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
import os
import re
import json
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Check for required API keys
if not os.getenv("TAVILY_API_KEY") or not os.getenv("GROQ_API_KEY"):
    st.error("Please set TAVILY_API_KEY and GROQ_API_KEY in your .env file. See https://console.groq.com/settings/keys for Groq API key setup.")
    st.stop()

# Rebuild the ChatGroq model
ChatGroq.model_rebuild()

# Initialize search tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
tavily = TavilySearchResults(max_results=5, search_depth="advanced")
tools = [arxiv, wiki, tavily]

# Streamlit App Configuration
st.set_page_config(page_title="Multi-Model Search Agent Chatbot 🔍🤖", page_icon="🔍", layout="wide")

st.title("Multi-Model Search Agent Chatbot 🔍🤖")
st.markdown("Ask about recent news, research papers, or anything else! Try: 'Recent AI news', 'Arsenal vs Real Madrid UCL 2025', or 'What is MCP?'.")

# Model list
models = {
    "deepseek-r1-distill-llama-70b": {
        "advantages": "Highly optimized for low latency with no token limits, ideal for large-scale deployments.",
        "disadvantages": "Limited daily requests compared to other models.",
    },
    "qwen-2.5-32b": {
        "advantages": "Powerful 32B model optimized for long-context comprehension and reasoning.",
        "disadvantages": "Requires more computational resources.",
    },
    "gemma2-9b-it": {
        "advantages": "Higher token throughput, suitable for large-scale, fast inference.",
        "disadvantages": "Limited versatility compared to larger LLaMA3 models.",
    },
    "llama-3.1-8b-instant": {
        "advantages": "High-speed processing with large token capacity, great for real-time applications.",
        "disadvantages": "Less accurate for complex reasoning tasks compared to larger models.",
    },
    "llama-3.3-70b-versatile": {
        "advantages": "Versatile model optimized for high accuracy in diverse scenarios.",
        "disadvantages": "Lower throughput compared to some smaller models.",
    },
    "llama3-70b-8192": {
        "advantages": "Long-context capabilities, ideal for handling detailed research papers and articles.",
        "disadvantages": "Moderate speed and accuracy for shorter tasks.",
    },
    "llama3-8b-8192": {
        "advantages": "Supports high-speed inference with long-context support.",
        "disadvantages": "Slightly less accurate for complex reasoning compared to larger models.",
    },
    "mistral-saba-24b": {
        "advantages": "Strong multi-turn conversation capabilities and effective retrieval augmentation.",
        "disadvantages": "Limited token capacity compared to LLaMA-70B.",
    },
    "mixtral-8x7b-32768": {
        "advantages": "Supports long document processing for better contextual understanding.",
        "disadvantages": "Lower token throughput compared to some other models.",
    },
}

# Sidebar Settings
st.sidebar.title("Settings")
selected_model = st.sidebar.selectbox("Choose Model", options=list(models.keys()), index=5)  # Default to llama3-70b-8192
temperature = st.sidebar.slider(
    label="Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.05,  # Stricter for less hallucination
    step=0.01,
    help="Adjust randomness: lower = more focused, higher = more creative."
)

# Display selected model details
st.sidebar.subheader("Selected Model Details")
st.sidebar.markdown(f"### {selected_model}")
st.sidebar.markdown(f"- **Advantages**: {models[selected_model]['advantages']}")
st.sidebar.markdown(f"- **Disadvantages**: {models[selected_model]['disadvantages']}")

# Clear chat button in sidebar
if st.sidebar.button("Clear Chat"):
    st.session_state["search_state"] = {
        "messages": [AIMessage(content="Hello! I'm your search agent. Ask me about recent news, research papers, or definitions like 'What is MCP?'.")]
    }
    st.rerun()

# Initialize graph if model or temperature changes
if "search_graph" not in st.session_state or st.session_state.get("search_selected_model") != selected_model or st.session_state.get("search_temperature") != temperature:
    llm = ChatGroq(model=selected_model, temperature=temperature)
    llm_with_tools = llm.bind_tools(tools=tools)

    # Define state schema
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # Build graph
    builder = StateGraph(State)
    def tool_calling_llm(state):
        context = context_summary(state["messages"][-3:])  # Limit to last 3 messages
        prompt = f"Context: {context}\nCurrent query: {state['messages'][-1].content}\nProvide a concise, relevant response using the provided tools. Focus on recent information (past 30 days for news) and avoid speculation."
        messages = state["messages"][-3:] + [HumanMessage(content=prompt)]  # Truncate history
        return {"messages": [llm_with_tools.invoke(messages)]}
    
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")
    st.session_state["search_graph"] = builder.compile()
    st.session_state["search_selected_model"] = selected_model
    st.session_state["search_temperature"] = temperature

# Initialize conversation state
if "search_state" not in st.session_state:
    st.session_state["search_state"] = {
        "messages": [AIMessage(content="Hello! I'm your search agent. Ask me about recent news, research papers, or definitions like 'What is MCP?'.")]
    }

# Function to summarize conversation context
def context_summary(messages):
    summary = ""
    for msg in messages:
        if msg.type == "human":
            summary += f"User asked: {msg.content[:50]}...\n"
        elif msg.type == "ai":
            summary += f"Bot responded: {msg.content[:50]}...\n"
    return summary.strip() or "No prior context."

# Function to filter and validate tool results
def filter_tool_results(results, query):
    filtered = []
    cutoff_date = datetime.now() - timedelta(days=30)
    for result in results:
        content = result.get("content", "")
        url = result.get("url", "")
        pub_date = result.get("published_date", None)
        if pub_date:
            try:
                pub_date = datetime.strptime(pub_date, "%Y-%m-%d")
                if pub_date < cutoff_date:
                    continue
            except ValueError:
                pass
        if query.lower() in content.lower() or any(keyword in content.lower() for keyword in query.split()):
            filtered.append({"content": content[:150], "url": url})  # Reduced length
    return filtered[:2]  # Limit to 2 results

# Function to render tool results with clickable links
def render_tool_result(content, query):
    try:
        results = json.loads(content)
        if isinstance(results, list):
            filtered_results = filter_tool_results(results, query)
            if not filtered_results:
                return "No recent, relevant results found."
            content = "Tool results:\n"
            for result in filtered_results:
                content += f"- [{result['content']}...]({result['url']})\n"
            return content
    except json.JSONDecodeError:
        pattern = r'\[(.*?)\]\((https?://\S+)\)'
        matches = re.findall(pattern, content, re.DOTALL)
        if not matches:
            return content.strip()[:150] + "..." if content.strip() else "No relevant results."
        filtered_content = ""
        for text, url in matches:
            if query.lower() in text.lower() or any(keyword in text.lower() for keyword in query.split()):
                filtered_content += f"- [{url}]({url})\n"
                lines = text.strip().split('\n')[:2]  # Limit to 2 lines
                for line in lines:
                    filtered_content += f"  - {line.strip()}\n"
        return filtered_content or "No recent, relevant results found."

# Chat Interface
st.markdown("---")
st.header("Search Agent Chat 🤖")

# Display conversation
for message in st.session_state["search_state"]["messages"]:
    if message.type == "human":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(message.content)
    elif message.type == "ai":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif message.type == "tool":
        with st.chat_message("assistant", avatar="🤖"):
            content = render_tool_result(message.content, message.content)
            st.markdown(content)

# User input
if prompt := st.chat_input("Ask a question (e.g., 'Recent AI news' or 'Arsenal vs Real Madrid UCL 2025'):"):
    st.session_state["search_state"]["messages"].append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    with st.spinner("Searching..."):
        try:
            output = st.session_state["search_graph"].invoke(st.session_state["search_state"])
            st.session_state["search_state"] = output
            if output["messages"]:
                latest_message = output["messages"][-1]
                with st.chat_message("assistant", avatar="🤖"):
                    if latest_message.type == "ai":
                        content = latest_message.content
                    elif latest_message.type == "tool":
                        content = render_tool_result(latest_message.content, prompt)
                    st.markdown(content)
                    # Add retry button if no results
                    if latest_message.type == "tool" and "No recent, relevant results found" in content:
                        if st.button("Retry Search"):
                            st.session_state["search_state"]["messages"].append(HumanMessage(content=prompt))
                            st.rerun()
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                st.error("Request too large. Truncating history and retrying...")
                st.session_state["search_state"]["messages"] = st.session_state["search_state"]["messages"][-2:]  # Keep last 2 messages
                st.rerun()
            else:
                st.error(f"Error during search: {str(e)}")