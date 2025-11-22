import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from styles import apply_custom_css
from config import MODELS
import re
import os
from dotenv import load_dotenv
from helpers import process_content, create_vector_store, hybrid_search
from chain_setup import get_chain, ask_question
import hashlib

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="YouTube Analyst",
    page_icon="📺",
    layout="wide"
)

# Apply styles
apply_custom_css()

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    # Regex for extracting video ID
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def get_transcript(video_id):
    """
    Fetches the transcript of a YouTube video.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_obj = api.fetch(video_id)
        transcript_text = " ".join([item.text for item in transcript_obj])
        return transcript_text
    except Exception as e:
        return f"Error: {str(e)}"

def get_summary_chain(model_name):
    """
    Creates a chain for summarizing text.
    """
    llm = ChatGroq(
        model_name=model_name,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    template = """
    You are an expert video analyst. Summarize the following YouTube video transcript.
    Provide a concise summary followed by key takeaways in bullet points.
    
    Transcript:
    {transcript}
    
    Summary:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

# --- UI ---
st.title("📺 YouTube Video Analyst")

# Initialize Session State
if "yt_transcript" not in st.session_state:
    st.session_state.yt_transcript = None
if "yt_video_id" not in st.session_state:
    st.session_state.yt_video_id = None
if "yt_vector_store" not in st.session_state:
    st.session_state.yt_vector_store = None
if "yt_chunks" not in st.session_state:
    st.session_state.yt_chunks = []
if "yt_messages" not in st.session_state:
    st.session_state.yt_messages = [{"role": "assistant", "content": "Analyze a video to start chatting!"}]
if "yt_conversation_history" not in st.session_state:
    st.session_state.yt_conversation_history = ""
if "yt_summary" not in st.session_state:
    st.session_state.yt_summary = None

# Sidebar Configuration & Video
with st.sidebar:
    st.header("Configuration")
    selected_model = st.selectbox("Select Model", options=list(MODELS.keys()), index=0)
    
    st.markdown("---")
    st.subheader("📹 Video Source")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if url:
        video_id = extract_video_id(url)
        
        if video_id:
            # Reset if new video
            if video_id != st.session_state.yt_video_id:
                st.session_state.yt_video_id = video_id
                st.session_state.yt_transcript = None
                st.session_state.yt_vector_store = None
                st.session_state.yt_chunks = []
                st.session_state.yt_messages = [{"role": "assistant", "content": "Analyze a video to start chatting!"}]
                st.session_state.yt_conversation_history = ""
                st.session_state.yt_summary = None
                
            st.video(url)
            
            if st.button("Analyze Video", type="primary"):
                with st.spinner("Fetching transcript and processing..."):
                    transcript = get_transcript(video_id)
                    
                    if "Error:" in transcript:
                        st.error(transcript)
                    else:
                        st.session_state.yt_transcript = transcript
                        
                        # Process for RAG
                        content_list = [(transcript, f"YouTube Video: {url}")]
                        chunks = process_content(content_list)
                        st.session_state.yt_chunks = chunks
                        
                        # Create unique collection name
                        collection_name = f"yt_{video_id}_{hashlib.md5(transcript.encode()).hexdigest()[:8]}"
                        st.session_state.yt_vector_store = create_vector_store(chunks, collection_name)
                        
                        st.success("Ready to chat!")
                        st.session_state.yt_messages.append({"role": "assistant", "content": "I've analyzed the video. You can now ask me anything about it!"})
                        
                        # Auto-generate summary
                        try:
                            chain = get_summary_chain(selected_model)
                            summary = chain.invoke({"transcript": st.session_state.yt_transcript[:30000]})
                            st.session_state.yt_summary = summary
                        except Exception as e:
                            st.error(f"Could not auto-generate summary: {e}")

        else:
            st.error("Invalid YouTube URL")

    # Display Summary in Sidebar
    if st.session_state.yt_summary:
        st.markdown("---")
        with st.expander("📝 Video Summary", expanded=True):
            st.markdown(st.session_state.yt_summary)

# Main Chat Interface
if st.session_state.yt_transcript:
    # Chat Container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.yt_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about the video..."):
        st.session_state.yt_messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                
                try:
                    # Retrieve Context
                    context_text = ""
                    if st.session_state.yt_vector_store:
                        final_docs = hybrid_search(
                            prompt, 
                            st.session_state.yt_vector_store, 
                            st.session_state.yt_chunks, 
                            top_n=5
                        )
                        context_text = "\n\n".join([d.page_content for d in final_docs])
                    else:
                        context_text = st.session_state.yt_transcript[:30000]

                    # Generate Response
                    chain = get_chain(selected_model)
                    response, _ = ask_question(
                        chain, 
                        prompt, 
                        context_text, 
                        chat_history=st.session_state.yt_conversation_history
                    )
                    
                    placeholder.markdown(response)
                    
                    # Update History
                    st.session_state.yt_messages.append({"role": "assistant", "content": response})
                    st.session_state.yt_conversation_history += f"\nUser: {prompt}\nAssistant: {response}"
                    
                except Exception as e:
                    placeholder.error(f"Error generating response: {e}")
else:
    st.info("👈 Paste a YouTube URL in the sidebar and click 'Analyze Video' to get started.")
