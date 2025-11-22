import streamlit as st

st.set_page_config(
    page_title="Cortex AI: Multi-Model Insights Hub",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for Premium Dark Theme ---
from styles import apply_custom_css
apply_custom_css()

# Main Content
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown('''
    <div class="hero-section">
        <h1 class="main-title">🤖 Cortex AI Hub</h1>
        <p class="subtitle">Multi-Model Document Analysis & Intelligent Search</p>
        <p class="tagline">Powered by Groq, LangChain & Advanced AI Models</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Stats Section
    st.markdown('''
    <div class="stats-container">
        <div class="stat-item">
            <span class="stat-number">5+</span>
            <span class="stat-label">AI Models</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">4</span>
            <span class="stat-label">Tools</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">∞</span>
            <span class="stat-label">Possibilities</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Feature Cards - Row 1
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('''
        <div class="feature-card">
            <span class="feature-icon">📄</span>
            <h2 class="feature-title">RAG Chatbot</h2>
            <p class="feature-description">
                Engage in intelligent dialogue with your documents. Upload PDFs or provide URLs 
                to get instant summaries, ask questions, and extract key information using 
                Retrieval-Augmented Generation technology.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.markdown('''
        <div class="feature-card">
            <span class="feature-icon">🔍</span>
            <h2 class="feature-title">Search Agent</h2>
            <p class="feature-description">
                Your intelligent agent for exploring the web. Leverages multiple search tools 
                including Tavily, Wikipedia, and ArXiv to find the most relevant and 
                up-to-date information on any topic.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Feature Cards - Row 2
    col3, col4 = st.columns(2, gap="large")
    
    with col3:
        st.markdown('''
        <div class="feature-card">
            <span class="feature-icon">🖼️</span>
            <h2 class="feature-title">Multimodal RAG</h2>
            <p class="feature-description">
                Advanced document analysis that understands both text AND images. 
                Analyze charts, graphs, infographics, and visual content alongside text 
                for comprehensive, multimodal insights.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown('''
        <div class="feature-card">
            <span class="feature-icon">📺</span>
            <h2 class="feature-title">YouTube Analyst</h2>
            <p class="feature-description">
                Transform YouTube videos into interactive knowledge bases. Extract transcripts, 
                generate summaries, and chat with video content using AI-powered analysis 
                for deeper understanding and insights.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Get Started Section
    st.markdown('''
    <div class="get-started">
        <p class="get-started-text">
            <span class="arrow-icon">←</span>
            Select a tool from the sidebar to begin your journey into AI-powered insights
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
