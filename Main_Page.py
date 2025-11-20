import streamlit as st

st.set_page_config(
    page_title="Cortex AI: Multi-Model Insights Hub",
    page_icon="🤖",
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

    /* Main Container */
    .main-container {
        padding: 2rem 1rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border-radius: 20px;
        border: 1px solid #333;
        margin-bottom: 3rem;
        box-shadow: 0 8px 32px rgba(0, 255, 157, 0.1);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff9d 0%, #00cc7a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(0, 255, 157, 0.3);
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #b0b0b0;
        margin-bottom: 0.5rem;
    }
    
    .tagline {
        font-size: 1rem;
        color: #808080;
        font-style: italic;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #111 0%, #1a1a1a 100%);
        border: 1px solid #333;
        border-radius: 15px;
        padding: 2rem;
        height: 100%;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00ff9d 0%, #00cc7a 100%);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        border-color: #00ff9d;
        box-shadow: 0 12px 40px rgba(0, 255, 157, 0.2);
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #00ff9d;
        margin-bottom: 1rem;
    }
    
    .feature-description {
        font-size: 1rem;
        color: #b0b0b0;
        line-height: 1.6;
    }
    
    /* Stats Section */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 3rem 0;
        padding: 2rem;
        background-color: #0a0a0a;
        border-radius: 15px;
        border: 1px solid #333;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00ff9d;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #808080;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Get Started Section */
    .get-started {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border-radius: 15px;
        border: 1px solid #333;
    }
    
    .get-started-text {
        font-size: 1.2rem;
        color: #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .arrow-icon {
        font-size: 2rem;
        color: #00ff9d;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateX(0);
        }
        40% {
            transform: translateX(-10px);
        }
        60% {
            transform: translateX(-5px);
        }
    }
</style>
''', unsafe_allow_html=True)

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
            <span class="stat-number">5</span>
            <span class="stat-label">AI Models</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">3</span>
            <span class="stat-label">Tools</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">∞</span>
            <span class="stat-label">Possibilities</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3, gap="large")
    
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
