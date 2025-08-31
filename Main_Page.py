import streamlit as st

st.set_page_config(
    page_title="Cortex AI: Multi-Model Insights Hub",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for the dark theme
st.markdown('''
<style>
    /* Using the theme from config.toml */
    body {
        font-family: 'monospace';
        color: #40e723; /* textColor */
        background-color: #000000; /* backgroundColor */
    }
    .stApp {
        background-color: #000000;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #40e723; /* primaryColor for headers */
    }
    .main-container {
        border: 2px solid #40e723;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem;
        background-color: #1a1a1a; /* A slightly lighter black for contrast */
    }
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        text-shadow: 0 0 10px #40e723;
    }
    .subtitle {
        font-size: 1.25rem;
        text-align: center;
        color: #FFFFFF; /* secondaryBackgroundColor for subtitle text */
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #0d0d0d;
        border: 1px solid #40e723;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        height: 100%;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 0 15px #40e723;
    }
    .feature-title {
        font-size: 2rem;
        font-weight: bold;
    }
    .feature-description {
        font-size: 1rem;
        color: #FFFFFF;
    }
    .get-started {
        text-align: center;
        margin-top: 2rem;
        font-size: 1.2rem;
    }
</style>
''', unsafe_allow_html=True)

# Main Content
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">Cortex AI Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-Model Document Analysis & Intelligent Search</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="feature-card">
            <h2 class="feature-title">📄 RAG Chatbot</h2>
            <p class="feature-description">
                Engage in a dialogue with your documents. Upload PDFs or provide URLs to get summaries, 
                ask questions, and extract key information with our Retrieval-Augmented Generation chatbot.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.markdown('''
        <div class="feature-card">
            <h2 class "feature-title">🔍 Search Agent</h2>
            <p class="feature-description">
                Your intelligent agent for exploring the web. Ask about any topic, 
                and our agent will use multiple tools to find the most relevant and up-to-date information for you.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown('<p class="get-started">Select a tool from the sidebar to begin your journey into AI-powered insights.</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
