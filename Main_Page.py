import streamlit as st

st.set_page_config(page_title="Cortex AI: Multi-Model Insights Hub", page_icon="🤖", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #4B0082;
        margin-top: 50px;
    }
    .subtitle {
        font-size: 24px;
        text-align: center;
        color: #2F4F4F;
        margin-bottom: 30px;
    }
    .description {
        font-size: 18px;
        text-align: center;
        margin: 20px 0;
    }
    .page-info {
        font-size: 16px;
        text-align: center;
        margin-top: 40px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Title and Subtitle
st.markdown('<div class="main-title">Cortex AI: Multi-Model Insights Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Empowering your research, analysis, and data exploration with multiple AI models</div>', unsafe_allow_html=True)


# Description Section
st.markdown(
    """
    <div class="description">
        Welcome to Cortex AI Hub, your one-stop platform for leveraging advanced AI models for multi-model retrieval, 
        interactive research, and dynamic data analysis. Our platform offers:
        <ul style="list-style-type:square; display: inline-block; text-align: left;">
            <li><strong>Article Chatbot</strong> – Engage with RAG-powered insights to explore research articles.</li>
            <li><strong>Data Analysis Chatbot</strong> – Visualize and analyze your data with AI-driven assistance.</li>
        </ul>
        <br>
        Select a page from the sidebar to get started!
    </div>
    """,
    unsafe_allow_html=True,
)

# Additional Footer or Info Section
st.markdown(
    """
    <div class="page-info">
        © 2025 Cortex AI Hub • Built with Streamlit &amp; advanced AI models
    </div>
    """,
    unsafe_allow_html=True,
)