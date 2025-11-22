import streamlit as st

def apply_custom_css():
    """
    Applies the premium dark theme CSS to the Streamlit app.
    Enhanced with glassmorphism, smooth animations, and modern design elements.
    """
    st.markdown('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* General App Styling */
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #050505 50%, #0d0d0d 100%);
            color: #e8e8e8;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        
        /* Typography */
        h1 {
            background: linear-gradient(135deg, #00ff9d 0%, #00d4aa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.5px;
            text-shadow: 0 0 30px rgba(0, 255, 157, 0.3);
        }
        
        h2, h3 {
            color: #00ff9d !important;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            letter-spacing: -0.3px;
        }
        
        p, div, label {
            font-family: 'Inter', sans-serif;
            color: #e8e8e8;
            font-weight: 400;
        }

        /* Sidebar Styling with Glassmorphism */
        [data-testid="stSidebar"] {
            background: rgba(10, 10, 10, 0.95);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(0, 255, 157, 0.1);
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
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
            background: linear-gradient(135deg, rgba(10, 10, 10, 0.8) 0%, rgba(26, 26, 26, 0.8) 100%);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(0, 255, 157, 0.2);
            margin-bottom: 3rem;
            box-shadow: 0 8px 32px rgba(0, 255, 157, 0.15);
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
            background: linear-gradient(135deg, rgba(17, 17, 17, 0.8) 0%, rgba(26, 26, 26, 0.8) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            height: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
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
            border-color: rgba(0, 255, 157, 0.5);
            box-shadow: 0 12px 40px rgba(0, 255, 157, 0.2);
            background: linear-gradient(135deg, rgba(17, 17, 17, 0.9) 0%, rgba(26, 26, 26, 0.9) 100%);
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
            background: rgba(10, 10, 10, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(0, 255, 157, 0.1);
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
            background: linear-gradient(135deg, rgba(10, 10, 10, 0.8) 0%, rgba(26, 26, 26, 0.8) 100%);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(0, 255, 157, 0.2);
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

        /* Input Fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: rgba(26, 26, 26, 0.8);
            color: #ffffff;
            border: 1.5px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 12px 16px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-family: 'Inter', sans-serif;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #00ff9d;
            box-shadow: 0 0 0 3px rgba(0, 255, 157, 0.15),
                        0 0 20px rgba(0, 255, 157, 0.2);
            background: rgba(26, 26, 26, 1);
            transform: translateY(-1px);
        }

        /* Buttons with Enhanced Gradient */
        .stButton > button {
            background: linear-gradient(135deg, #00ff9d 0%, #00cc7a 100%);
            color: #000000;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(0, 255, 157, 0.2);
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.3px;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 255, 157, 0.4);
            background: linear-gradient(135deg, #00ffaa 0%, #00d47a 100%);
        }
        
        .stButton > button:active {
            transform: translateY(-1px);
        }

        /* Chat Messages with Glassmorphism */
        .stChatMessage {
            background: rgba(17, 17, 17, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .stChatMessage:hover {
            background: rgba(17, 17, 17, 0.8);
            border-color: rgba(0, 255, 157, 0.2);
            transform: translateX(4px);
        }
        
        [data-testid="stChatMessageContent"] {
            color: #e8e8e8;
            line-height: 1.6;
        }
        
        /* User Message with Neon Accent */
        .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border-left: 3px solid #00ff9d;
            background: linear-gradient(90deg, rgba(0, 255, 157, 0.05) 0%, rgba(17, 17, 17, 0.6) 100%);
        }
        
        /* Assistant Message with Blue Accent */
        .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-left: 3px solid #00b8ff;
            background: linear-gradient(90deg, rgba(0, 184, 255, 0.05) 0%, rgba(17, 17, 17, 0.6) 100%);
        }

        /* Chat Input */
        .stChatInputContainer {
            border-top: 1px solid rgba(0, 255, 157, 0.1);
            background: rgba(10, 10, 10, 0.8);
            backdrop-filter: blur(10px);
        }

        /* Expanders with Enhanced Style */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, rgba(26, 26, 26, 0.8) 0%, rgba(20, 20, 20, 0.8) 100%);
            border: 1px solid rgba(0, 255, 157, 0.2);
            border-radius: 12px;
            color: #00ff9d;
            font-weight: 500;
            padding: 12px 16px;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, rgba(0, 255, 157, 0.1) 0%, rgba(20, 20, 20, 0.9) 100%);
            border-color: #00ff9d;
            box-shadow: 0 4px 12px rgba(0, 255, 157, 0.15);
        }
        
        .streamlit-expanderContent {
            background: rgba(15, 15, 15, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-top: none;
            border-radius: 0 0 12px 12px;
        }
        
        /* Success/Info/Warning Messages */
        .stSuccess {
            background: linear-gradient(135deg, rgba(0, 255, 157, 0.1) 0%, rgba(0, 204, 122, 0.05) 100%);
            color: #e8e8e8;
            border-left: 4px solid #00ff9d;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 2px 8px rgba(0, 255, 157, 0.1);
        }
        
        .stInfo {
            background: linear-gradient(135deg, rgba(0, 184, 255, 0.1) 0%, rgba(0, 150, 255, 0.05) 100%);
            color: #e8e8e8;
            border-left: 4px solid #00b8ff;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 2px 8px rgba(0, 184, 255, 0.1);
        }
        
        .stWarning {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.05) 100%);
            color: #e8e8e8;
            border-left: 4px solid #ffc107;
            border-radius: 8px;
            padding: 12px 16px;
        }
        
        .stError {
            background: linear-gradient(135deg, rgba(255, 82, 82, 0.1) 0%, rgba(255, 23, 68, 0.05) 100%);
            color: #e8e8e8;
            border-left: 4px solid #ff5252;
            border-radius: 8px;
            padding: 12px 16px;
        }
        
        /* Image Container with Glow Effect */
        .image-container {
            border: 2px solid rgba(0, 255, 157, 0.3);
            border-radius: 16px;
            padding: 16px;
            margin: 16px 0;
            background: linear-gradient(135deg, rgba(17, 17, 17, 0.8) 0%, rgba(10, 10, 10, 0.8) 100%);
            box-shadow: 0 4px 20px rgba(0, 255, 157, 0.1),
                        inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }
        
        .image-container:hover {
            border-color: #00ff9d;
            box-shadow: 0 8px 30px rgba(0, 255, 157, 0.2),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
        }
        
        /* Select Boxes and Dropdowns */
        .stSelectbox > div > div {
            background: rgba(26, 26, 26, 0.8);
            border: 1.5px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div:hover {
            border-color: rgba(0, 255, 157, 0.5);
        }
        
        /* Sliders */
        .stSlider > div > div > div {
            background: rgba(0, 255, 157, 0.2);
        }
        
        .stSlider > div > div > div > div {
            background: linear-gradient(135deg, #00ff9d 0%, #00cc7a 100%);
            box-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        }
        
        /* Checkboxes */
        .stCheckbox > label {
            font-weight: 500;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: rgba(26, 26, 26, 0.6);
            border: 2px dashed rgba(0, 255, 157, 0.3);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #00ff9d;
            background: rgba(26, 26, 26, 0.8);
            box-shadow: 0 4px 20px rgba(0, 255, 157, 0.1);
        }
        
        /* Progress Bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #00ff9d 0%, #00cc7a 100%);
            box-shadow: 0 0 15px rgba(0, 255, 157, 0.5);
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #00ff9d !important;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(10, 10, 10, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #00ff9d 0%, #00cc7a 100%);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #00ffaa 0%, #00d47a 100%);
        }
        
        /* Markdown Content */
        .stMarkdown {
            color: #e8e8e8;
        }
        
        .stMarkdown a {
            color: #00ff9d;
            text-decoration: none;
            transition: all 0.2s ease;
        }
        
        .stMarkdown a:hover {
            color: #00ffaa;
            text-shadow: 0 0 8px rgba(0, 255, 157, 0.5);
        }
        
        /* Code Blocks */
        code {
            background: rgba(0, 255, 157, 0.1);
            color: #00ff9d;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        
        pre {
            background: rgba(15, 15, 15, 0.8);
            border: 1px solid rgba(0, 255, 157, 0.2);
            border-radius: 12px;
            padding: 16px;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(10, 10, 10, 0.5);
            padding: 8px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(26, 26, 26, 0.6);
            border-radius: 8px;
            color: #808080;
            padding: 8px 16px;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(0, 255, 157, 0.1);
            color: #00ff9d;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00ff9d 0%, #00cc7a 100%);
            color: #000000;
            font-weight: 600;
        }
        
        /* Dataframes */
        .stDataFrame {
            border: 1px solid rgba(0, 255, 157, 0.2);
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #00ff9d;
            font-weight: 700;
        }
        
        /* Animations */
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px rgba(0, 255, 157, 0.2); }
            50% { box-shadow: 0 0 20px rgba(0, 255, 157, 0.4); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .stChatMessage {
            animation: fadeIn 0.3s ease-out;
        }
        
        /* Radio Buttons */
        .stRadio > div {
            background: rgba(26, 26, 26, 0.4);
            padding: 8px;
            border-radius: 12px;
        }
        
        /* Number Input */
        .stNumberInput > div > div > input {
            background: rgba(26, 26, 26, 0.8);
            color: #ffffff;
            border: 1.5px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
        }
        
        .stNumberInput > div > div > input:focus {
            border-color: #00ff9d;
            box-shadow: 0 0 0 3px rgba(0, 255, 157, 0.15);
        }
    </style>
    ''', unsafe_allow_html=True)
