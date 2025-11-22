import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model Definitions
MODELS = {
    "llama-3.3-70b-versatile": {
        "advantages": "High accuracy in diverse scenarios.",
        "disadvantages": "Lower throughput.",
        "provider": "Meta"
    },
    "llama-3.1-8b-instant": {
        "advantages": "High-speed for real-time apps.",
        "disadvantages": "Less accurate for complex tasks.",
        "provider": "Meta"
    },
    "meta-llama/llama-guard-4-12b": {
        "advantages": "Optimized for safety and guardrailing.",
        "disadvantages": "Specialized for safety, not general chat.",
        "provider": "Meta"
    },
    "openai/gpt-oss-120b": {
        "advantages": "120B params, browser search, code execution.",
        "disadvantages": "Slower speed.",
        "provider": "OpenAI"
    },
    "openai/gpt-oss-20b": {
        "advantages": "20B params, browser search, code execution.",
        "disadvantages": "Smaller model.",
        "provider": "OpenAI"
    },
}

# Vision Models
VISION_MODELS = {
    "blip": {
        "name": "BLIP (Fast)",
        "description": "Fast image captioning, good for basic analysis"
    },
    "blip2": {
        "name": "BLIP-2 (Advanced)",
        "description": "Advanced understanding, slower but more detailed"
    },
    "git": {
        "name": "GIT (Detailed)",
        "description": "Detailed image descriptions, best for complex images"
    }
}

# Default Settings
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.7
