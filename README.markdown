# Cortex AI: Multi-Model Insights Hub

Cortex AI: Multi-Model Insights Hub is an advanced platform that leverages cutting-edge AI to empower your research, and analysis. By integrating multiple Large Language Models (LLMs) with a sophisticated Retrieve-and-Generate (RAG) system, Cortex AI Hub enables you to extract insights from documents and data with ease and precision.

Deployed on **Streamlit Cloud**, this platform offers an intuitive interface for interactive document analysis, dynamic data exploration, and multi-model comparisons—all in one place.

---

## 📌 **Features**

- **✨ Modern & Intuitive UI:** A sleek, dark-themed user interface across the entire application for an enhanced user experience.
- **📂 Multi-Document Management:** Upload and manage a PDF file or provide URL to online articles.
- **📖 Intelligent Text Extraction:** Automatically extract and process text from uploaded documents or web pages.
- **📑 Dynamic Chunking & Embedding:** Split extracted text into manageable chunks and generate embeddings for efficient search and retrieval.
- **⚡ Persistent Vector Database (ChromaDB):** Embeddings are stored in a high-performance, persistent ChromaDB vector database, eliminating the need to re-process documents.
- **🔍 Hybrid Search (Semantic + BM25):** Combines semantic search with keyword (BM25) search for highly accurate and relevant context retrieval.
- **⚙️ Configurable RAG Parameters:** Adjust the number of retrieved chunks and the maximum characters per chunk to fine-tune context retrieval for different query types.
- **💬 Persistent Conversation History:** Maintain dialogue continuity with stored dialogue history, with automatic truncation to manage token limits.
- **🧠 Multi-LLM Integration:** Leverage a suite of LLMs (including DeepSeek, Qwen, Llama, and OpenAI models) to generate context-aware, accurate responses.
- **✅ Accurate Citations & Source Highlighting:** Answers include precise citations, linking directly to the numbered source chunks from the document for transparency.
- **🔄 Flexible Model Selection:** Easily switch between different LLMs or compare outputs by selecting multiple models simultaneously.
- **🔍 Search Agent Bot:** Discover insights from research articles, recent news, and general queries using an AI-powered search assistant.
- **🚀 Scalable Streamlit Cloud Deployment:** Access the platform through a responsive, cloud-hosted interface for global reach.

---

## 🚀 **Supported Models**

| Model                         | Provider      | Advantages                                   | Disadvantages                    |
| :---------------------------- | :------------ | :------------------------------------------- | :------------------------------- |
| llama-3.3-70b-versatile       | Meta          | High accuracy in diverse scenarios.          | Lower throughput.                |
| llama-3.1-8b-instant          | Meta          | High-speed for real-time apps.               | Less accurate for complex tasks. |
| deepseek-r1-distill-llama-70b | DeepSeek      | Low latency, no token limits.                | Limited daily requests.          |
| qwen/qwen3-32b                | Alibaba Cloud | Powerful 32B model for long-context.         | Computationally intensive.       |
| openai/gpt-oss-120b           | OpenAI        | 120B params, browser search, code execution. | Slower speed.                    |
| openai/gpt-oss-20b            | OpenAI        | 20B params, browser search, code execution.  | Smaller model.                   |

---

## How It Works

### Article Chatbot (RAG-Powered)

1.  **User Input**: Upload PDF files or provide URLs to articles/documents. The UI allows for multi-document selection.
2.  **Text Extraction & Processing**: The system automatically extracts text from your documents/webpages, splits it into manageable chunks, and generates embeddings.
3.  **Persistent Vector Store**: Embeddings are stored in a persistent ChromaDB vector database. Existing document stores are loaded, new ones are created and saved.
4.  **Hybrid Retrieval**: When you ask a question, the system performs both semantic (vector) search and keyword (BM25) search across all selected documents to retrieve the most relevant context, configurable via sidebar parameters.
5.  **Querying & Response Generation**: The system retrieves relevant context and generates answers using one or more selected LLMs. Answers include precise citations to the source documents.
6.  **Conversation History**: Maintain dialogue continuity with stored conversation history, automatically truncated to fit model token limits.
5.  **Querying & Response Generation**: The system retrieves relevant context and generates answers using one or more selected LLMs. Answers include precise citations to the source documents.
6.  **Conversation History**: Maintain dialogue continuity with stored conversation history.
7.  **Model Switching & Comparison**: Easily switch between LLM models or run them concurrently to compare their outputs.

### Search Agent Bot

1.  **User Query**: Enter a question or topic (e.g., recent news, research papers, or definitions like "What is MCP?") via the dedicated Search Agent page.
2.  **Tool Integration**: The bot leverages external tools (e.g., Arxiv, Wikipedia, Tavily) to fetch relevant, up-to-date information.
3.  **Context-Aware Responses**: The system processes queries with multi-LLM integration, delivering concise, accurate answers based on recent data (e.g., past 30 days for news).
4.  **Conversation Continuity**: Maintains dialogue history for context-aware follow-up questions.
5.  **Model Selection**: Choose from multiple LLMs to tailor the response style and depth, with options to retry searches for better results.

---

## Getting Started

Follow these steps to set up and run Cortex AI: Multi-Model Insights Hub locally or contribute to its development.

### Prerequisites

- **Python 3.12**
- **pip** for package management
- A **.env** file with your API keys (see below)
- **ChatGroq API Key** for LLM integration

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/RobinMillford/Cortex-AI-Multi-Model-Insights-Hub.git
cd Cortex-AI-Multi-Model-Insights-Hub
```

### Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the root directory and add your ChatGroq API key:

```
GROQ_API_KEY=your_chatgroq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Replace `your_chatgroq_api_key` with your actual API key.

### Running the App Locally

Start the application locally with:

```bash
streamlit run Main_Page.py
```

### Deployed on Streamlit Cloud

Cortex AI Hub is live on Streamlit Cloud. Explore the demo here:

[Streamlit Deployed Demo](https://cortex-ai-multi-model-insights-app.streamlit.app/)

---

## Usage

1.  **For Articles**: Upload PDF files or provide URLs. The system extracts text, processes it into chunks, and allows you to ask questions through the chatbot interface.
2.  **For Search Queries**: Use the Search Agent Bot to explore research papers, recent news, or general topics, with AI-powered answers tailored to your query.

---

## Contributing

Contributions are welcome! To contribute:

1.  **Fork the Repository** on GitHub.
2.  Clone your fork:

    ```bash
    git clone https://github.com/RobinMillford/Multi-Model-RAG-Powered-Article-Chatbot.git
    ```

3.  Create a new branch:

    ```bash
    git checkout -b feature-name
    ```

4.  Make your changes and commit them:

    ```bash
    git add .
    git commit -m "Description of the changes"
    ```

5.  Push your branch:

    ```bash
    git push origin feature-name
    ```

6.  Open a **Pull Request** to the `main` branch of the original repository.

---

## License

This project is licensed under the **AGPL-3.0 license**. See the [LICENSE](LICENSE) file for details.
