# Cortex AI: Multi-Model Insights Hub

Cortex AI: Multi-Model Insights Hub is an advanced platform that leverages cutting-edge AI to empower your research, analysis, and data exploration. By integrating multiple Large Language Models (LLMs) with a sophisticated Retrieve-and-Generate (RAG) system, Cortex AI Hub enables you to extract insights from documents and data with ease and precision.

Deployed on **Streamlit Cloud**, this platform offers an intuitive interface for interactive document analysis, dynamic data exploration, and multi-model comparisons—all in one place.

---

## 📌 **Features**

- **📂 Document Upload**: Upload PDFs or enter URLs containing articles or documents.
- **📖 Intelligent Text Extraction**: Automatically extract and process text from uploaded documents or web pages.
- **📑 Dynamic Chunking & Embedding**: Split the extracted text into manageable chunks and generate embeddings for efficient search and retrieval.
- **⚡ Robust Vector Database**: Store embeddings in a high-performance vector database for rapid, context-driven searches.
- **🧠 Multi-LLM Integration**: Leverage a suite of LLMs (including DeepSeek, Qwen, Gemma, Llama, Mistral, and Mixtral) to generate context-aware, accurate responses.
- **💬 Persistent Conversation History**: Maintain conversation continuity with stored dialogue history.
- **🔄 Flexible Model Selection**: Easily switch between different LLMs or compare outputs by selecting multiple models simultaneously.
- **📊 Data Analysis Chatbot**: Interactively visualize and analyze your data with AI-driven assistance on a dedicated page.
- **🔍 Search Agent Bot**: Discover insights from research articles, recent news, and general queries using an AI-powered search assistant.
- **🚀 Scalable Streamlit Cloud Deployment**: Access the platform through a responsive, cloud-hosted interface for global reach.

---

## 🚀 **Supported Models**

| Model                         | RPM | RPD    | Tokens/Min | Tokens/Day | Advantages                                                                      | Disadvantages                                     |
| ----------------------------- | --- | ------ | ---------- | ---------- | ------------------------------------------------------------------------------- | ------------------------------------------------- |
| deepseek-r1-distill-llama-70b | 30  | 1,000  | 6,000      | Unlimited  | Highly optimized for low latency with unlimited token capacity.                 | Limited daily requests.                           |
| qwen-2.5-32b                  | 30  | 14,400 | 10,000     | 500,000    | Powerful 32B model optimized for long-context comprehension.                    | Requires more computational resources.            |
| gemma2-9b-it                  | 30  | 14,400 | 15,000     | 500,000    | High token throughput, suitable for fast inference on large-scale tasks.        | Limited versatility compared to larger models.    |
| llama-3.1-8b-instant          | 30  | 14,400 | 20,000     | 500,000    | High-speed processing with large token capacity, ideal for real-time apps.      | Less accurate for complex reasoning.              |
| llama-3.3-70b-versatile       | 30  | 1,000  | 6,000      | 100,000    | Versatile model optimized for high accuracy in diverse scenarios.               | Lower throughput compared to smaller models.      |
| llama3-70b-8192               | 30  | 14,400 | 6,000      | 500,000    | Long-context capabilities, perfect for detailed research articles.              | Moderate speed and accuracy for shorter tasks.    |
| llama3-8b-8192                | 30  | 14,400 | 20,000     | 500,000    | High-speed inference with extensive long-context support.                       | Slightly less accurate for complex reasoning.     |
| mistral-saba-24b              | 30  | 7,000  | 7,000      | 250,000    | Excellent for multi-turn conversations and effective retrieval augmentation.    | Limited token capacity compared to larger models. |
| mixtral-8x7b-32768            | 30  | 14,400 | 5,000      | 500,000    | Optimized for processing long documents with superior contextual understanding. | Lower token throughput.                           |

---

## How It Works

### Article Chatbot

1. **User Input**: Upload a PDF file or provide a URL to an article or document.
2. **Text Extraction**: The system automatically extracts text from your document or webpage.
3. **Chunking & Embeddings**: The extracted text is segmented into smaller chunks, and embeddings are generated using advanced LLM techniques.
4. **Vector Store**: Embeddings are stored in a high-performance vector database (e.g., ChromaDB, FAISS, Pinecone, or DocArrayInMemorySearch from LangChain) for rapid retrieval.
5. **Querying & Response Generation**: Ask questions through the chatbot interface, and the system retrieves relevant context and generates answers using one or more selected LLMs.
6. **Conversation History**: Maintain dialogue continuity with stored conversation history.
7. **Model Switching & Comparison**: Easily switch between LLM models or run them concurrently to compare their outputs.

### Data Analysis Chatbot

1. **Data Upload**: Upload data files in various formats (CSV, Excel, JSON, Parquet) via the dedicated Data Analysis page.
2. **Data Processing**: The app processes and previews your data for further analysis.
3. **Interactive Analysis**: Use the AI-driven chat interface to ask questions, generate visualizations, and analyze your data.
4. **Dynamic Visualizations**: Receive Python code for generating plots and visualizations, ensuring that your data insights are both accurate and engaging.

### Search Agent Bot

1. **User Query**: Enter a question or topic (e.g., recent news, research papers, or definitions like "What is MCP?") via the dedicated Search Agent page.
2. **Tool Integration**: The bot leverages external tools (e.g., Arxiv, Wikipedia, Tavily) to fetch relevant, up-to-date information.
3. **Context-Aware Responses**: The system processes queries with multi-LLM integration, delivering concise, accurate answers based on recent data (e.g., past 30 days for news).
4. **Conversation Continuity**: Maintains dialogue history for context-aware follow-up questions.
5. **Model Selection**: Choose from multiple LLMs to tailor the response style and depth, with options to retry searches for better results.

![System Diagram](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/images/Uml%20Diagram.png)

![System Diagram](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/images/Uml_diagram_2.png)

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
git clone https://github.com/RobinMillford/Multi-Model-RAG-Powered-Article-Chatbot.git
cd Multi-Model-RAG-Powered-Article-Chatbot
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
streamlit run app.py
```

### Deployed on Streamlit Cloud

Cortex AI Hub is live on Streamlit Cloud. Explore the demo here:

[Streamlit Deployed Demo](https://cortex-ai-multi-model-insights-app.streamlit.app/)

![Demo Image 1](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/images/Llama3-RAG-Chatbot-1.png)
![Demo Image 2](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/images/Llama3-RAG-Chatbot-2.png)
![Demo Image 3](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/images/Data-Analysis-Chatbot-with-Multi-Model-Responses-·-Streamlit.png)

---

## Usage

1. **For Articles**: Upload a PDF file or provide a URL of an article. The system extracts text, processes it into chunks, and allows you to ask questions through the chatbot interface.
2. **For Data Analysis**: Navigate to the Data Analysis page to upload your data files, preview your dataset, and interact with an AI-driven assistant for visualizations and analysis.
3. **For Search Queries**: Use the Search Agent Bot to explore research papers, recent news, or general topics, with AI-powered answers tailored to your query.

---

## Contributing

Contributions are welcome! To contribute:

1. **Fork the Repository** on GitHub.
2. Clone your fork:

   ```bash
   git clone https://github.com/RobinMillford/Multi-Model-RAG-Powered-Article-Chatbot.git
   ```

3. Create a new branch:

   ```bash
   git checkout -b feature-name
   ```

4. Make your changes and commit them:

   ```bash
   git add .
   git commit -m "Description of the changes"
   ```

5. Push your branch:

   ```bash
   git push origin feature-name
   ```

6. Open a **Pull Request** to the `main` branch of the original repository.

---

## License

This project is licensed under the **AGPL-3.0 license**. See the [LICENSE](LICENSE) file for details.