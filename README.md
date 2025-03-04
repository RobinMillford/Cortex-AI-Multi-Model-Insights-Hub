# Multi-Model RAG-Powered Article Chatbot

This project creates a **Retrieve-and-Generate (RAG)** powered chatbot for summarizing and interacting with articles. The system processes articles provided as PDFs or URLs, extracts text, splits the content into chunks, generates embeddings, and stores them in a vector database. The chatbot then responds to user queries using a **Large Language Model (LLM)** to generate context-aware answers.

The project is deployed on **Streamlit Cloud**, allowing users to interact with the chatbot via an easy-to-use interface.

---

## 📌 **Features**

- **📂 PDF/URL Upload**: Upload a PDF file or provide a URL containing an article.
- **📖 Text Extraction**: The app extracts text content from the uploaded document or webpage.
- **📑 Chunking & Embeddings**: The extracted text is split into smaller chunks, and embeddings are generated for efficient search.
- **⚡ Vector Database**: The embeddings are stored in a vector database for fast retrieval.
- **🧠 LLM Integration**: Queries are processed using the **ChatGroq** LLM for generating contextually relevant answers.
- **💬 Conversation History**: Users can continue the chat with the chatbot, as it stores conversation history and context.
- **🔄 Changeable LLM Models**: There is an option to switch between different LLM models for generating responses.
- **🤖 Multi-Model Chat**: Users can **select multiple models** at once and compare responses side by side.
- **🚀 Deployable on Streamlit**: The app is deployed on **Streamlit Cloud** for easy access and interaction.

---

## 🚀 **Supported Models**

| Model                         | Requests per Minute | Requests per Day | Tokens per Minute | Tokens per Day | Advantages                                                            | Disadvantages                        |
| ----------------------------- | ------------------- | ---------------- | ----------------- | -------------- | --------------------------------------------------------------------- | ------------------------------------ |
| deepseek-r1-distill-llama-70b | 30                  | 1,000            | 6,000             | Unlimited      | Unlimited token capacity, low latency                                 | Limited daily requests               |
| gemma2-9b-it                  | 30                  | 14,400           | 15,000            | 500,000        | High throughput, suitable for large-scale inference                   | Limited versatility                  |
| llama-3.1-8b-instant          | 30                  | 14,400           | 20,000            | 500,000        | High-speed processing, great for real-time applications               | Less accurate for complex reasoning  |
| llama-3.3-70b-specdec         | 30                  | 1,000            | 6,000             | 100,000        | Precision-focused for decision-making                                 | Limited token capacity               |
| llama-3.3-70b-versatile       | 30                  | 1,000            | 6,000             | 100,000        | Versatile for high-accuracy scenarios                                 | Low throughput                       |
| llama3-70b-8192               | 30                  | 14,400           | 6,000             | 500,000        | Long-context, ideal for extended conversations                        | Moderate speed and accuracy          |
| llama3-8b-8192                | 30                  | 14,400           | 20,000            | 500,000        | High-speed inference with long-context support                        | Slightly less accurate for reasoning |
| mixtral-8x7b-32768            | 30                  | 14,400           | 5,000             | 500,000        | Supports long document processing for better contextual understanding | Lower token throughput               |

---

## How It Works

1. **User Input**: Users upload a PDF file or provide a URL.
2. **Text Extraction**: The app extracts text from the document using **Text Extraction** functions.
3. **Chunking & Embeddings**: The extracted text is split into smaller chunks, and embeddings are created using the **ChatGroq** model.
4. **Vector Store**: Embeddings are stored in a vector database (e.g., **ChromaDB**, **FAISS**, **Pinecone**, or **DocArrayInMemorySearch** from **LangChain**).
5. **Querying**: Users can ask questions through the chatbot interface.
6. **Response Generation**: The system retrieves relevant chunks from the vector database and feeds them into the selected **LLM** to generate the response.
7. **Conversation History**: The chatbot maintains conversation history, enabling users to continue their chat without losing context.
8. **Model Switching**: Users have the flexibility to switch between different LLM models as per their preferences.
9. **Final Answer**: The LLM generates an answer based on the retrieved context and returns it to the user.

![Alt Text](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/Uml%20Diagram.png)

## Getting Started

To use the chatbot locally or contribute to this repository, follow the steps below.

### Prerequisites

Before starting, ensure you have the following:

- **Python 3.12**
- **pip** for installing packages
- **.env** file for your API keys (see below)
- **ChatGroq API Key** (for using the LLM model)

### Clone the Repository

First, clone the repository to your local machine.

```bash
git clone https://github.com/RobinMillford/Multi-Model-RAG-Powered-Article-Chatbot.git
cd Multi-Model-RAG-Powered-Article-Chatbot
```

### Install Dependencies

Create a virtual environment and install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
pip install -r requirements.txt
```

### Set Up Environment Variables

You need to set your **ChatGroq API key**. Create a `.env` file in the root directory and add the following:

```
GROQ_API_KEY=your_chatgroq_api_key
```

Make sure to replace `your_chatgroq_api_key` with your actual API key from ChatGroq.

### Running the App Locally

Once the dependencies are installed and the environment variables are set up, you can start the application locally.

```bash
streamlit run app.py
```

### Deployed on Streamlit Cloud

I deployed the app on Streamlit Cloud :

1. Visit [Streamlit Deployed Demo](https://multi-model-rag-powered-article-chatbot.streamlit.app/)

![Alt Text](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/Llama3-RAG-Chatbot-1.png)

![Alt Text](https://github.com/RobinMillford/LLM-Based-Text-Summarizer/blob/main/Llama3-RAG-Chatbot-2.png)

### Usage

Once the app is running (locally or on Streamlit Cloud), follow these steps:

1. Upload a PDF file or provide a URL of an article.
2. The system will extract text from the document or URL.
3. The chatbot interface will appear, allowing you to ask questions about the content of the document.
4. The chatbot will generate a response based on the context of the document.

### Contributing

I welcome contributions! To contribute, follow these steps:

1. **Fork the repository** on GitHub.
2. Clone your forked repository:

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

5. Push your changes to your forked repository:

```bash
git push origin feature-name
```

6. Open a **Pull Request** from your branch to the `main` branch of the original repository.

### License

This project is licensed under the AGPL-3.0 license - see the [LICENSE](LICENSE) file for details.

---
