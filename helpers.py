import fitz
from time import sleep
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        pdf_document = fitz.open("pdf", file.read())
        all_text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            all_text += page.get_text()
        pdf_document.close()
        return all_text
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to extract text from a URL
def extract_text_from_url(url, retries=3):

    for attempt in range(retries):
        try:
            article = Article(url)
            article.download()
            article.parse()

            if len(article.text.strip()) == 0:
                raise ValueError("No text extracted. The article might be behind a paywall or inaccessible.")

            return article.text
        except Exception as e:
            if attempt < retries - 1:
                sleep(2)
                continue
            return f"Error processing URL after {retries} attempts: {e}"

# Function to split content into chunks
def process_content(content, chunk_size, chunk_overlap):
    document = Document(page_content=content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents([document])
    return chunks

# Function to create an in-memory vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    return vector_store