import fitz
from time import sleep
from newspaper import Article
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

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
def process_content(content):
    document = Document(page_content=content)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    semantic_text_splitter = SemanticChunker(embeddings)
    chunks = semantic_text_splitter.split_documents([document])

    return chunks

# Function to create a persistent vector store with ChromaDB
def create_vector_store(chunks, collection_name):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    return vector_store

def keyword_search(query, chunks, top_n=3):
    """
    Performs keyword search using BM25.
    """
    tokenized_chunks = [chunk.page_content.split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split(" ")
    
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get the top_n scores and their indices
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_n]
    
    # Get the corresponding chunks
    top_chunks = [chunks[i] for i in top_n_indices]
    
    return top_chunks

def hybrid_search(query, vector_store, chunks, top_n=3, alpha=0.5):
    """
    Performs hybrid search combining vector and keyword search.
    """
    # Vector search
    vector_results = vector_store.similarity_search_with_score(query, k=top_n)
    vector_docs = [doc for doc, _ in vector_results]
    
    # Keyword search
    keyword_docs = keyword_search(query, chunks, top_n)
    
    # Combine and re-rank (simple weighted combination)
    combined_docs = {}
    
    for doc in vector_docs:
        combined_docs[doc.page_content] = combined_docs.get(doc.page_content, 0) + alpha
        
    for doc in keyword_docs:
        combined_docs[doc.page_content] = combined_docs.get(doc.page_content, 0) + (1 - alpha)
        
    # Sort by combined score and return top_n documents
    sorted_docs_content = sorted(combined_docs.items(), key=lambda item: item[1], reverse=True)
    
    final_docs = []
    for content, _ in sorted_docs_content:
        # Find the original Document object
        for chunk in chunks:
            if chunk.page_content == content:
                final_docs.append(chunk)
                break
        if len(final_docs) == top_n:
            break
            
    return final_docs

def keyword_search(query, chunks, top_n=3):
    """
    Performs keyword search using BM25.
    """
    tokenized_chunks = [chunk.page_content.split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split(" ")
    
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get the top_n scores and their indices
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_n]
    
    # Get the corresponding chunks
    top_chunks = [chunks[i] for i in top_n_indices]
    
    return top_chunks

def hybrid_search(query, vector_store, chunks, top_n=3, alpha=0.5):
    """
    Performs hybrid search combining vector and keyword search.
    """
    # Vector search
    vector_results = vector_store.similarity_search_with_score(query, k=top_n)
    vector_docs = [doc for doc, _ in vector_results]
    
    # Keyword search
    keyword_docs = keyword_search(query, chunks, top_n)
    
    # Combine and re-rank (simple weighted combination)
    combined_docs = {}
    
    for doc in vector_docs:
        combined_docs[doc.page_content] = combined_docs.get(doc.page_content, 0) + alpha
        
    for doc in keyword_docs:
        combined_docs[doc.page_content] = combined_docs.get(doc.page_content, 0) + (1 - alpha)
        
    # Sort by combined score and return top_n documents
    sorted_docs_content = sorted(combined_docs.items(), key=lambda item: item[1], reverse=True)
    
    final_docs = []
    for content, _ in sorted_docs_content:
        # Find the original Document object
        for chunk in chunks:
            if chunk.page_content == content:
                final_docs.append(chunk)
                break
        if len(final_docs) == top_n:
            break
            
    return final_docs

def format_docs(docs, max_chars_per_doc=500):
    """
    Formats the retrieved documents for display, truncating each document's content.
    """
    formatted_string = ""
    for i, doc in enumerate(docs):
        truncated_content = doc.page_content[:max_chars_per_doc]
        if len(doc.page_content) > max_chars_per_doc:
            truncated_content += "..."
        formatted_string += f"Source {i+1}: {truncated_content}\n\n"
    return formatted_string

