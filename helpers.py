import fitz
from time import sleep
from newspaper import Article
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
import os
import io
import fitz  # PyMuPDF

# Import multimodal helpers at module level to avoid conditional imports
try:
    from multimodal_helpers import FreeMultimodalEmbedder
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("Warning: Multimodal helpers not available. Image extraction will be disabled.")

# Function to extract text from a PDF file with optional image extraction
def extract_text_from_pdf(file, extract_images=False):
    try:
        # Read the file content once
        file_content = file.read()
        
        # Extract text
        pdf_document = fitz.open("pdf", file_content)
        all_text = ""
        images = []
        
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            all_text += page.get_text()
                
        pdf_document.close()
        
        # Extract images if requested using the same file content
        if extract_images and MULTIMODAL_AVAILABLE:
            embedder = FreeMultimodalEmbedder()
            # Create a new file-like object from the content
            file_like = io.BytesIO(file_content)
            page_images = embedder.extract_images_from_pdf(file_like)
            images.extend(page_images)
        elif extract_images and not MULTIMODAL_AVAILABLE:
            print("Warning: Multimodal features not available. Skipping image extraction.")
        
        if extract_images:
            return all_text, images
        return all_text
        
    except Exception as e:
        if extract_images:
            return f"Error reading PDF: {e}", []
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

def format_multimodal_results(results: List[Dict], max_chars_per_result=500):
    """
    Format multimodal search results (text + images) for display
    """
    formatted_string = ""
    text_count = 0
    image_count = 0
    
    for result in results:
        content_item = result['content']
        content_type = content_item['type']
        
        if content_type == 'text':
            text_count += 1
            truncated_content = content_item['content'][:max_chars_per_result]
            if len(content_item['content']) > max_chars_per_result:
                truncated_content += "..."
            formatted_string += f"Text Source {text_count}: {truncated_content}\n\n"
            
        elif content_type == 'image':
            image_count += 1
            image_desc = content_item['content'][:max_chars_per_result]
            if len(content_item['content']) > max_chars_per_result:
                image_desc += "..."
            formatted_string += f"Image Source {image_count}: {image_desc}\n\n"
    
    return formatted_string

def hybrid_multimodal_search(query: str, vector_store, chunks, multimodal_store=None, top_n=3, alpha=0.5, include_images=True):
    """
    Enhanced hybrid search that combines traditional text search with multimodal search
    """
    # Traditional hybrid search
    text_results = hybrid_search(query, vector_store, chunks, top_n, alpha)
    
    # Multimodal search if available
    if multimodal_store and include_images:
        try:
            # Use cached embedder if available
            import streamlit as st
            if hasattr(st.session_state, 'multimodal_embedder') and st.session_state.multimodal_embedder:
                embedder = st.session_state.multimodal_embedder
            else:
                from multimodal_helpers import FreeMultimodalEmbedder
                embedder = FreeMultimodalEmbedder()
            
            multimodal_results = embedder.search_multimodal(query, multimodal_store, top_k=top_n)
            
            # Convert multimodal results to Document format for consistency
            multimodal_docs = []
            for result in multimodal_results:
                content_item = result['content']
                if content_item['type'] == 'image':
                    # Create document from image description
                    doc = Document(
                        page_content=content_item['content'],
                        metadata={
                            'type': 'image',
                            'score': result['score'],
                            'image_id': content_item['id'],
                            'image_data': content_item.get('image_data')
                        }
                    )
                    multimodal_docs.append(doc)
            
            # Combine text and image results based on scores
            # Prioritize by relevance score rather than type
            all_results = []
            
            # Add text results with default score
            for doc in text_results:
                all_results.append((doc, 0.5))  # Default score for text
            
            # Add image results with their actual scores
            for doc in multimodal_docs:
                score = doc.metadata.get('score', 0.0)
                all_results.append((doc, score))
            
            # Sort by score and return top results
            all_results.sort(key=lambda x: x[1], reverse=True)
            final_results = [doc for doc, _ in all_results[:top_n]]
            
            return final_results
            
        except Exception as e:
            print(f"Error in multimodal search: {e}")
            return text_results
    
    return text_results

