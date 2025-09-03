"""
Multimodal RAG Helper Functions
Implements image extraction, analysis, and embedding using free open-source models
"""

import fitz  # PyMuPDF
import torch
import numpy as np
import faiss
from PIL import Image
import io
import base64
import streamlit as st
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    GitProcessor, GitForCausalLM
)
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Tuple, Optional
import hashlib
import json

# Global cache for models to avoid reloading
_model_cache = {}
_embedder_instance = None

class MultimodalEmbedder:
    """
    Multimodal embedder using open-source models from Hugging Face
    No paid API dependencies - all processing done locally
    Uses caching to avoid reloading models
    """
    
    def __init__(self, vision_model: str = "blip", device: str = None):
        global _embedder_instance
        
        # Return cached instance if same model requested
        if _embedder_instance is not None and _embedder_instance.vision_model_name == vision_model:
            self.__dict__.update(_embedder_instance.__dict__)
            return
            
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model_name = vision_model
        
        # Load CLIP for unified text-image embeddings (free)
        if 'clip' not in _model_cache:
            print(f"Loading CLIP model on {self.device}...")
            _model_cache['clip'] = {
                'model': CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device),
                'processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            }
        
        self.clip_model = _model_cache['clip']['model']
        self.clip_processor = _model_cache['clip']['processor']
        
        # Load vision-language model for image analysis (free)
        self._load_vision_model(vision_model)
        
        # Text embedder for hybrid search
        if 'text_embedder' not in _model_cache:
            print("Loading text embedder...")
            _model_cache['text_embedder'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.text_embedder = _model_cache['text_embedder']
        
        # Cache this instance
        _embedder_instance = self
        
    def _load_vision_model(self, model_name: str):
        """Load the specified vision model with caching"""
        model_key = f'vision_{model_name}'
        
        if model_key not in _model_cache:
            print(f"Loading {model_name} vision model...")
            
            if model_name == "blip":
                _model_cache[model_key] = {
                    'processor': BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
                    'model': BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
                }
            elif model_name == "blip2":
                _model_cache[model_key] = {
                    'processor': Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b"),
                    'model': Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.device)
                }
            elif model_name == "git":
                _model_cache[model_key] = {
                    'processor': GitProcessor.from_pretrained("microsoft/git-base"),
                    'model': GitForCausalLM.from_pretrained("microsoft/git-base").to(self.device)
                }
            else:
                raise ValueError(f"Unsupported vision model: {model_name}")
        
        self.vision_processor = _model_cache[model_key]['processor']
        self.vision_model = _model_cache[model_key]['model']
    
    def extract_images_from_pdf(self, pdf_file, min_size: Tuple[int, int] = (100, 100)) -> List[Dict]:
        """
        Extract images from PDF with metadata
        """
        try:
            pdf_document = fitz.open("pdf", pdf_file.read())
            images = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        # Skip very small images (likely decorative)
                        if pix.width < min_size[0] or pix.height < min_size[1]:
                            pix = None
                            continue
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Convert to base64 for storage
                            buffered = io.BytesIO()
                            img_pil.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()
                            
                            images.append({
                                'page': page_num + 1,
                                'index': img_index,
                                'image': img_pil,
                                'base64': img_base64,
                                'size': (pix.width, pix.height),
                                'id': f"page_{page_num+1}_img_{img_index}"
                            })
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            pdf_document.close()
            print(f"Extracted {len(images)} images from PDF")
            return images
            
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
            return []
    
    def analyze_image(self, image: Image.Image, context: str = "") -> str:
        """
        Analyze image content using the loaded vision model with enhanced prompts for charts/graphs
        """
        try:
            # Generate multiple descriptions for better understanding
            base_description = ""
            
            if self.vision_model_name == "blip":
                inputs = self.vision_processor(image, return_tensors="pt").to(self.device)
                out = self.vision_model.generate(**inputs, max_length=50)
                base_description = self.vision_processor.decode(out[0], skip_special_tokens=True)
                
                # Try with specific prompts for different content types
                prompts = [
                    "a chart showing",
                    "a graph displaying", 
                    "data visualization of",
                    "an infographic about"
                ]
                
                descriptions = [base_description]
                for prompt in prompts:
                    try:
                        inputs = self.vision_processor(image, text=prompt, return_tensors="pt").to(self.device)
                        out = self.vision_model.generate(**inputs, max_length=50)
                        desc = self.vision_processor.decode(out[0], skip_special_tokens=True)
                        if desc and desc != base_description:
                            descriptions.append(desc)
                    except:
                        continue
                
                # Combine descriptions
                caption = ". ".join(descriptions[:3])  # Use top 3 descriptions
                
            elif self.vision_model_name == "blip2":
                inputs = self.vision_processor(image, return_tensors="pt").to(self.device)
                generated_ids = self.vision_model.generate(**inputs, max_length=100)  # Longer for more detail
                caption = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
            elif self.vision_model_name == "git":
                inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
                generated_ids = self.vision_model.generate(pixel_values=inputs.pixel_values, max_length=100)
                caption = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Enhance caption with context and identify chart/graph elements
            enhanced_caption = caption
            
            # Add chart/graph specific terms if detected
            chart_indicators = ['chart', 'graph', 'plot', 'data', 'bar', 'line', 'pie', 'axis', 'trend']
            if any(indicator in caption.lower() for indicator in chart_indicators):
                enhanced_caption = f"Data visualization: {caption}"
            
            if context:
                enhanced_caption = f"Image from document (page context: {context[:50]}...): {enhanced_caption}"
            else:
                enhanced_caption = f"Document image: {enhanced_caption}"
                
            return enhanced_caption
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return "Visual content: Unable to analyze image details"
    
    def create_unified_embeddings(self, text_chunks: List[str], image_data: List[Dict]) -> Dict:
        """
        Create unified embeddings for both text and images using CLIP
        """
        all_embeddings = []
        all_content = []
        content_types = []
        
        # Process text chunks
        for i, text in enumerate(text_chunks):
            # Create CLIP text embedding
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                text_embedding = text_features.cpu().numpy().flatten()
            
            all_embeddings.append(text_embedding)
            all_content.append({
                'type': 'text',
                'content': text,
                'index': i,
                'id': f"text_{i}"
            })
            content_types.append('text')
        
        # Process images
        for img_data in image_data:
            try:
                # Analyze image first
                image_description = self.analyze_image(img_data['image'])
                
                # Create CLIP image embedding
                inputs = self.clip_processor(images=img_data['image'], return_tensors="pt")
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    image_embedding = image_features.cpu().numpy().flatten()
                
                all_embeddings.append(image_embedding)
                all_content.append({
                    'type': 'image',
                    'content': image_description,
                    'image_data': img_data,
                    'id': img_data['id']
                })
                content_types.append('image')
                
            except Exception as e:
                print(f"Error processing image {img_data['id']}: {e}")
                continue
        
        # Create FAISS index
        if all_embeddings:
            embeddings_array = np.array(all_embeddings).astype('float32')
            dimension = embeddings_array.shape[1]
            
            # Use IndexFlatIP for cosine similarity
            index = faiss.IndexFlatIP(dimension)
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            index.add(embeddings_array)
            
            return {
                'index': index,
                'content': all_content,
                'content_types': content_types,
                'embeddings': embeddings_array
            }
        else:
            return None
    
    def search_multimodal(self, query: str, multimodal_store: Dict, top_k: int = 5) -> List[Dict]:
        """
        Search both text and images using the query
        """
        if not multimodal_store:
            return []
        
        try:
            # Create query embedding using CLIP text encoder
            inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                query_features = self.clip_model.get_text_features(**inputs)
                query_embedding = query_features.cpu().numpy().astype('float32')
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = multimodal_store['index'].search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    content_item = multimodal_store['content'][idx]
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'content': content_item,
                        'type': content_item['type']
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in multimodal search: {e}")
            return []

def save_multimodal_store(store: Dict, filepath: str):
    """Save multimodal store to disk"""
    try:
        # Convert FAISS index to bytes for storage
        store_copy = store.copy()
        index_bytes = faiss.serialize_index(store['index'])
        store_copy['index_bytes'] = index_bytes
        del store_copy['index']  # Remove the actual index object
        
        with open(filepath, 'wb') as f:
            pickle.dump(store_copy, f)
        print(f"Multimodal store saved to {filepath}")
    except Exception as e:
        print(f"Error saving multimodal store: {e}")

def load_multimodal_store(filepath: str) -> Dict:
    """Load multimodal store from disk"""
    try:
        with open(filepath, 'rb') as f:
            store = pickle.load(f)
        
        # Reconstruct FAISS index from bytes
        index = faiss.deserialize_index(store['index_bytes'])
        store['index'] = index
        del store['index_bytes']
        
        print(f"Multimodal store loaded from {filepath}")
        return store
    except Exception as e:
        print(f"Error loading multimodal store: {e}")
        return None

def get_collection_hash(content: str, has_images: bool = False) -> str:
    """Generate hash for collection identification"""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    if has_images:
        content_hash += "_multimodal"
    return content_hash

# Backward compatibility alias
FreeMultimodalEmbedder = MultimodalEmbedder