logger.error(f"Error checking collections: {e}")
                return []
            
            # Get unique file_ids
            unique_docs = {}
            
            try:
                # Use scroll API to get all points
                offset = None
                has_more = True
                
                while has_more:
                    response = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=100,
                        with_payload=["payload.metadata.file_id", "payload.metadata.filename"],
                        with_vectors=False,
                        offset=offset
                    )
                    
                    points = response[0]
                    has_more = len(points) > 0
                    
                    if has_more:
                        offset = points[-1].id
                        
                        for point in points:
                            if hasattr(point, 'payload') and "metadata" in point.payload:
                                metadata = point.payload.get("metadata", {})
                                file_id = metadata.get("file_id")
                                filename = metadata.get("filename", "Unknown")
                                
                                if file_id and file_id not in unique_docs:
                                    unique_docs[file_id] = filename
            
            except Exception as e:
                logger.error(f"Error scrolling documents: {e}")
                logger.error(traceback.format_exc())
            
            return [{"id": k, "name": v} for k, v in unique_docs.items()]
            
        except Exception as e:
            logger.error(f"Error getting document IDs: {e}")
            logger.error(traceback.format_exc())
            return []
            
    def delete_document(self, file_id: str) -> bool:
        """Delete a document from the collection."""
        try:
            # Check if collection exists
            try:
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                if self.collection_name not in collection_names:
                    logger.warning(f"Collection {self.collection_name} does not exist, nothing to delete")
                    return False
            except Exception as e:
                logger.error(f"Error checking collections for deletion: {e}")
                return False
                
            # Create a filter to match points with this file_id
            filter_param = models.Filter(
                must=[
                    models.FieldCondition(
                        key="payload.metadata.file_id",
                        match=models.MatchValue(value=file_id)
                    )
                ]
            )
            
            # Execute deletion
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=filter_param)
                )
                
                logger.info(f"✅ Deleted document with ID {file_id} from Qdrant")
                return True
            except Exception as del_error:
                logger.error(f"Error during deletion operation: {del_error}")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            logger.error(traceback.format_exc())
            return False


class RAGManager:
    """Main class to manage the RAG application."""
    def __init__(self):
        # Initialize basic components
        self.config_manager = ConfigManager()
        self.embedding_manager = None
        self.pdf_processor = None
        self.qdrant_manager = None
        
        # Initialize components with error handling
        try:
            self.embedding_manager = EmbeddingManager(self.config_manager)
            logger.info("Embedding manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding manager: {e}")
            
        try:
            self.pdf_processor = PDFProcessor(self.config_manager)
            logger.info("PDF processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PDF processor: {e}")
            
        try:
            if self.embedding_manager:
                embedding_dimension = self.embedding_manager.embedding_dimension
                self.qdrant_manager = QdrantManager(self.config_manager, embedding_dimension)
                logger.info("Qdrant manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant manager: {e}")
    
    def _ensure_components(self):
        """Ensure all components are initialized."""
        if not self.embedding_manager:
            logger.info("Initializing embedding manager on demand")
            self.embedding_manager = EmbeddingManager(self.config_manager)
            
        if not self.pdf_processor:
            logger.info("Initializing PDF processor on demand")
            self.pdf_processor = PDFProcessor(self.config_manager)
            
        if not self.qdrant_manager and self.embedding_manager:
            logger.info("Initializing Qdrant manager on demand")
            embedding_dimension = self.embedding_manager.embedding_dimension
            self.qdrant_manager = QdrantManager(self.config_manager, embedding_dimension)
    
    def process_pdf_file(self, file_path: str, progress_callbacks=None) -> ProcessedDocument:
        """Process a PDF file and store it in the vector store."""
        # Ensure components are initialized
        self._ensure_components()
        
        try:
            # Define progress callback functions
            extraction_callback = progress_callbacks.get('extraction') if progress_callbacks else None
            embedding_callback = progress_callbacks.get('embedding') if progress_callbacks else None
            
            # 1. Process PDF to get documents and metadata
            if not self.pdf_processor:
                return ProcessedDocument(
                    filename=os.path.basename(file_path),
                    file_id="error",
                    total_pages=0,
                    chunks=[],
                    summary="Error: PDF processor not available",
                    status="error"
                )
                
            processed_doc = self.pdf_processor.process_pdf(file_path, extraction_callback)
            
            if processed_doc.status == "error" or not processed_doc.chunks:
                logger.warning(f"Error processing file or no chunks extracted: {file_path}")
                return processed_doc
            
            # 2. Generate embeddings for chunks
            if not self.embedding_manager:
                processed_doc.status = "error"
                processed_doc.summary += "\nError: Embedding manager not available"
                return processed_doc
                
            chunk_texts = [chunk.text for chunk in processed_doc.chunks]
            
            if embedding_callback:
                embedding_callback(0, len(chunk_texts), "Starting embeddings generation")
                
            embeddings = self.embedding_manager.embed_texts(chunk_texts)
            
            if embedding_callback:
                embedding_callback(len(chunk_texts), len(chunk_texts), "Completed embeddings generation")
                
            if not embeddings or len(embeddings) != len(chunk_texts):
                processed_doc.status = "error"
                processed_doc.summary += f"\nError: Failed to generate embeddings. Got {len(embeddings)} embeddings for {len(chunk_texts)} chunks."
                return processed_doc
                
            # 3. Store in Qdrant
            if not self.qdrant_manager:
                processed_doc.status = "error"
                processed_doc.summary += "\nError: Vector database manager not available"
                return processed_doc
                
            # Try storing with retries
            max_attempts = 3
            stored = False
            
            for attempt in range(max_attempts):
                try:
                    if embedding_callback:
                        embedding_callback(0, len(chunk_texts), f"Storing in database (attempt {attempt+1}/{max_attempts})")
                    
                    # Make sure collection exists before storing
                    self.qdrant_manager._setup_collection()
                    
                    stored = self.qdrant_manager.add_chunks(
                        processed_doc, 
                        embeddings, 
                        lambda current, total, msg: embedding_callback(current, total, f"{msg} (attempt {attempt+1}/{max_attempts})") if embedding_callback else None
                    )
                    
                    if stored:
                        logger.info(f"Successfully stored chunks on attempt {attempt+1}/{max_attempts}")
                        break
                    else:
                        logger.warning(f"Storage attempt {attempt+1} returned false, retrying...")
                        
                except Exception as e:
                    logger.error(f"Error in storage attempt {attempt+1}: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(1)  # Wait before retry
            
            if not stored:
                processed_doc.status = "error"
                processed_doc.summary += "\nError: Failed to store chunks in vector database after multiple attempts."
                return processed_doc
            
            logger.info(f"✅ Successfully processed file: {file_path}")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            logger.error(traceback.format_exc())
            
            # Return error document
            return ProcessedDocument(
                filename=os.path.basename(file_path),
                file_id="error",
                total_pages=0,
                chunks=[],
                summary=f"Error processing document: {str(e)}",
                status="error"
            )
    
    def query(self, query_text: str, file_ids: List[str] = None) -> Dict:
        """Query the RAG system."""
        # Ensure components are initialized
        self._ensure_components()
        
        try:
            # 1. Generate query embedding
            if not self.embedding_manager:
                return {
                    "answer": "Error: Embedding system not available",
                    "sources": []
                }
                
            query_embedding = self.embedding_manager.embed_query(query_text)
            
            if not query_embedding:
                return {
                    "answer": "Error: Failed to generate embedding for query",
                    "sources": []
                }
                
            # 2. Search Qdrant
            if not self.qdrant_manager:
                return {
                    "answer": "Error: Vector database not available",
                    "sources": []
                }
                
            results = self.qdrant_manager.search(query_embedding, filter_file_ids=file_ids)
            
            if not results:
                return {
                    "answer": "No relevant information found. Please try a different query or upload more documents.",
                    "sources": []
                }
                
            # 3. Format context for response generation
            context = "\n\n".join([f"Source {i+1}:\n{result['text']}" for i, result in enumerate(results)])
            
            # 4. Generate sources information
            sources = []
            for result in results:
                metadata = result.get("metadata", {})
                sources.append({
                    "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                    "file": metadata.get("filename", "Unknown"),
                    "page": metadata.get("page_num", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                    "score": f"{result['score']:.2f}"
                })
                
            # 5. Generate response using mocked LLM
            prompt = f"""
            Based on the following context, please answer the query.
            
            Query: {query_text}
            
            Context:
            {context}
            """
            
            answer = abc_response(prompt)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"Error processing your query: {str(e)}",
                "sources": []
            }
    
    def get_available_documents(self) -> List[Dict]:
        """Get all available documents."""
        # Ensure components are initialized
        self._ensure_components()
        
        if not self.qdrant_manager:
            logger.error("Qdrant manager not available for getting document IDs")
            return []
            
        return self.qdrant_manager.get_document_ids()
        
    def delete_document(self, file_id: str) -> bool:
        """Delete a document."""
        # Ensure components are initialized
        self._ensure_components()
        
        if not self.qdrant_manager:
            logger.error("Qdrant manager not available for deleting document")
            return False
            
        return self.qdrant_manager.delete_document(file_id)


# Mock function for LLM response
def abc_response(prompt: str) -> str:
    """
    This is a mock function to simulate an LLM response.
    In production, replace with actual LLM API call.
    """
    # Extract query from prompt
    query_match = re.search(r"Query: (.*?)\n", prompt)
    query = query_match.group(1) if query_match else "Unknown query"
    
    # Extract context
    context_match = re.search(r"Context:\s*(.*)", prompt, re.DOTALL)
    context = context_match.group(1) if context_match else ""
    
    # Generate a simple response based on available context
    if not context:
        return "I don't have enough information to answer this question."
    
    # Very simple response generation - just extract relevant sentences
    sentences = re.split(r'[.!?]\s+', context)
    query_words = set(query.lower().split())
    
    relevant_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if sentence is relevant to query
        sentence_words = set(sentence.lower().split())
        overlap = query_words.intersection(sentence_words)
        
        if len(overlap) >= 1 or any(word in sentence.lower() for word in query.lower().split()):
            relevant_sentences.append(sentence)
    
    if relevant_sentences:
        # Generate a simple response combining relevant sentences
        response = "Based on the documents, " + ". ".join(relevant_sentences[:3]) + "."
        return response
    else:
        return "I found some information but it doesn't seem to directly answer your query."import os
import re
import time
import logging
import tempfile
import traceback
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import uuid
import hashlib
import json
from datetime import datetime

# Simple vector store implementation
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Document handling
import fitz  # PyMuPDF

# Initialize directories
for directory in ["./data", "./data/embeddings", "./data/qdrant_db", "./models", "./logs", "./temp"]:
    os.makedirs(directory, exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Class to represent a chunk of a document."""
    text: str
    metadata: Dict[str, Any]


@dataclass
class ProcessedDocument:
    """Class to store information about a processed document."""
    filename: str
    file_id: str
    total_pages: int
    chunks: List[DocumentChunk]
    summary: str = ""
    status: str = "success"


class ConfigManager:
    """Class to manage configuration."""
    def __init__(self, config_path="config.txt"):
        self.config = {
            "DATA_DIR": "./data",
            "EMBEDDINGS_DIR": "./data/embeddings",
            "QDRANT_DIR": "./data/qdrant_db",
            "MODEL_DIR": "./models",
            "LOGS_DIR": "./logs",
            "TEMP_DIR": "./temp",
            "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
            "CHUNK_SIZE": 500,
            "CHUNK_OVERLAP": 50,
            "TOP_K_RETRIEVALS": 4,
            "SIMILARITY_THRESHOLD": 0.7,
            "MAX_FILE_SIZE_MB": 25,
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
        self.load_config(config_path)
        logger.info(f"Loaded configuration: EMBEDDING_MODEL_NAME={self.get('EMBEDDING_MODEL_NAME')}")

    def load_config(self, config_path):
        """Load configuration from file."""
        try:
            # Check if config file exists
            if not os.path.exists(config_path):
                logger.warning(f"Config file {config_path} not found, using defaults")
                return
                
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            self.config[key] = value
                        except ValueError:
                            logger.warning(f"Skipping invalid config line: {line}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.error(traceback.format_exc())

    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)


class EmbeddingManager:
    """Class to manage embeddings."""
    def __init__(self, config_manager):
        self.config = config_manager
        self.model_name = self.config.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        
        # Set up model path
        self.model_path = os.path.join(
            self.config.get("MODEL_DIR"),
            self.model_name.replace("/", "_")
        )
        
        # Load the embedding model
        self.model = self._load_model()
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
        
    def _load_model(self):
        """Load embedding model from local path or download if not available."""
        try:
            logger.info(f"Attempting to load model from {self.model_path}")
            if os.path.exists(self.model_path):
                model = SentenceTransformer(self.model_path)
                logger.info("✅ Model loaded from local path")
            else:
                logger.info(f"Model not found locally, downloading {self.model_name}")
                # Download model
                model = SentenceTransformer(self.model_name)
                # Save the model for future use
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                model.save(self.model_path)
                logger.info("✅ Model downloaded and saved locally")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            
            # Fall back to default model as a last resort
            try:
                logger.info("Trying to load the default all-MiniLM-L6-v2 model")
                model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("✅ Default model loaded")
                return model
            except Exception as e2:
                logger.error(f"Failed to load default model: {e2}")
                raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            if not texts:
                return []
            
            # Generate embeddings in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings.tolist())
                
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            logger.error(traceback.format_exc())
            return []

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query."""
        try:
            if not query.strip():
                return []
            
            embedding = self.model.encode(query, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            logger.error(traceback.format_exc())
            return []


class PDFProcessor:
    """Class to process PDF documents."""
    def __init__(self, config_manager):
        self.config = config_manager
        self.chunk_size = int(self.config.get("CHUNK_SIZE", 500))
        self.chunk_overlap = int(self.config.get("CHUNK_OVERLAP", 50))
        
    def _get_file_id(self, file_path: str) -> str:
        """Generate a unique ID for a file based on its content."""
        try:
            with open(file_path, 'rb') as f:
                # Only read first 1MB to avoid memory issues with large files
                file_content = f.read(1024 * 1024)
                file_hash = hashlib.md5(file_content).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error generating file ID: {e}")
            # Generate a simple random ID without using UUID
            return f"file_{int(time.time())}_{hash(file_path) % 10000:04d}"

    def extract_text_from_pdf(self, file_path: str, progress_callback=None) -> Tuple[str, List[Dict]]:
        """
        Extract text and metadata from a PDF file.
        Returns a tuple of (text, metadata).
        """
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            full_text = ""
            pages_metadata = []
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text("text")
                
                # Get page metadata
                page_metadata = {
                    "page_num": page_num + 1,
                    "total_pages": total_pages,
                    "width": page.rect.width,
                    "height": page.rect.height,
                }
                
                # Extract images locations and captions
                images = []
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        if base_image:
                            bbox = page.get_image_bbox(img)
                            if bbox:
                                # Look for potential captions - text within ~100 pixels of image
                                extended_bbox = fitz.Rect(
                                    bbox.x0 - 10, bbox.y0 - 10, 
                                    bbox.x1 + 10, bbox.y1 + 100
                                )
                                caption_blocks = [b for b in page.get_text("blocks") 
                                                if fitz.Rect(b[:4]).intersects(extended_bbox)
                                                and b[4].strip() != text.strip()]
                                
                                surrounding_text = " ".join([b[4] for b in caption_blocks])
                                
                                images.append({
                                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                                    "surrounding_text": surrounding_text
                                })
                    except Exception as img_error:
                        logger.warning(f"Error extracting image: {img_error}")
                
                # Extract tables (identified as rectangular areas with text)
                # This is a simple heuristic - production would use more advanced table detection
                tables = []
                try:
                    blocks = page.get_text("blocks")
                    for i, block in enumerate(blocks):
                        # Simple heuristic: looking for blocks with tabular content
                        text_content = block[4]
                        if len(text_content.split('\n')) > 3 and any('  ' in line for line in text_content.split('\n')):
                            tables.append({
                                "bbox": [block[0], block[1], block[2], block[3]],
                                "content": text_content
                            })
                except Exception as table_error:
                    logger.warning(f"Error extracting tables: {table_error}")
                
                # Extract document structure
                headings = []
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    # Simple heuristic for heading detection
                    if line and len(line) < 100 and line.endswith(':') or (
                            len(line.split()) < 8 and line.upper() == line and line):
                        headings.append({
                            "text": line,
                            "line_num": i
                        })
                
                # Store the metadata
                page_metadata.update({
                    "images": images,
                    "tables": tables,
                    "headings": headings
                })
                
                pages_metadata.append(page_metadata)
                full_text += text + "\n\n"
                
                if progress_callback:
                    progress_callback(page_num + 1, total_pages, f"Processed page {page_num + 1}/{total_pages}")
            
            doc.close()
            return full_text, pages_metadata
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            logger.error(traceback.format_exc())
            return "", []

    def smart_chunking(self, text: str, metadata: List[Dict]) -> List[DocumentChunk]:
        """
        Apply smart chunking to the document text.
        Takes into account document structure like sections, tables, etc.
        """
        chunks = []
        
        try:
            # Flatten headings from all pages
            all_headings = []
            for page in metadata:
                for heading in page.get("headings", []):
                    all_headings.append({
                        "text": heading["text"],
                        "page_num": page["page_num"]
                    })
            
            # Split by sections first if headings exist
            if all_headings:
                # Get all text by pages
                pages_text = text.split("\n\n")
                current_section = "Introduction"
                section_text = ""
                page_counter = 1
                
                # Track which heading we're currently processing
                heading_index = 0
                
                for i, page_text in enumerate(pages_text):
                    if heading_index < len(all_headings) and i + 1 >= all_headings[heading_index]["page_num"]:
                        # Process previous section
                        if section_text:
                            section_chunks = self._chunk_text(
                                section_text, 
                                {
                                    "section": current_section,
                                    "page_range": f"{page_counter}-{i+1}"
                                }
                            )
                            chunks.extend(section_chunks)
                        
                        # Start new section
                        current_section = all_headings[heading_index]["text"]
                        section_text = page_text
                        page_counter = i + 1
                        heading_index += 1
                    else:
                        section_text += "\n\n" + page_text
                
                # Process final section
                if section_text:
                    section_chunks = self._chunk_text(
                        section_text, 
                        {
                            "section": current_section,
                            "page_range": f"{page_counter}-{len(pages_text)}"
                        }
                    )
                    chunks.extend(section_chunks)
                    
            else:
                # No headings, use simple chunking
                chunks = self._chunk_text(text, {"section": "Main Content"})
                
            # Special handling for tables (keep them whole if possible)
            for page in metadata:
                for table in page.get("tables", []):
                    table_content = table.get("content", "")
                    if len(table_content) < self.chunk_size * 1.5:  # Only keep small tables whole
                        # Check if this table is already covered by any chunk
                        table_covered = False
                        for chunk in chunks:
                            if table_content in chunk.text:
                                table_covered = True
                                break
                        
                        if not table_covered:
                            chunks.append(DocumentChunk(
                                text=table_content,
                                metadata={
                                    "type": "table",
                                    "page_num": page["page_num"]
                                }
                            ))
            
            # Special handling for image contexts
            for page in metadata:
                for img in page.get("images", []):
                    surrounding_text = img.get("surrounding_text", "")
                    if surrounding_text and len(surrounding_text) < self.chunk_size:
                        # Check if this image context is already in any chunk
                        context_covered = False
                        for chunk in chunks:
                            if surrounding_text in chunk.text:
                                context_covered = True
                                break
                        
                        if not context_covered:
                            chunks.append(DocumentChunk(
                                text=surrounding_text,
                                metadata={
                                    "type": "image_context",
                                    "page_num": page["page_num"]
                                }
                            ))
                            
            return chunks
        except Exception as e:
            logger.error(f"Error in smart chunking: {e}")
            logger.error(traceback.format_exc())
            # Fallback to basic chunking
            return self._chunk_text(text, {"section": "Main Content"})
    
    def _chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Chunk text with overlap.
        """
        if not text.strip():
            return []
            
        chunks = []
        
        try:
            # Clean the text
            text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
            
            # Try to split by paragraphs first
            paragraphs = text.split("\n\n")
            
            current_chunk = ""
            current_chunk_metadata = metadata.copy()
            current_chunk_metadata.update({"chunk_index": len(chunks)})
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # If adding this paragraph exceeds chunk size, store current chunk and start a new one
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        metadata=current_chunk_metadata
                    ))
                    
                    # Start new chunk with overlap
                    overlap_text = " ".join(current_chunk.split(" ")[-self.chunk_overlap:]) if self.chunk_overlap > 0 else ""
                    current_chunk = overlap_text + " " + para if overlap_text else para
                    current_chunk_metadata = metadata.copy()
                    current_chunk_metadata.update({"chunk_index": len(chunks)})
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += " " + para
                    else:
                        current_chunk = para
            
            # Add the last chunk if it's not empty
            if current_chunk.strip():
                chunks.append(DocumentChunk(
                    text=current_chunk.strip(),
                    metadata=current_chunk_metadata
                ))
        except Exception as e:
            logger.error(f"Error in _chunk_text: {e}")
            logger.error(traceback.format_exc())
            
        return chunks
    
    def generate_summary(self, text: str, metadata: List[Dict]) -> str:
        """
        Generate a simple summary of the document.
        In production, this would call an LLM for better summarization.
        """
        try:
            # Extract document title if available
            title = "Unknown Title"
            for page in metadata:
                if page["page_num"] == 1 and page.get("headings"):
                    title = page["headings"][0]["text"]
                    break
                    
            # Get total pages
            total_pages = metadata[-1]["page_num"] if metadata else 0
            
            # Count images and tables
            image_count = sum(len(page.get("images", [])) for page in metadata)
            table_count = sum(len(page.get("tables", [])) for page in metadata)
            
            # Get top sections
            sections = []
            for page in metadata:
                for heading in page.get("headings", []):
                    sections.append(heading["text"])
            
            # Create summary
            summary = f"Title: {title}\n"
            summary += f"Pages: {total_pages}\n"
            summary += f"Contains: ~{image_count} images, ~{table_count} tables\n"
            
            if sections:
                summary += "Main Sections:\n"
                for i, section in enumerate(sections[:5]):  # Only show top 5 sections
                    summary += f"- {section}\n"
                if len(sections) > 5:
                    summary += f"- ...and {len(sections) - 5} more sections\n"
                    
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def process_pdf(self, file_path: str, progress_callback=None) -> ProcessedDocument:
        """
        Process a PDF file and return chunks and metadata.
        """
        try:
            filename = os.path.basename(file_path)
            file_id = self._get_file_id(file_path)
            
            logger.info(f"Processing PDF file: {filename}")
            
            # Extract text and metadata
            text, metadata = self.extract_text_from_pdf(file_path, progress_callback)
            
            if not text:
                logger.warning(f"No text extracted from file: {filename}")
                return ProcessedDocument(
                    filename=filename,
                    file_id=file_id,
                    total_pages=0,
                    chunks=[],
                    summary="Error: Could not extract text from this PDF.",
                    status="error"
                )
                
            # Create chunks
            chunks = self.smart_chunking(text, metadata)
            
            # Generate summary
            summary = self.generate_summary(text, metadata)
            
            return ProcessedDocument(
                filename=filename,
                file_id=file_id,
                total_pages=len(metadata),
                chunks=chunks,
                summary=summary
            )
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            logger.error(traceback.format_exc())
            return ProcessedDocument(
                filename=os.path.basename(file_path),
                file_id="error",
                total_pages=0,
                chunks=[],
                summary=f"Error processing PDF: {str(e)}",
                status="error"
            )


class QdrantManager:
    """Class to manage Qdrant vector database."""
    def __init__(self, config_manager, embedding_dimension=None):
        self.config = config_manager
        self.qdrant_path = self.config.get("QDRANT_DIR")
        self.client = self._setup_client()
        self.collection_name = "pdf_chunks"  # Ensure a simple collection name
        # Use the actual embedding dimension if provided, otherwise use config
        self.vector_size = embedding_dimension or int(self.config.get("EMBEDDING_MODEL_DIMENSIONS", 384))
        logger.info(f"Using vector size for Qdrant: {self.vector_size}")
        self._setup_collection()
        
    def _setup_client(self):
        """Setup Qdrant client with local persistence."""
        try:
            client = QdrantClient(path=self.qdrant_path)
            logger.info("✅ Connected to local Qdrant database")
            return client
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            logger.error(traceback.format_exc())
            return None
            
    def _setup_collection(self):
        """Create collection if it doesn't exist."""
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized")
                return
                
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            # Check if collection exists
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_size}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE
                    ),
                )
                logger.info(f"✅ Collection '{self.collection_name}' created")
            else:
                # Check collection info
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size
                
                if existing_size != self.vector_size:
                    logger.warning(f"Collection exists with vector size {existing_size}, but current model has size {self.vector_size}")
                    # Recreate collection with correct size
                    logger.info(f"Recreating collection with correct vector size {self.vector_size}")
                    self.client.delete_collection(self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_size, 
                            distance=Distance.COSINE
                        ),
                    )
                    logger.info(f"✅ Collection '{self.collection_name}' recreated with correct vector size")
                else:
                    logger.info(f"Collection '{self.collection_name}' already exists with correct vector size {existing_size}")
                
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            logger.error(traceback.format_exc())
            
    def add_chunks(self, processed_doc: ProcessedDocument, embeddings: List[List[float]], 
                   progress_callback=None) -> bool:
        """
        Add document chunks to Qdrant.
        """
        try:
            if not embeddings or len(embeddings) != len(processed_doc.chunks):
                logger.error(f"Mismatch between chunks and embeddings: {len(processed_doc.chunks)} chunks, {len(embeddings)} embeddings")
                return False
            
            # Verify embedding dimensions match collection configuration
            if embeddings and len(embeddings[0]) != self.vector_size:
                logger.error(f"Embedding dimension mismatch: Expected {self.vector_size}, got {len(embeddings[0])}")
                # Try to check collection info
                try:
                    collection_info = self.client.get_collection(self.collection_name)
                    logger.error(f"Collection vector size: {collection_info.config.params.vectors.size}")
                except Exception as e:
                    logger.error(f"Error getting collection info: {e}")
                
                return False
                
            # Create points
            points = []
            base_id = int(time.time() * 1000)  # Use timestamp as base ID
            
            for i, (chunk, embedding) in enumerate(zip(processed_doc.chunks, embeddings)):
                # Prepare metadata - only include serializable data
                metadata = {
                    "file_id": processed_doc.file_id,
                    "filename": processed_doc.filename,
                    "chunk_id": f"{processed_doc.file_id}_{i}",
                    "section": chunk.metadata.get("section", "Unknown"),
                    "page_num": chunk.metadata.get("page_num", 0),
                    "chunk_index": i,
                    "created_at": datetime.now().isoformat()
                }
                
                # Create point with numeric ID (base_id + index)
                point_id = base_id + i  # This ensures unique numeric IDs
                
                # Debug log for first point
                if i == 0:
                    logger.info(f"First point - ID: {point_id}, Vector dimension: {len(embedding)}")
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "metadata": metadata
                    }
                )
                
                points.append(point)
                
                if progress_callback:
                    progress_callback(i + 1, len(processed_doc.chunks), f"Preparing chunk {i+1}/{len(processed_doc.chunks)}")
                    
            # Upload in batches of 100
            if not points:
                logger.warning("No points to add to database")
                return False
                
            logger.info(f"Adding {len(points)} points to collection {self.collection_name}")
            batch_size = 10  # Smaller batch size to reduce errors
            success_count = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    success_count += len(batch)
                    
                    if progress_callback:
                        progress_callback(min(i + batch_size, len(points)), len(points), 
                                         f"Stored {success_count}/{len(points)} chunks")
                except Exception as batch_error:
                    logger.error(f"Error upserting batch {i//batch_size}: {batch_error}")
                
            logger.info(f"✅ Added {success_count}/{len(points)} chunks to Qdrant for file: {processed_doc.filename}")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error adding chunks to Qdrant: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def search(self, query_vector: List[float], limit: int = None, 
               filter_file_ids: List[str] = None) -> List[Dict]:
        """
        Search for similar chunks in Qdrant.
        """
        try:
            if not query_vector:
                return []
                
            # Set limit
            if not limit:
                limit = int(self.config.get("TOP_K_RETRIEVALS", 4))
                
            # Set filter if needed
            filter_param = None
            if filter_file_ids:
                filter_param = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="payload.metadata.file_id",
                            match=models.MatchAny(any=filter_file_ids)
                        )
                    ]
                )
                
            # Double-check collection exists
            try:
                self.client.get_collection(self.collection_name)
            except Exception as coll_error:
                logger.error(f"Collection error during search: {coll_error}")
                return []
                
            # Execute search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_param,
                with_payload=True,
                score_threshold=float(self.config.get("SIMILARITY_THRESHOLD", 0.7))
            )
            
            # Format results
            formatted_results = []
            for res in results:
                if not hasattr(res, 'payload') or not res.payload:
                    continue
                    
                # Handle missing keys gracefully
                payload = res.payload
                text = payload.get("text", "No text available")
                metadata = payload.get("metadata", {})
                
                formatted_results.append({
                    "text": text,
                    "metadata": metadata,
                    "score": res.score
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            logger.error(traceback.format_exc())
            return []
            
    def get_document_ids(self) -> List[Dict]:
        """Get all document IDs in the collection."""
        try:
            # Check if collection exists
            try:
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                if self.collection_name not in collection_names:
                    logger.warning(f"Collection {self.collection_name} does not exist")
                    return []
            except Exception as e:
                logger.error(f"import os
import re
import time
import logging
import tempfile
import traceback
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import uuid
import hashlib
import json
from datetime import datetime

# Vector DB
from qdrant_client import QdrantClient
from qdrant_client.http import models

# LangChain Imports
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Initialize logging
for directory in ["./data", "./data/embeddings", "./data/qdrant_db", "./models", "./logs", "./temp"]:
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Class to store information about a processed document."""
    filename: str
    file_id: str
    total_pages: int
    summary: str = ""
    status: str = "success"


class ConfigManager:
    """Class to manage configuration."""
    def __init__(self, config_path="config.txt"):
        self.config = {
            "DATA_DIR": "./data",
            "EMBEDDINGS_DIR": "./data/embeddings",
            "QDRANT_DIR": "./data/qdrant_db",
            "MODEL_DIR": "./models",
            "LOGS_DIR": "./logs",
            "TEMP_DIR": "./temp",
            "EMBEDDING_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
            "CHUNK_SIZE": 500,
            "CHUNK_OVERLAP": 50,
            "TOP_K_RETRIEVALS": 4,
            "SIMILARITY_THRESHOLD": 0.7,
            "MAX_FILE_SIZE_MB": 25,
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
        self.load_config(config_path)
        logger.info(f"Loaded configuration: EMBEDDING_MODEL_NAME={self.get('EMBEDDING_MODEL_NAME')}")

    def load_config(self, config_path):
        """Load configuration from file."""
        try:
            # Check if config file exists
            if not os.path.exists(config_path):
                logger.warning(f"Config file {config_path} not found, using defaults")
                return
                
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            self.config[key] = value
                        except ValueError:
                            logger.warning(f"Skipping invalid config line: {line}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.error(traceback.format_exc())

    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)


class EmbeddingManager:
    """Class to manage embeddings using LangChain's HuggingFaceEmbeddings."""
    def __init__(self, config_manager):
        self.config = config_manager
        self.model_name = self.config.get("EMBEDDING_MODEL_NAME")
        
        # Set up model path
        self.model_path = os.path.join(
            self.config.get("MODEL_DIR"),
            self.model_name.split('/')[-1]
        )
        
        # Create the embeddings model
        self.embeddings = self._create_embeddings()
        
    def _create_embeddings(self):
        """Create HuggingFaceEmbeddings instance."""
        try:
            logger.info(f"Loading embedding model {self.model_name}")
            
            # Try to load from default location
            try:
                # Configure the embeddings with explicit model parameters
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    cache_folder=self.config.get("MODEL_DIR"),
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}  # Ensure cosine similarity works well
                )
                logger.info(f"✅ Embeddings model loaded successfully: {self.model_name}")
                return embeddings
            except Exception as model_error:
                logger.warning(f"Error loading model from default location: {model_error}")
                
                # Fall back to all-MiniLM-L6-v2 which is reliable
                logger.info("Falling back to all-MiniLM-L6-v2 model")
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                embeddings = HuggingFaceEmbeddings(
                    model_name=fallback_model,
                    cache_folder=self.config.get("MODEL_DIR"),
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"✅ Fallback embeddings model loaded successfully: {fallback_model}")
                return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            logger.error(traceback.format_exc())
            raise


class DocumentProcessor:
    """Class to process PDF documents using LangChain's document loaders and text splitters."""
    def __init__(self, config_manager):
        self.config = config_manager
        self.chunk_size = int(self.config.get("CHUNK_SIZE", 500))
        self.chunk_overlap = int(self.config.get("CHUNK_OVERLAP", 50))
        
    def _get_file_id(self, file_path: str) -> str:
        """Generate a unique ID for a file based on its content."""
        try:
            with open(file_path, 'rb') as f:
                # Only read first 1MB to avoid memory issues with large files
                file_content = f.read(1024 * 1024)
                file_hash = hashlib.md5(file_content).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error generating file ID: {e}")
            # Generate a simple random ID without using UUID
            return f"file_{int(time.time())}_{hash(file_path) % 10000:04d}"

    def process_pdf(self, file_path: str, progress_callback=None) -> Tuple[ProcessedDocument, List[Document]]:
        """
        Process a PDF file and return chunks and metadata.
        """
        filename = os.path.basename(file_path)
        file_id = self._get_file_id(file_path)
        
        logger.info(f"Processing PDF file: {filename}")
        
        try:
            # Use PyMuPDFLoader to load the document
            loader = PyMuPDFLoader(file_path)
            
            # Load all pages
            documents = loader.load()
            total_pages = len(documents)
            
            if progress_callback:
                progress_callback(1, 3, "PDF loaded successfully")
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )
            
            # Split the documents
            split_documents = text_splitter.split_documents(documents)
            
            if progress_callback:
                progress_callback(2, 3, f"Split into {len(split_documents)} chunks")
            
            # Generate basic summary
            summary = self._generate_summary(documents, total_pages, filename)
            
            if progress_callback:
                progress_callback(3, 3, "Summary generated")
            
            # Create processed document object
            processed_doc = ProcessedDocument(
                filename=filename,
                file_id=file_id,
                total_pages=total_pages,
                summary=summary
            )
            
            return processed_doc, split_documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            logger.error(traceback.format_exc())
            
            # Return empty document with error status
            processed_doc = ProcessedDocument(
                filename=filename,
                file_id=file_id,
                total_pages=0,
                summary=f"Error processing document: {str(e)}",
                status="error"
            )
            
            return processed_doc, []
    
    def _generate_summary(self, documents: List[Document], total_pages: int, filename: str) -> str:
        """Generate a simple summary of the document."""
        try:
            # Extract title from first page if possible
            title = filename
            if documents and hasattr(documents[0], 'page_content'):
                first_page = documents[0].page_content
                # Try to extract title from first few lines
                lines = first_page.split('\n')
                for line in lines[:5]:
                    if line.strip() and len(line.strip()) < 100:
                        title = line.strip()
                        break
            
            # Count images and tables (approximation)
            image_pattern = r'image|figure|pic|img'
            table_pattern = r'table|tbl'
            
            image_count = sum(1 for doc in documents if re.search(image_pattern, doc.page_content.lower()))
            table_count = sum(1 for doc in documents if re.search(table_pattern, doc.page_content.lower()))
            
            # Create summary
            summary = f"Title: {title}\n"
            summary += f"Pages: {total_pages}\n"
            summary += f"Contains: ~{image_count} images, ~{table_count} tables\n"
            
            # Try to extract main headings
            headings = []
            heading_pattern = r'^[A-Z][^.!?]*$'
            
            for doc in documents:
                content = doc.page_content
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if re.match(heading_pattern, line) and 3 < len(line) < 80:
                        headings.append(line)
            
            # Add unique headings
            unique_headings = list(dict.fromkeys(headings))[:5]  # Get first 5 unique headings
            
            if unique_headings:
                summary += "\nMain Sections:\n"
                for heading in unique_headings:
                    summary += f"- {heading}\n"
                    
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Summary unavailable due to error: {str(e)}"


class VectorStoreManager:
    """Class to manage vector store using LangChain's Qdrant wrapper."""
    def __init__(self, config_manager, embeddings):
        self.config = config_manager
        self.embeddings = embeddings
        self.qdrant_path = self.config.get("QDRANT_DIR")
        self.collection_name = "pdf_documents"  # This is the collection name we'll use consistently
        
        # Initialize Qdrant client
        self.client = self._setup_client()
        
        # Initialize vector store
        self.vector_store = self._setup_vector_store()
        
    def _setup_client(self):
        """Set up Qdrant client."""
        try:
            client = QdrantClient(path=self.qdrant_path)
            logger.info("✅ Connected to local Qdrant database")
            return client
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _setup_vector_store(self):
        """Set up Qdrant vector store."""
        try:
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            # Ensure collection exists before trying to use it
            if self.collection_name not in collection_names:
                # Create the collection with appropriate vector size
                vector_size = self.embeddings.get_query_embedding_dimension()
                logger.info(f"Creating collection '{self.collection_name}' with vector size {vector_size}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created")
            
            # Create vector store
            vector_store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
                
            return vector_store
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def add_documents(self, documents: List[Document], file_id: str, filename: str, 
                      progress_callback=None) -> bool:
        """Add documents to vector store."""
        try:
            if not documents:
                logger.warning(f"No documents to add for file: {filename}")
                return False
            
            # Add file metadata to each document
            for i, doc in enumerate(documents):
                if hasattr(doc, 'metadata'):
                    doc.metadata['file_id'] = file_id
                    doc.metadata['filename'] = filename
                    doc.metadata['chunk_id'] = f"{file_id}_{i}"
                else:
                    # If the document doesn't have metadata, create it
                    doc.metadata = {
                        'file_id': file_id,
                        'filename': filename,
                        'chunk_id': f"{file_id}_{i}"
                    }
            
            # Track progress
            total_docs = len(documents)
            if progress_callback:
                progress_callback(0, total_docs, "Starting document embedding")
            
            # Make sure the collection exists
            try:
                # Check if collection exists
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                # Create collection if it doesn't exist
                if self.collection_name not in collection_names:
                    vector_size = self.embeddings.get_query_embedding_dimension()
                    logger.info(f"Creating collection '{self.collection_name}' with vector size {vector_size}")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    logger.info(f"Collection '{self.collection_name}' created")
                    
                    # Recreate the vector store with the new collection
                    self.vector_store = Qdrant(
                        client=self.client,
                        collection_name=self.collection_name,
                        embeddings=self.embeddings
                    )
            except Exception as coll_error:
                logger.error(f"Error checking/creating collection: {coll_error}")
                logger.error(traceback.format_exc())
                return False
            
            # Add documents in batches
            batch_size = 25  # Smaller batch size for better progress tracking
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i+batch_size]
                try:
                    self.vector_store.add_documents(batch)
                    
                    if progress_callback:
                        progress_callback(min(i + batch_size, total_docs), total_docs, 
                                         f"Embedded {min(i + batch_size, total_docs)} of {total_docs} chunks")
                except Exception as batch_error:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {batch_error}")
                    logger.error(traceback.format_exc())
                    # Continue with next batch instead of failing completely
                    continue
                    
            logger.info(f"✅ Added {total_docs} documents to vector store for file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def get_retriever(self, file_ids: List[str] = None, top_k: int = None):
        """Get retriever configured with optional filters."""
        try:
            # Set default top_k if not provided
            if top_k is None:
                top_k = int(self.config.get("TOP_K_RETRIEVALS", 4))
            
            # Create search filter if file_ids are provided
            search_filter = None
            if file_ids:
                search_filter = {
                    "must": [
                        {
                            "key": "file_id",
                            "match": {
                                "any": file_ids
                            }
                        }
                    ]
                }
            
            # Create retriever with search_kwargs
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": top_k,
                    "filter": search_filter
                }
            )
            
            # Customize retrieval settings
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            logger.error(traceback.format_exc())
            raise
        
    def get_document_ids(self) -> List[Dict]:
        """Get all document IDs and filenames in the vector store."""
        try:
            # Get all points from collection
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=["file_id", "filename"],
                with_vectors=False
            )
            
            # Extract unique document IDs
            unique_docs = {}
            points = response[0]
            
            while points:
                for point in points:
                    if "payload" in point and "file_id" in point.payload:
                        file_id = point.payload["file_id"]
                        filename = point.payload.get("filename", "Unknown")
                        unique_docs[file_id] = filename
                
                # Get next batch
                last_id = points[-1].id if points else None
                if not last_id:
                    break
                    
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    with_payload=["file_id", "filename"],
                    with_vectors=False,
                    offset=last_id
                )
                points = response[0]
            
            return [{"id": k, "name": v} for k, v in unique_docs.items()]
            
        except Exception as e:
            logger.error(f"Error getting document IDs: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def delete_document(self, file_id: str) -> bool:
        """Delete documents with specified file_id."""
        try:
            # Create filter to match documents with this file_id
            filter_param = {
                "must": [
                    {
                        "key": "file_id",
                        "match": {
                            "value": file_id
                        }
                    }
                ]
            }
            
            # Delete matching documents
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=filter_param
                )
            )
            
            logger.info(f"✅ Deleted documents with file_id {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            logger.error(traceback.format_exc())
            return False


class RAGManager:
    """Main class to manage the RAG application."""
    def __init__(self):
        self.config_manager = ConfigManager()
        self.embedding_manager = EmbeddingManager(self.config_manager)
        self.document_processor = DocumentProcessor(self.config_manager)
        
        # Try up to 3 times to initialize the vector store manager
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.vector_store_manager = VectorStoreManager(
                    self.config_manager, 
                    self.embedding_manager.embeddings
                )
                logger.info("Vector store manager initialized successfully")
                break
            except Exception as e:
                logger.error(f"Attempt {attempt+1}/{max_retries} to initialize vector store manager failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All attempts to initialize vector store manager failed")
                    # We'll continue and try to initialize on demand
                else:
                    time.sleep(1)  # Wait a bit before retrying
    
    def process_pdf_file(self, file_path: str, progress_callbacks=None) -> ProcessedDocument:
        """Process a PDF file and store it in the vector store."""
        try:
            # Define progress callback functions
            extraction_callback = progress_callbacks.get('extraction') if progress_callbacks else None
            embedding_callback = progress_callbacks.get('embedding') if progress_callbacks else None
            
            # 1. Process PDF to get documents
            processed_doc, documents = self.document_processor.process_pdf(
                file_path, extraction_callback
            )
            
            if processed_doc.status == "error" or not documents:
                logger.warning(f"Error processing file or no documents extracted: {file_path}")
                return processed_doc
            
            # Make sure vector store manager is initialized
            if not hasattr(self, 'vector_store_manager') or self.vector_store_manager is None:
                try:
                    logger.info("Initializing vector store manager on demand")
                    self.vector_store_manager = VectorStoreManager(
                        self.config_manager, 
                        self.embedding_manager.embeddings
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize vector store manager: {e}")
                    processed_doc.status = "error"
                    processed_doc.summary += "\nError: Failed to initialize vector database."
                    return processed_doc
            
            # 2. Add documents to vector store (with retries)
            max_attempts = 3
            stored = False
            
            for attempt in range(max_attempts):
                try:
                    if embedding_callback:
                        embedding_callback(0, len(documents), f"Attempt {attempt+1}/{max_attempts}")
                    
                    stored = self.vector_store_manager.add_documents(
                        documents,
                        processed_doc.file_id,
                        processed_doc.filename,
                        embedding_callback
                    )
                    
                    if stored:
                        logger.info(f"Successfully stored documents on attempt {attempt+1}/{max_attempts}")
                        break
                except Exception as e:
                    logger.error(f"Attempt {attempt+1}/{max_attempts} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(1)  # Wait before retry
            
            if not stored:
                processed_doc.status = "error"
                processed_doc.summary += "\nError: Failed to store document in vector database after multiple attempts."
                logger.warning(f"Failed to store documents for file: {file_path} after {max_attempts} attempts")
                return processed_doc
            
            logger.info(f"✅ Successfully processed file: {file_path}")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            logger.error(traceback.format_exc())
            
            # Return error document
            return ProcessedDocument(
                filename=os.path.basename(file_path),
                file_id="error",
                total_pages=0,
                summary=f"Error processing document: {str(e)}",
                status="error"
            )
    
    def query(self, query_text: str, file_ids: List[str] = None) -> Dict:
        """Query the RAG system."""
        try:
            # Make sure vector store manager is initialized
            if not hasattr(self, 'vector_store_manager') or self.vector_store_manager is None:
                try:
                    self.vector_store_manager = VectorStoreManager(
                        self.config_manager, 
                        self.embedding_manager.embeddings
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize vector store manager on demand: {e}")
                    return {
                        "answer": "There was an error accessing the document database. Please try uploading your documents again.",
                        "sources": []
                    }
            
            # 1. Get retriever with file filters if specified
            try:
                retriever = self.vector_store_manager.get_retriever(file_ids)
            except Exception as retriever_error:
                logger.error(f"Error creating retriever: {retriever_error}")
                return {
                    "answer": "There was an error setting up the document retrieval system. Please try again.",
                    "sources": []
                }
            
            # 2. Retrieve relevant documents
            try:
                docs = retriever.get_relevant_documents(query_text)
            except Exception as retrieval_error:
                logger.error(f"Error retrieving documents: {retrieval_error}")
                return {
                    "answer": "There was an error searching the documents. Please try a different question or reload the documents.",
                    "sources": []
                }
            
            if not docs:
                return {
                    "answer": "No relevant information found in the documents. Please try rephrasing your question or adding more documents.",
                    "sources": []
                }
            
            # 3. Format context for response generation
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 4. Generate sources information
            sources = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                text = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                sources.append({
                    "text": text,
                    "file": metadata.get("filename", "Unknown"),
                    "page": metadata.get("page", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                    "score": metadata.get("score", "N/A")
                })
            
            # 5. Generate response using prompt template
            prompt = f"""
            Based on the following context, please answer the query.
            
            Query: {query_text}
            
            Context:
            {context}
            """
            
            # Call the mocked LLM response function
            answer = abc_response(prompt)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": []
            }
    
    def get_available_documents(self) -> List[Dict]:
        """Get all available documents."""
        return self.vector_store_manager.get_document_ids()
    
    def delete_document(self, file_id: str) -> bool:
        """Delete a document."""
        return self.vector_store_manager.delete_document(file_id)


# Mock function for LLM response
def abc_response(prompt: str) -> str:
    """
    This is a mock function to simulate an LLM response.
    In production, replace with actual LLM API call.
    """
    # Extract query from prompt
    query_match = re.search(r"Query: (.*?)\n", prompt)
    query = query_match.group(1) if query_match else "Unknown query"
    
    # Extract context
    context_match = re.search(r"Context:\s*(.*)", prompt, re.DOTALL)
    context = context_match.group(1) if context_match else ""
    
    # Generate a simple response based on available context
    if not context:
        return "I don't have enough information to answer this question."
    
    # Very simple response generation - just extract relevant sentences
    sentences = re.split(r'[.!?]\s+', context)
    query_words = set(query.lower().split())
    
    # Filter for relevant sentences
    relevant_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if sentence is relevant to query
        sentence_words = set(sentence.lower().split())
        overlap = query_words.intersection(sentence_words)
        
        if len(overlap) >= 1 or any(word in sentence.lower() for word in query.lower().split()):
            relevant_sentences.append(sentence)
    
    if relevant_sentences:
        # Generate a simple response combining relevant sentences
        response = "Based on the documents, " + ". ".join(relevant_sentences[:3]) + "."
        return response
    else:
        return "I found some information in the documents, but it doesn't directly answer your query. Can you please rephrase your question?"
