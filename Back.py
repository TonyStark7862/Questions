import os
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

# Vector DB and Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# PDF Processing
import fitz  # PyMuPDF
import pandas as pd
from tqdm.auto import tqdm

# Config
import configparser

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

# Ensure directories exist
for directory in ["./data", "./data/embeddings", "./data/qdrant_db", "./models", "./logs", "./temp"]:
    os.makedirs(directory, exist_ok=True)


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
    embedding_status: str = "not_started"
    

class ConfigManager:
    """Class to manage configuration."""
    def __init__(self, config_path="config.txt"):
        self.config = {}
        self.load_config(config_path)

    def load_config(self, config_path):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        self.config[key.strip()] = value.strip().strip('"\'')
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Set default values
            self.config = {
                "DATA_DIR": "./data",
                "EMBEDDINGS_DIR": "./data/embeddings",
                "QDRANT_DIR": "./data/qdrant_db",
                "MODEL_DIR": "./models",
                "LOGS_DIR": "./logs",
                "TEMP_DIR": "./temp",
                "EMBEDDING_MODEL_NAME": "BAAI/bge-small-en-v1.5",
                "EMBEDDING_MODEL_DIMENSIONS": 384,
                "CHUNK_SIZE": 500,
                "CHUNK_OVERLAP": 50,
                "TOP_K_RETRIEVALS": 4,
                "SIMILARITY_THRESHOLD": 0.7,
                "MAX_FILE_SIZE_MB": 25,
                "LOG_LEVEL": "INFO",
                "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }

    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)


class EmbeddingManager:
    """Class to manage embeddings."""
    def __init__(self, config_manager):
        self.config = config_manager
        self.model_path = os.path.join(
            self.config.get("MODEL_DIR"),
            self.config.get("EMBEDDING_MODEL_NAME").replace("/", "_")
        )
        self.model = self._load_model()

    def _load_model(self):
        """Load embedding model from local path or download if not available."""
        try:
            logger.info(f"Loading embedding model from {self.model_path}...")
            # Try loading from local path
            model = SentenceTransformer(self.model_path)
            logger.info("✅ Model loaded from local path")
            return model
        except Exception as e:
            logger.warning(f"Model not found locally or error loading: {e}. Downloading model...")
            # Download model
            model = SentenceTransformer(self.config.get("EMBEDDING_MODEL_NAME"))
            # Save the model for future use
            os.makedirs(self.model_path, exist_ok=True)
            model.save(self.model_path)
            logger.info("✅ Model downloaded and saved locally")
            return model

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
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error generating file ID: {e}")
            return str(uuid.uuid4())

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
                
                # Extract tables (identified as rectangular areas with text)
                # This is a simple heuristic - production would use more advanced table detection
                tables = []
                blocks = page.get_text("blocks")
                for i, block in enumerate(blocks):
                    # Simple heuristic: looking for blocks with tabular content
                    text_content = block[4]
                    if len(text_content.split('\n')) > 3 and any('  ' in line for line in text_content.split('\n')):
                        tables.append({
                            "bbox": [block[0], block[1], block[2], block[3]],
                            "content": text_content
                        })
                
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
                    progress_callback(page_num + 1, total_pages)
            
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
    
    def _chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Chunk text with overlap.
        """
        if not text.strip():
            return []
            
        chunks = []
        
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
                overlap_text = " ".join(current_chunk.split(" ")[-self.chunk_overlap:])
                current_chunk = overlap_text + " " + para
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
            
        return chunks
    
    def generate_summary(self, text: str, metadata: List[Dict]) -> str:
        """
        Generate a simple summary of the document.
        In production, this would call an LLM for better summarization.
        """
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
        summary += f"Contains: {image_count} images, {table_count} tables\n"
        
        if sections:
            summary += "Main Sections:\n"
            for i, section in enumerate(sections[:5]):  # Only show top 5 sections
                summary += f"- {section}\n"
            if len(sections) > 5:
                summary += f"- ...and {len(sections) - 5} more sections\n"
                
        return summary
    
    def process_pdf(self, file_path: str, progress_callback=None) -> ProcessedDocument:
        """
        Process a PDF file and return chunks and metadata.
        """
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
                summary="Error: Could not extract text from this PDF."
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


class QdrantManager:
    """Class to manage Qdrant vector database."""
    def __init__(self, config_manager):
        self.config = config_manager
        self.qdrant_path = self.config.get("QDRANT_DIR")
        self.client = self._setup_client()
        self.collection_name = "pdf_chunks"
        self.vector_size = int(self.config.get("EMBEDDING_MODEL_DIMENSIONS"))
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
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE
                    ),
                )
                logger.info(f"✅ Collection '{self.collection_name}' created")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
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
                
            # Create points
            points = []
            for i, (chunk, embedding) in enumerate(zip(processed_doc.chunks, embeddings)):
                # Prepare metadata
                metadata = chunk.metadata.copy()
                metadata.update({
                    "file_id": processed_doc.file_id,
                    "filename": processed_doc.filename,
                    "chunk_id": f"{processed_doc.file_id}_{i}",
                    "created_at": datetime.now().isoformat()
                })
                
                # Create point
                point = PointStruct(
                    id=f"{processed_doc.file_id}_{i}",
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "metadata": metadata
                    }
                )
                
                points.append(point)
                
                if progress_callback:
                    progress_callback(i + 1, len(processed_doc.chunks))
                    
            # Upload in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
            logger.info(f"✅ Added {len(points)} chunks to Qdrant for file: {processed_doc.filename}")
            return True
            
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
                            key="metadata.file_id",
                            match=models.MatchAny(any=filter_file_ids)
                        )
                    ]
                )
                
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
                formatted_results.append({
                    "text": res.payload["text"],
                    "metadata": res.payload["metadata"],
                    "score": res.score
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            logger.error(traceback.format_exc())
            return []
            
    def get_document_ids(self) -> List[str]:
        """Get all document IDs in the collection."""
        try:
            # Get unique file_ids
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=100,
                with_payload=["metadata.file_id", "metadata.filename"],
                with_vectors=False
            )
            
            unique_docs = {}
            points = response[0]
            
            while points:
                for point in points:
                    if "metadata" in point.payload and "file_id" in point.payload["metadata"]:
                        file_id = point.payload["metadata"]["file_id"]
                        filename = point.payload["metadata"].get("filename", "Unknown")
                        if file_id not in unique_docs:
                            unique_docs[file_id] = filename
                
                # Get next batch
                last_id = points[-1].id
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=100,
                    with_payload=["metadata.file_id", "metadata.filename"],
                    with_vectors=False,
                    offset=last_id
                )
                points = response[0]
                if not points or points[0].id == last_id:
                    break
            
            return [{"id": k, "name": v} for k, v in unique_docs.items()]
            
        except Exception as e:
            logger.error(f"Error getting document IDs: {e}")
            logger.error(traceback.format_exc())
            return []
            
    def delete_document(self, file_id: str) -> bool:
        """Delete a document from the collection."""
        try:
            # Create a filter to match points with this file_id
            filter_param = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchValue(value=file_id)
                    )
                ]
            )
            
            # Execute deletion
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_param
            )
            
            logger.info(f"✅ Deleted document with ID {file_id} from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            logger.error(traceback.format_exc())
            return False


class RAGManager:
    """Main class to manage the RAG application."""
    def __init__(self):
        self.config_manager = ConfigManager()
        self.embedding_manager = EmbeddingManager(self.config_manager)
        self.pdf_processor = PDFProcessor(self.config_manager)
        self.qdrant_manager = QdrantManager(self.config_manager)
        
    def process_pdf_file(self, file_path: str, progress_callbacks=None) -> bool:
        """
        Process a PDF file and store it in Qdrant.
        progress_callbacks is a dict with keys: 'extraction', 'embedding', 'storage'
        """
        try:
            # 1. Process PDF
            extraction_callback = progress_callbacks.get('extraction') if progress_callbacks else None
            processed_doc = self.pdf_processor.process_pdf(file_path, extraction_callback)
            
            if not processed_doc.chunks:
                logger.warning(f"No chunks extracted from file: {file_path}")
                return False
                
            # 2. Generate embeddings
            embedding_callback = progress_callbacks.get('embedding') if progress_callbacks else None
            chunk_texts = [chunk.text for chunk in processed_doc.chunks]
            
            if embedding_callback:
                embedding_callback(0, len(chunk_texts))
                
            embeddings = self.embedding_manager.embed_texts(chunk_texts)
            
            if embedding_callback:
                embedding_callback(len(chunk_texts), len(chunk_texts))
                
            if not embeddings:
                logger.warning(f"Failed to generate embeddings for file: {file_path}")
                return False
                
            # 3. Store in Qdrant
            storage_callback = progress_callbacks.get('storage') if progress_callbacks else None
            stored = self.qdrant_manager.add_chunks(processed_doc, embeddings, storage_callback)
            
            if not stored:
                logger.warning(f"Failed to store chunks for file: {file_path}")
                return False
                
            logger.info(f"✅ Successfully processed file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def query(self, query_text: str, file_ids: List[str] = None) -> Dict:
        """
        Query the RAG system.
        """
        try:
            # 1. Generate query embedding
            query_embedding = self.embedding_manager.embed_query(query_text)
            
            if not query_embedding:
                return {
                    "answer": "Error: Failed to generate embedding for query.",
                    "sources": []
                }
                
            # 2. Search Qdrant
            results = self.qdrant_manager.search(query_embedding, filter_file_ids=file_ids)
            
            if not results:
                return {
                    "answer": "No relevant information found. Please try a different query or upload more documents.",
                    "sources": []
                }
                
            # 3. Format context for LLM
            context = "\n\n".join([f"Source {i+1}:\n{result['text']}" for i, result in enumerate(results)])
            
            # 4. Generate sources information
            sources = []
            for result in results:
                metadata = result["metadata"]
                sources.append({
                    "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                    "file": metadata.get("filename", "Unknown"),
                    "page": metadata.get("page_num", "Unknown"),
                    "section": metadata.get("section", "Unknown"),
                    "score": f"{result['score']:.2f}"
                })
                
            # 5. Generate response using LLM
            # In a real implementation, this would call an LLM API
            # For now, we'll mock a basic response
            prompt = f"""
            Based on the following context, please answer the query.
            
            Query: {query_text}
            
            Context:
            {context}
            """
            
            # This is a placeholder - in production, call actual LLM
            answer = abc_response(prompt)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            }
            
    def get_available_documents(self) -> List[Dict]:
        """Get all available documents."""
        return self.qdrant_manager.get_document_ids()
        
    def delete_document(self, file_id: str) -> bool:
        """Delete a document."""
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
        return "I found some information but it doesn't seem to directly answer your query."
