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
            
            # Check if model directory exists
            model_exists = os.path.exists(self.model_path) and os.path.isdir(self.model_path)
            
            # Configure the embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=self.config.get("MODEL_DIR"),
                model_kwargs={'device': 'cpu'}
            )
            
            logger.info(f"✅ Embeddings model loaded successfully: {self.model_name}")
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
        self.collection_name = "pdf_documents"
        
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
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            # Create vector store
            vector_store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            
            if self.collection_name not in collection_names:
                logger.info(f"Collection '{self.collection_name}' created")
            else:
                logger.info(f"Using existing collection '{self.collection_name}'")
                
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
            
            # Add documents in batches
            batch_size = 50
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i+batch_size]
                self.vector_store.add_documents(batch)
                
                if progress_callback:
                    progress_callback(min(i + batch_size, total_docs), total_docs, 
                                     f"Embedded {min(i + batch_size, total_docs)} of {total_docs} chunks")
                    
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
        self.vector_store_manager = VectorStoreManager(
            self.config_manager, 
            self.embedding_manager.embeddings
        )
    
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
            
            # 2. Add documents to vector store
            stored = self.vector_store_manager.add_documents(
                documents,
                processed_doc.file_id,
                processed_doc.filename,
                embedding_callback
            )
            
            if not stored:
                processed_doc.status = "error"
                processed_doc.summary += "\nError: Failed to store document in vector database."
                logger.warning(f"Failed to store documents for file: {file_path}")
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
            # 1. Get retriever with file filters if specified
            retriever = self.vector_store_manager.get_retriever(file_ids)
            
            # 2. Retrieve relevant documents
            docs = retriever.get_relevant_documents(query_text)
            
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
