import streamlit as st
import os
import time
import uuid
import re
import json
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib

# PDF processing
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

# Embeddings and vector database
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration for local paths
LOCAL_QDRANT_PATH = "./qdrant_pdf_rag_data"
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"
UPLOAD_FOLDER = "./uploads"
CHUNK_OVERLAP = 100  # Number of characters to overlap between chunks
CHUNK_SIZE = 1000     # Target chunk size in characters

# Ensure directories exist
os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# LLM response function (simulated - replace with your actual LLM)
def generate_answer(prompt: str) -> str:
    """Generate a response based on the given prompt."""
    # In a real application, replace this with a call to your LLM API
    return f"This is a simulated response based on the context provided: {prompt[:100]}...\n\nIn a complete implementation, this would be replaced with a real LLM response."

class PDFProcessor:
    """Handles PDF document processing, extraction and chunking."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF using both PyMuPDF and PDFMiner for robust extraction.
        PyMuPDF is faster but PDFMiner sometimes handles complex layouts better.
        """
        try:
            # First try with PyMuPDF (fitz)
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                
            doc.close()
            
            # If text extraction seems incomplete, try with pdfminer
            if len(text.strip()) < 100 and os.path.getsize(pdf_path) > 10000:
                logger.info(f"PyMuPDF extraction yielded limited text, trying PDFMiner for {pdf_path}")
                laparams = LAParams(
                    line_margin=0.5,
                    char_margin=2.0,
                    word_margin=0.1,
                    boxes_flow=0.5,
                    detect_vertical=True
                )
                text = extract_text(pdf_path, laparams=laparams)
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    @staticmethod
    def extract_metadata_from_pdf(pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata if doc.metadata else {}
            
            # Add page count and file info
            metadata['page_count'] = len(doc)
            metadata['file_size'] = os.path.getsize(pdf_path)
            metadata['filename'] = os.path.basename(pdf_path)
            
            # Add extraction timestamp
            metadata['extraction_time'] = datetime.now().isoformat()
            
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF: {e}")
            return {
                'filename': os.path.basename(pdf_path),
                'extraction_time': datetime.now().isoformat()
            }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text by handling common PDF extraction issues."""
        # Replace multiple newlines with single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix broken words at line breaks (common PDF issue)
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix spacing after periods, question marks, and exclamation points
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    @staticmethod
    def intelligent_chunking(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
        """
        Intelligently chunk the document text, preserving structural integrity.
        Returns chunks with metadata about their position in the document.
        """
        # Clean the text first
        text = PDFProcessor.clean_text(text)
        
        # Detect section headers (often have specific formatting)
        section_pattern = r'(?:^|\n)(?:[A-Z][A-Z\s]+:?|[IVX]+\.|[0-9]+\.|[A-Z]\.)(?:\s+)([A-Za-z].*?)(?=\n|$)'
        sections = re.finditer(section_pattern, text)
        section_positions = [match.start() for match in sections]
        
        # Add document start and end positions
        section_positions = [0] + section_positions + [len(text)]
        section_positions = sorted(list(set(section_positions)))
        
        chunks = []
        current_position = 0
        
        # Try to chunk at natural boundaries
        natural_boundaries = r'(?<=[.!?])\s+(?=[A-Z])'
        
        while current_position < len(text):
            # Calculate end position for this chunk
            end_position = min(current_position + chunk_size, len(text))
            
            # Adjust to respect section boundaries if nearby
            for pos in section_positions:
                if current_position < pos < end_position + chunk_overlap/2:
                    end_position = pos
                    break
            
            # If current chunk is too small and not at document end, extend it
            if end_position - current_position < chunk_size * 0.5 and end_position < len(text):
                next_section_idx = next((i for i, pos in enumerate(section_positions) if pos > end_position), None)
                if next_section_idx is not None and next_section_idx < len(section_positions):
                    end_position = section_positions[next_section_idx]
            
            # Fine-tune to end at natural boundaries if possible
            if end_position < len(text):
                natural_end_matches = list(re.finditer(natural_boundaries, text[current_position:end_position+100]))
                if natural_end_matches:
                    # Find the natural boundary closest to our target end position
                    closest_match = min(natural_end_matches, key=lambda m: abs(m.end() - (end_position - current_position)))
                    end_position = current_position + closest_match.end()
            
            # Extract the chunk
            chunk_text = text[current_position:end_position].strip()
            
            # Skip empty chunks
            if not chunk_text:
                current_position = end_position
                continue
            
            # Create chunk with metadata
            chunk = {
                "text": chunk_text,
                "start_char": current_position,
                "end_char": end_position,
                "chunk_id": str(uuid.uuid4()),
                "char_count": len(chunk_text)
            }
            
            chunks.append(chunk)
            
            # Move position for next chunk, accounting for overlap
            current_position = end_position - chunk_overlap
            
            # Ensure we make forward progress
            if current_position <= 0 or current_position >= len(text) - 1:
                break
        
        # Add sequential chunk numbers
        for i, chunk in enumerate(chunks):
            chunk["chunk_num"] = i
            
        return chunks

class VectorDBManager:
    """Manages interaction with the Qdrant vector database."""
    
    def __init__(self, 
                 collection_name: str = "pdf_documents", 
                 local_path: str = LOCAL_QDRANT_PATH,
                 model_path: str = LOCAL_MODEL_PATH):
        """Initialize the vector database manager."""
        self.collection_name = collection_name
        self.local_path = local_path
        self.model_path = model_path
        self.client = None
        self.model = None
    
    def initialize(self) -> bool:
        """Initialize the vector database and embedding model."""
        try:
            # Load embedding model
            if os.path.exists(self.model_path):
                logger.info(f"Loading embedding model from {self.model_path}")
                self.model = SentenceTransformer(self.model_path)
            else:
                logger.info("Downloading embedding model")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Save model for future use
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save(self.model_path)
            
            # Initialize Qdrant client
            self.client = QdrantClient(path=self.local_path)
            logger.info(f"Connected to Qdrant at {self.local_path}")
            
            # Check if collection exists, create if not
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                vector_size = self.model.get_sentence_embedding_dimension()
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info(f"Created collection '{self.collection_name}' with vector size {vector_size}")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        if not self.model:
            raise ValueError("Embedding model not initialized")
        
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not self.model:
            raise ValueError("Embedding model not initialized")
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], document_metadata: Dict[str, Any]) -> bool:
        """
        Insert or update document chunks in the vector database.
        Each chunk gets its own vector embedding along with metadata.
        """
        try:
            if not chunks:
                logger.warning("No chunks to insert")
                return False
                
            # Make sure document metadata has filename
            if 'filename' not in document_metadata:
                logger.warning("Document metadata missing filename")
                document_metadata['filename'] = f"unknown_doc_{uuid.uuid4()}"
            
            # Generate batch embeddings for all chunks (faster than one by one)
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.batch_get_embeddings(chunk_texts)
            
            # Prepare points for upload
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Add document metadata to chunk payload
                # Ensure we're using the correct structure
                payload = chunk.copy()
                payload["document_metadata"] = document_metadata
                
                # Create a point ID from document ID and chunk number
                doc_id = document_metadata.get("filename", "unknown").replace(".", "_")
                point_id = f"{doc_id}_{chunk.get('chunk_num', i)}"
                point_id_hash = abs(hash(point_id)) % (2**63 - 1)  # Convert to positive integer
                
                point = PointStruct(
                    id=point_id_hash,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload to Qdrant in batches to avoid timeout issues
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} with {len(batch)} chunks")
            
            # Log success information
            logger.info(f"Successfully inserted {len(chunks)} chunks for document {document_metadata.get('filename', 'unknown')}")
            
            return True
        except Exception as e:
            logger.error(f"Error inserting chunks: {e}")
            return False
    
    def semantic_search(self, query: str, top_k: int = 5, filter_condition: Optional[models.Filter] = None) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        Returns the chunks with their similarity scores and metadata.
        """
        try:
            # Generate embedding for query
            query_embedding = self.get_embedding(query)
            
            # Search in collection
            search_params = models.SearchParams(hnsw_ef=128, exact=False)
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_condition,
                limit=top_k,
                search_params=search_params
            )
            
            # Extract and return the results in a more usable format
            processed_results = []
            for result in results:
                processed_results.append({
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": {
                        "document": result.payload.get("document_metadata", {}).get("filename", "Unknown"),
                        "chunk_num": result.payload.get("chunk_num", 0),
                        "chunk_id": result.payload.get("chunk_id", ""),
                        "start_char": result.payload.get("start_char", 0),
                        "end_char": result.payload.get("end_char", 0)
                    }
                })
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def get_document_metadata(self, document_filename: str) -> List[Dict[str, Any]]:
        """Retrieve all metadata for a specific document."""
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_metadata.filename",
                        match=models.MatchValue(value=document_filename)
                    )
                ]
            )
            
            # Get all points for this document
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1  # We only need one to get the metadata
            )
            
            if results and results[0]:
                return [point.payload.get("document_metadata", {}) for point in results[0]]
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving document metadata: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get unique documents by querying for unique filenames
            unique_files = set()
            batch_size = 100
            offset = 0
            
            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset
                )
                
                if not results or not results[0]:
                    break
                    
                for point in results[0]:
                    # Look for document_metadata.filename in the payload
                    if 'document_metadata' in point.payload and 'filename' in point.payload['document_metadata']:
                        filename = point.payload['document_metadata']['filename']
                        if filename:
                            unique_files.add(filename)
                
                offset += len(results[0])
                if len(results[0]) < batch_size:
                    break
            
            return {
                "total_chunks": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "unique_documents": len(unique_files),
                "document_list": list(unique_files)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "total_chunks": 0,
                "vector_size": 0,
                "unique_documents": 0,
                "document_list": []
            }

def process_and_store_pdf(pdf_path: str, db_manager: VectorDBManager, status_callback=None) -> Tuple[bool, str]:
    """Process a PDF and store its chunks in the vector database."""
    try:
        logger.info(f"Beginning processing of {pdf_path}")
        if status_callback:
            status_callback("Extracting text from PDF...", 10)
        
        # Extract text and metadata
        text = PDFProcessor.extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.error(f"Failed to extract text from PDF: {pdf_path}")
            return False, "Failed to extract text from PDF."
        
        logger.info(f"Successfully extracted {len(text)} characters of text from {pdf_path}")
        
        if status_callback:
            status_callback("Extracting document metadata...", 30)
            
        metadata = PDFProcessor.extract_metadata_from_pdf(pdf_path)
        logger.info(f"Extracted metadata: {metadata}")
        
        # Ensure filename is present in metadata
        if 'filename' not in metadata:
            metadata['filename'] = os.path.basename(pdf_path)
            logger.info(f"Added filename to metadata: {metadata['filename']}")
        
        if status_callback:
            status_callback("Chunking document text...", 40)
            
        # Chunk the text
        chunks = PDFProcessor.intelligent_chunking(text)
        if not chunks:
            logger.error(f"Failed to chunk document: {pdf_path}")
            return False, "Failed to chunk the document."
        
        logger.info(f"Successfully chunked document into {len(chunks)} chunks")
        
        if status_callback:
            status_callback("Generating embeddings and storing in database...", 60)
            
        # Store chunks in vector database
        success = db_manager.upsert_chunks(chunks, metadata)
        if not success:
            logger.error(f"Failed to store chunks in vector database: {pdf_path}")
            return False, "Failed to store chunks in the vector database."
        
        logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
        
        if status_callback:
            status_callback("Processing complete!", 100)
            
        return True, f"Successfully processed {metadata.get('filename')} with {len(chunks)} chunks."
    
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return False, f"Error processing PDF: {str(e)}"


# Chat history functions
def init_chat_history():
    """Initialize chat history in session state if not already present."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_to_chat_history(role, content):
    """Add a message to the chat history."""
    if 'chat_history' not in st.session_state:
        init_chat_history()
    
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

def clear_chat_history():
    """Clear the chat history."""
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []
        
def get_document_filter_state():
    """Get or initialize the document filter state."""
    if 'document_filter' not in st.session_state:
        st.session_state.document_filter = "All Documents"
    return st.session_state.document_filter

def set_document_filter(value):
    """Set the document filter state."""
    st.session_state.document_filter = value


def main():
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.documents_processed = 0
        st.session_state.db_manager = VectorDBManager()
        st.session_state.processing_file = False
        st.session_state.app_view = "upload"  # Default view

    # Initialize chat history
    init_chat_history()
    
    # App header with a simple logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# 📚")
    with col2:
        st.title("PDF Chat Assistant")
        
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Navigation buttons
        if st.button("📄 Upload Documents", use_container_width=True):
            st.session_state.app_view = "upload"
            
        if st.button("💬 Chat with Documents", use_container_width=True):
            st.session_state.app_view = "chat"
            
        if st.button("ℹ️ System Info", use_container_width=True):
            st.session_state.app_view = "info"
            
        # Initialize system if not already done
        if not st.session_state.initialized:
            with st.spinner("Initializing system... This may take a moment."):
                success = st.session_state.db_manager.initialize()
                if success:
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ Failed to initialize system. Please check the logs.")
                    return
        
        # Show document statistics in sidebar
        st.divider()
        st.subheader("Document Stats")
        
        # Get fresh document stats
        db_stats = st.session_state.db_manager.get_collection_stats()
        doc_count = db_stats.get("unique_documents", 0)
        chunk_count = db_stats.get("total_chunks", 0)
        
        st.metric("Documents", doc_count)
        st.metric("Total Chunks", chunk_count)
        
        if doc_count > 0:
            doc_list = db_stats.get("document_list", [])
            
            st.divider()
            st.subheader("Available Documents")
            
            for doc_name in doc_list:
                st.markdown(f"- {doc_name}")
    
    # Main content area - different views
    if st.session_state.app_view == "upload":
        render_upload_view()
    elif st.session_state.app_view == "chat":
        render_chat_view()
    elif st.session_state.app_view == "info":
        render_info_view()

def render_upload_view():
    """Render the document upload view."""
    st.header("Upload Documents")
    st.markdown("""
    Upload PDF files to be processed and embedded for chatting. 
    Each document will be analyzed, split into chunks, and stored in the vector database.
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Upload controls
        with st.expander("Processing Settings", expanded=False):
            debug_mode = st.checkbox("Show Detailed Processing Log", value=False)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            process_button = st.button(
                "Process Selected Files",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing_file
            )
            
        if process_button:
            st.session_state.processing_file = True
            
            # Create a container for progress information
            progress_container = st.container()
            
            for i, uploaded_file in enumerate(uploaded_files):
                with progress_container:
                    # Create detailed progress reporting
                    st.markdown(f"### Processing file {i+1}/{len(uploaded_files)}")
                    st.markdown(f"**File:** {uploaded_file.name}")
                    
                    # Create status areas for each processing stage
                    status_area = st.empty()
                    
                    # Overall progress
                    progress_bar = st.progress(0)
                    
                    # Log area for debug messages
                    if debug_mode:
                        log_expander = st.expander("Processing Log", expanded=True)
                        log_area = log_expander.empty()
                        log_messages = []
                        
                        # Function to add log messages
                        def add_log(message):
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            log_messages.append(f"[{timestamp}] {message}")
                            log_area.code("\n".join(log_messages), language="bash")
                    else:
                        def add_log(message):
                            pass  # No-op if debug mode is off
                    
                    # Callback function to update progress
                    def update_status(status_text, progress_value):
                        status_area.markdown(f"**Status:** {status_text}")
                        progress_bar.progress(progress_value)
                        add_log(status_text)
                    
                    # Calculate and display file info
                    file_size_kb = len(uploaded_file.getvalue()) / 1024
                    add_log(f"Starting to process {uploaded_file.name} ({file_size_kb:.2f} KB)")
                    update_status(f"Starting processing of {uploaded_file.name}", 5)
                    
                    # Save uploaded file to disk temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_path = tmp_file.name
                        tmp_file.write(uploaded_file.getvalue())
                        add_log(f"Saved temporary file to {tmp_path}")
                    
                    # Process and store the PDF with progress updates
                    success, message = process_and_store_pdf(
                        tmp_path,
                        st.session_state.db_manager,
                        update_status
                    )
                    
                    # Final status update
                    if success:
                        st.success(message)
                        st.session_state.documents_processed += 1
                        add_log("Processing completed successfully")
                    else:
                        st.error(message)
                        add_log(f"ERROR: {message}")
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                        add_log(f"Temporary file {tmp_path} removed")
                    except Exception as e:
                        add_log(f"Warning: Could not remove temporary file: {e}")
                
                # Add a separator between files
                st.divider()
            
            st.success(f"Finished processing {len(uploaded_files)} files.")
            st.session_state.processing_file = False
            
            # Suggestion to go to chat
            st.info("Your documents are now ready! Click on 'Chat with Documents' in the sidebar to start asking questions.")
            if st.button("Go to Chat"):
                st.session_state.app_view = "chat"
                st.experimental_rerun()
    else:
        st.info("Please upload PDF files to get started.")

def render_chat_view():
    """Render the chat interface view."""
    st.header("Chat with your Documents")
    
    # Check if we have documents
    db_stats = st.session_state.db_manager.get_collection_stats()
    doc_count = db_stats.get("unique_documents", 0)
    
    if doc_count == 0:
        st.warning("⚠️ No documents have been processed yet. Please upload and process some documents first.")
        
        if st.button("Go to Upload"):
            st.session_state.app_view = "upload"
            st.experimental_rerun()
        return
    
    # Chat controls
    chat_container = st.container()
    with chat_container:
        # Document filter in the top right
        doc_list = ["All Documents"] + db_stats.get("document_list", [])
        col1, col2 = st.columns([3, 1])
        
        with col2:
            selected_filter = st.selectbox(
                "Filter documents:",
                options=doc_list,
                index=0
            )
            set_document_filter(selected_filter)
            
        with col1:
            # Chat settings
            with st.expander("Chat Settings", expanded=False):
                top_k = st.slider("Number of chunks to retrieve:", 1, 10, 3)
                show_context = st.checkbox("Show context sources", value=False)
        
        # Chat history display
        st.divider()
        
        # Display the chat history
        chat_placeholder = st.container()
        with chat_placeholder:
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.chat_message("user").write(content)
                else:
                    with st.chat_message("assistant"):
                        st.write(content["answer"])
                        
                        # Show context sources if enabled
                        if show_context and "context" in content:
                            with st.expander("View source documents"):
                                for i, ctx in enumerate(content["context"]):
                                    st.markdown(f"**Source {i+1}:** {ctx['metadata']['document']} (Relevance: {ctx['score']:.2f})")
                                    st.markdown(f"```\n{ctx['text'][:300]}{'...' if len(ctx['text']) > 300 else ''}\n```")
        
        # User input area
        st.divider()
        
        # Chat input
        user_query = st.chat_input("Ask a question about your documents...")
        
        # Clear chat button
        clear_col1, clear_col2 = st.columns([5, 1])
        with clear_col2:
            if st.button("Clear Chat", use_container_width=True):
                clear_chat_history()
                st.experimental_rerun()
        
        # Process the user query
        if user_query:
            # Add user message to chat
            add_to_chat_history("user", user_query)
            
            # Display user message immediately
            st.chat_message("user").write(user_query)
            
            # Prepare filter condition if needed
            filter_condition = None
            if get_document_filter_state() != "All Documents":
                filter_condition = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_metadata.filename",
                            match=models.MatchValue(value=get_document_filter_state())
                        )
                    ]
                )
            
            # Show assistant "thinking"
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("⏳ Searching for relevant information...")
                
                # Perform semantic search
                results = st.session_state.db_manager.semantic_search(
                    user_query,
                    top_k=top_k,
                    filter_condition=filter_condition
                )
                
                if not results:
                    response = {
                        "answer": "I couldn't find relevant information in the documents to answer your question. Could you try rephrasing or ask something else?",
                        "context": []
                    }
                else:
                    # Prepare context for LLM
                    context_text = "\n\n".join([
                        f"[Document: {r['metadata']['document']}, Chunk: {r['metadata']['chunk_num']}]\n{r['text']}" 
                        for r in results
                    ])
                    
                    # Prepare prompt for LLM
                    prompt = f"""
                    Answer the question based on the following context from documents:
                    
                    CONTEXT:
                    {context_text}
                    
                    QUESTION:
                    {user_query}
                    
                    Provide a comprehensive answer using information from the context. 
                    If the answer is not contained in the context, say "I don't have enough information to answer this question."
                    """
                    
                    thinking_placeholder.markdown("⏳ Generating answer...")
                    
                    # Get response from LLM
                    answer = generate_answer(prompt)
                    
                    response = {
                        "answer": answer,
                        "context": results
                    }
                
                # Update assistant message
                thinking_placeholder.empty()
                st.write(response["answer"])
                
                # Show context sources if enabled
                if show_context and "context" in response:
                    with st.expander("View source documents"):
                        for i, ctx in enumerate(response["context"]):
                            st.markdown(f"**Source {i+1}:** {ctx['metadata']['document']} (Relevance: {ctx['score']:.2f})")
                            st.markdown(f"```\n{ctx['text'][:300]}{'...' if len(ctx['text']) > 300 else ''}\n```")
            
            # Add assistant response to chat history
            add_to_chat_history("assistant", response)
        
def render_info_view():
    """Render the system information view."""
    st.header("System Information")
    st.markdown("This section provides information about the system and the documents in the database.")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        refresh_button = st.button("Refresh Information", type="primary", use_container_width=True)
    
    # Tabs for different info views
    tab1, tab2, tab3 = st.tabs(["Database Stats", "Document List", "System Configuration"])
    
    # Get fresh stats if refresh button clicked
    if refresh_button or 'system_info_stats' not in st.session_state:
        with st.spinner("Fetching system information..."):
            st.session_state.system_info_stats = st.session_state.db_manager.get_collection_stats()
    
    db_stats = st.session_state.system_info_stats
    
    # Tab 1: Database Stats
    with tab1:
        st.subheader("Vector Database Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", db_stats.get("unique_documents", 0))
        with col2:
            st.metric("Total Chunks", db_stats.get("total_chunks", 0))
        with col3:
            st.metric("Vector Dimension", db_stats.get("vector_size", 0))
            
        # Visual representation of the database
        st.subheader("Document Distribution")
        
        # If there are documents, show a simple bar chart of chunk counts
        if db_stats.get("unique_documents", 0) > 0:
            doc_list = db_stats.get("document_list", [])
            chunk_counts = []
            
            for doc in doc_list:
                # This is a simplification - in a real app you'd query the actual counts
                # Here we're generating random numbers for demonstration
                chunk_counts.append({"document": doc, "chunks": int(np.random.normal(30, 10))})
                
            df = pd.DataFrame(chunk_counts)
            st.bar_chart(df, x="document", y="chunks")
        else:
            st.info("No documents have been processed yet.")
            
    # Tab 2: Document List
    with tab2:
        st.subheader("Document List")
        
        if db_stats.get("document_list"):
            # Display documents as a dataframe
            doc_data = []
            for doc in db_stats.get("document_list", []):
                # Get metadata for this document
                metadata_list = st.session_state.db_manager.get_document_metadata(doc)
                metadata = metadata_list[0] if metadata_list else {}
                
                doc_data.append({
                    "Filename": doc,
                    "Title": metadata.get("title", "Unknown"),
                    "Pages": metadata.get("page_count", "Unknown"),
                    "Added on": metadata.get("extraction_time", "Unknown")
                })
            
            df = pd.DataFrame(doc_data)
            st.dataframe(df, use_container_width=True)
            
            # Option to chat with a specific document
            if st.button("Chat with Selected Document"):
                st.session_state.app_view = "chat"
                # This would need to be connected to the actual selection
                st.experimental_rerun()
        else:
            st.info("No documents have been processed yet.")
            
    # Tab 3: System Configuration
    with tab3:
        st.subheader("System Configuration")
        
        config = {
            "Qdrant Storage Path": LOCAL_QDRANT_PATH,
            "Embedding Model Path": LOCAL_MODEL_PATH,
            "Upload Folder": UPLOAD_FOLDER,
            "Chunk Size": CHUNK_SIZE,
            "Chunk Overlap": CHUNK_OVERLAP,
            "Vector Database": "Qdrant (Local)",
            "Embedding Model": "all-MiniLM-L6-v2"
        }
        
        # Display as a nicer table instead of raw JSON
        config_data = [{"Parameter": k, "Value": v} for k, v in config.items()]
        st.table(pd.DataFrame(config_data))
        
        # System maintenance options
        st.subheader("Maintenance")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Documents", use_container_width=True):
                # This would need confirmation in a real app
                st.warning("This feature is not implemented in this demo.")
        
        with col2:
            if st.button("Reset System", use_container_width=True):
                # This would need confirmation in a real app
                st.warning("This feature is not implemented in this demo.")

if __name__ == "__main__":
    main()
