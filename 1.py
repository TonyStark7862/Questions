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

# Mock LLM response function (to be replaced with actual LLM)
def abc_response(prompt: str) -> str:
    """Simulated LLM response function."""
    return f"This is a simulated response based on the context provided: {prompt[:100]}..."

class PDFProcessor:
    """Handles PDF document processing, extraction and chunking."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str, add_log=None) -> str:
        """
        Extract text from PDF using both PyMuPDF and PDFMiner for robust extraction.
        PyMuPDF is faster but PDFMiner sometimes handles complex layouts better.
        """
        try:
            if add_log:
                add_log("Starting PDF text extraction with PyMuPDF")
            
            # First try with PyMuPDF (fitz)
            doc = fitz.open(pdf_path)
            text = ""
            
            if add_log:
                add_log(f"PDF has {len(doc)} pages")
            
            for page_num in range(len(doc)):
                if add_log:
                    add_log(f"Extracting text from page {page_num+1}/{len(doc)}")
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text
                
            doc.close()
            
            # If text extraction seems incomplete, try with pdfminer
            if len(text.strip()) < 100 and os.path.getsize(pdf_path) > 10000:
                if add_log:
                    add_log("PyMuPDF extraction yielded limited text, trying PDFMiner")
                    
                laparams = LAParams(
                    line_margin=0.5,
                    char_margin=2.0,
                    word_margin=0.1,
                    boxes_flow=0.5,
                    detect_vertical=True
                )
                text = extract_text(pdf_path, laparams=laparams)
                
                if add_log:
                    add_log(f"PDFMiner extraction completed, extracted {len(text)} characters")
            else:
                if add_log:
                    add_log(f"PyMuPDF extraction completed, extracted {len(text)} characters")
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            if add_log:
                add_log(f"ERROR in text extraction: {str(e)}")
            return ""
    
    @staticmethod
    def extract_metadata_from_pdf(pdf_path: str, add_log=None) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        try:
            if add_log:
                add_log("Extracting PDF metadata")
                
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Add page count and file info
            metadata['page_count'] = len(doc)
            metadata['file_size'] = os.path.getsize(pdf_path)
            metadata['filename'] = os.path.basename(pdf_path)
            
            # Add extraction timestamp
            metadata['extraction_time'] = datetime.now().isoformat()
            
            doc.close()
            
            if add_log:
                add_log(f"Metadata extraction complete: {len(metadata)} metadata fields")
                
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF: {e}")
            if add_log:
                add_log(f"ERROR in metadata extraction: {str(e)}")
            return {
                'filename': os.path.basename(pdf_path),
                'extraction_time': datetime.now().isoformat()
            }
    
    @staticmethod
    def clean_text(text: str, add_log=None) -> str:
        """Clean extracted text by handling common PDF extraction issues."""
        if add_log:
            add_log(f"Cleaning extracted text ({len(text)} characters)")
            
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
        
        if add_log:
            add_log(f"Text cleaning complete, final length: {len(text.strip())} characters")
            
        return text.strip()
    
    @staticmethod
    def intelligent_chunking(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, add_log=None) -> List[Dict[str, Any]]:
        """
        Intelligently chunk the document text, preserving structural integrity.
        Returns chunks with metadata about their position in the document.
        """
        # Clean the text first
        if add_log:
            add_log("Starting intelligent chunking process")
            
        text = PDFProcessor.clean_text(text, add_log)
        
        # Detect section headers (often have specific formatting)
        if add_log:
            add_log("Detecting document section boundaries")
            
        section_pattern = r'(?:^|\n)(?:[A-Z][A-Z\s]+:?|[IVX]+\.|[0-9]+\.|[A-Z]\.)(?:\s+)([A-Za-z].*?)(?=\n|$)'
        sections = re.finditer(section_pattern, text)
        section_positions = [match.start() for match in sections]
        
        # Add document start and end positions
        section_positions = [0] + section_positions + [len(text)]
        section_positions = sorted(list(set(section_positions)))
        
        if add_log:
            add_log(f"Detected {len(section_positions)-2} potential section boundaries")
        
        chunks = []
        current_position = 0
        
        # Try to chunk at natural boundaries
        natural_boundaries = r'(?<=[.!?])\s+(?=[A-Z])'
        
        chunking_progress = 0
        estimated_chunks = max(1, len(text) // (chunk_size - chunk_overlap))
        
        if add_log:
            add_log(f"Beginning chunking with target size of {chunk_size} characters and {chunk_overlap} overlap")
            add_log(f"Estimated number of chunks: ~{estimated_chunks}")
        
        while current_position < len(text):
            chunking_progress += 1
            if add_log and chunking_progress % 5 == 0:
                add_log(f"Processing chunk {chunking_progress}...")
                
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
        
        if add_log:
            add_log(f"Chunking complete: created {len(chunks)} chunks")
            avg_chunk_size = sum(c["char_count"] for c in chunks) / max(1, len(chunks))
            add_log(f"Average chunk size: {avg_chunk_size:.1f} characters")
            
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
    
    def batch_get_embeddings(self, texts: List[str], add_log=None) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not self.model:
            raise ValueError("Embedding model not initialized")
        
        if add_log:
            add_log(f"Generating embeddings for {len(texts)} chunks")
            
        embeddings = self.model.encode(texts)
        
        if add_log:
            add_log(f"Embedding generation complete. Vector size: {embeddings.shape[1]}")
            
        return embeddings.tolist()
    
    def check_document_exists(self, filename: str) -> bool:
        """Check if a document with the given filename already exists in the collection."""
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_metadata.filename",
                        match=models.MatchValue(value=filename)
                    )
                ]
            )
            
            # Check if any points exist with this filename
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1  # We only need one to confirm existence
            )
            
            return bool(results and results[0])
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], document_metadata: Dict[str, Any], add_log=None) -> bool:
        """
        Insert or update document chunks in the vector database.
        Each chunk gets its own vector embedding along with metadata.
        """
        try:
            if not chunks:
                logger.warning("No chunks to insert")
                if add_log:
                    add_log("WARNING: No chunks to insert")
                return False
            
            # Check if document already exists
            if self.check_document_exists(document_metadata.get('filename')):
                if add_log:
                    add_log(f"Document {document_metadata.get('filename')} already exists in the database. Skipping.")
                return True
            
            # Generate batch embeddings for all chunks (faster than one by one)
            if add_log:
                add_log(f"Preparing to generate embeddings for {len(chunks)} chunks")
                
            chunk_texts = [chunk["text"] for chunk in chunks]
            
            # Process embeddings in batches if there are many chunks
            batch_size = 100  # Adjust based on memory constraints
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i+batch_size]
                if add_log:
                    add_log(f"Generating embeddings for batch {i//batch_size + 1}/{(len(chunk_texts)-1)//batch_size + 1} ({len(batch)} chunks)")
                batch_embeddings = self.batch_get_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                
            if add_log:
                add_log(f"Completed embedding generation for all {len(chunks)} chunks")
            
            # Prepare points for upload
            if add_log:
                add_log("Preparing points for database insertion")
                
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                # Add document metadata to chunk
                payload = {**chunk, "document_metadata": document_metadata}
                
                # Create a point ID from document ID and chunk number
                doc_id = document_metadata.get("filename", "unknown").replace(".", "_")
                point_id = f"{doc_id}_{chunk['chunk_num']}"
                point_id_hash = abs(hash(point_id)) % (2**63 - 1)  # Convert to positive integer
                
                point = PointStruct(
                    id=point_id_hash,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload to Qdrant in batches to avoid timeout issues
            upload_batch_size = 100
            for i in range(0, len(points), upload_batch_size):
                batch = points[i:i+upload_batch_size]
                if add_log:
                    add_log(f"Uploading batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1} to Qdrant ({len(batch)} points)")
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Inserted batch {i//upload_batch_size + 1}/{(len(points)-1)//upload_batch_size + 1} with {len(batch)} chunks")
            
            if add_log:
                add_log(f"Successfully inserted all {len(points)} chunks into Qdrant")
                
            return True
        except Exception as e:
            logger.error(f"Error inserting chunks: {e}")
            if add_log:
                add_log(f"ERROR inserting chunks: {str(e)}")
            return False
    
    def semantic_search(self, query: str, top_k: int = 5, filter_condition: Optional[models.Filter] = None) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        Returns the chunks with their similarity scores and metadata.
        """
        try:
            # Generate embedding for query
            query_embedding = self.get_embedding(query)
            
            # Search in collection using query_points (not deprecated search)
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
                    filename = point.payload.get("document_metadata", {}).get("filename")
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

def process_and_store_pdf(pdf_path: str, db_manager: VectorDBManager, add_log=None, extraction_timeout=120) -> Tuple[bool, str]:
    """Process a PDF and store its chunks in the vector database."""
    try:
        # Extract text and metadata
        if add_log:
            add_log(f"Starting processing of {os.path.basename(pdf_path)}")
            
        text = PDFProcessor.extract_text_from_pdf(pdf_path, add_log)
        if not text.strip():
            if add_log:
                add_log("ERROR: Failed to extract text from PDF")
            return False, "Failed to extract text from PDF."
        
        metadata = PDFProcessor.extract_metadata_from_pdf(pdf_path, add_log)
        
        # Check if document already exists
        if db_manager.check_document_exists(metadata.get('filename')):
            if add_log:
                add_log(f"Document {metadata.get('filename')} already exists in the database")
            return True, f"Document {metadata.get('filename')} already processed (skipped re-processing)."
        
        # Chunk the text
        chunks = PDFProcessor.intelligent_chunking(text, add_log=add_log)
        if not chunks:
            if add_log:
                add_log("ERROR: Failed to chunk the document")
            return False, "Failed to chunk the document."
        
        # Store chunks in vector database
        success = db_manager.upsert_chunks(chunks, metadata, add_log)
        if not success:
            if add_log:
                add_log("ERROR: Failed to store chunks in the vector database")
            return False, "Failed to store chunks in the vector database."
        
        if add_log:
            add_log(f"Successfully processed {metadata.get('filename')} with {len(chunks)} chunks")
            
        return True, f"Successfully processed {metadata.get('filename')} with {len(chunks)} chunks."
    
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        if add_log:
            add_log(f"ERROR processing PDF: {str(e)}")
        return False, f"Error processing PDF: {str(e)}"

def main():
    st.set_page_config(
        page_title="PDF RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Retrieval-Augmented Generation System")
    st.markdown("""
    This application allows you to:
    1. Upload PDF documents for processing and embedding
    2. Ask questions about your documents
    3. Receive answers based on the content of your documents
    
    All data is stored locally for privacy and persistence.
    """)
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.documents_processed = 0
        st.session_state.processed_document_list = []
        st.session_state.db_manager = VectorDBManager()
    
    # Initialize vector database and embedding model
    if not st.session_state.initialized:
        with st.spinner("Initializing system... This may take a moment."):
            success = st.session_state.db_manager.initialize()
            if success:
                st.session_state.initialized = True
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system. Please check the logs.")
                return
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Upload & Process", "Ask Questions", "System Info"])
    
    # Tab 1: Upload & Process
    with tab1:
        st.header("Upload Documents")
        st.markdown("Upload PDF files to be processed and embedded in the vector database.")
        
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            # Add a settings section for debugging
            with st.expander("Processing Settings", expanded=False):
                debug_mode = st.checkbox("Enable Verbose Debugging", value=True)
                extraction_timeout = st.slider("PDF Extraction Timeout (seconds)", 30, 300, 120)
            
            process_button = st.button("Process Selected Files", type="primary")
            
            if process_button:
                # Create a container for progress information
                progress_container = st.container()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with progress_container:
                        # Create detailed progress reporting
                        st.markdown(f"### Processing file {i+1}/{len(uploaded_files)}")
                        st.markdown(f"**File:** {uploaded_file.name}")
                        
                        # Create status areas for each processing stage
                        file_info = st.empty()
                        extraction_status = st.empty()
                        chunking_status = st.empty()
                        embedding_status = st.empty()
                        db_status = st.empty()
                        
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
                        
                        # Calculate and display file info
                        file_size_kb = len(uploaded_file.getvalue()) / 1024
                        file_info.markdown(f"📄 **File size:** {file_size_kb:.2f} KB")
                        add_log(f"Starting to process {uploaded_file.name} ({file_size_kb:.2f} KB)")
                        
                        # Save uploaded file to disk temporarily
                        extraction_status.markdown("⏳ **Step 1/4:** Extracting text from PDF...")
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_path = tmp_file.name
                            tmp_file.write(uploaded_file.getvalue())
                            add_log(f"Saved temporary file to {tmp_path}")
                        
                        # Extract text with timeout and progress updates
                        add_log("Starting text extraction...")
                        progress_bar.progress(10)
                        
                        # Process the PDF with real extraction and progress updates
                        try:
                            # Extract text
                            add_log("Opening PDF document")
                            extraction_status.markdown("⏳ **Step 1/4:** Opening PDF document...")
                            
                            # Extract text from PDF
                            text = PDFProcessor.extract_text_from_pdf(tmp_path, add_log)
                            if not text.strip():
                                raise ValueError("Failed to extract text from PDF")
                                
                            extraction_status.markdown("✅ **Step 1/4:** Text extracted successfully")
                            progress_bar.progress(30)
                            
                            # Extract metadata
                            add_log("Extracting document metadata")
                            extraction_status.markdown("⏳ **Step 1/4:** Extracting document metadata...")
                            metadata = PDFProcessor.extract_metadata_from_pdf(tmp_path, add_log)
                            progress_bar.progress(35)
                            
                            # Check if document already exists
                            if st.session_state.db_manager.check_document_exists(metadata.get('filename')):
                                add_log(f"Document {metadata.get('filename')} already exists in the database")
                                st.success(f"Document {metadata.get('filename')} already processed (skipped re-processing).")
                                progress_bar.progress(100)
                                continue
                            
                            # Chunking
                            chunking_status.markdown("⏳ **Step 2/4:** Chunking document text...")
                            add_log("Starting text chunking process")
                            chunks = PDFProcessor.intelligent_chunking(text, add_log=add_log)
                            
                            if not chunks:
                                raise ValueError("Failed to chunk the document")
                            
                            chunking_status.markdown("✅ **Step 2/4:** Document chunked successfully")
                            progress_bar.progress(50)
                            
# Embedding Generation
                            embedding_status.markdown("⏳ **Step 3/4:** Generating embeddings...")
                            add_log("Starting embedding generation")
                            
                            # Generate embeddings in batches
                            chunk_texts = [chunk["text"] for chunk in chunks]
                            batch_size = 50  # Adjust based on your memory constraints
                            
                            all_embeddings = []
                            for j in range(0, len(chunk_texts), batch_size):
                                batch = chunk_texts[j:j+batch_size]
                                batch_num = j // batch_size + 1
                                total_batches = (len(chunk_texts) - 1) // batch_size + 1
                                
                                add_log(f"Generating embeddings for batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                                embedding_status.markdown(f"⏳ **Step 3/4:** Generating embeddings (batch {batch_num}/{total_batches})...")
                                progress_value = 50 + (batch_num / total_batches) * 30
                                progress_bar.progress(int(progress_value))
                                
                                batch_embeddings = st.session_state.db_manager.batch_get_embeddings(batch)
                                all_embeddings.extend(batch_embeddings)
                            
                            embedding_status.markdown("✅ **Step 3/4:** Embeddings generated successfully")
                            progress_bar.progress(80)
                            
                            # Database Storage
                            db_status.markdown("⏳ **Step 4/4:** Storing in vector database...")
                            add_log("Preparing database points")
                            
                            # Prepare points for upload
                            points = []
                            for j, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                                # Add document metadata to chunk
                                payload = {**chunk, "document_metadata": metadata}
                                
                                # Create a point ID from document ID and chunk number
                                doc_id = metadata.get("filename", "unknown").replace(".", "_")
                                point_id = f"{doc_id}_{chunk['chunk_num']}"
                                point_id_hash = abs(hash(point_id)) % (2**63 - 1)  # Convert to positive integer
                                
                                point = PointStruct(
                                    id=point_id_hash,
                                    vector=embedding,
                                    payload=payload
                                )
                                points.append(point)
                            
                            # Upload to Qdrant in batches
                            upload_batch_size = 100
                            for j in range(0, len(points), upload_batch_size):
                                batch = points[j:j+upload_batch_size]
                                batch_num = j // upload_batch_size + 1
                                total_batches = (len(points) - 1) // upload_batch_size + 1
                                
                                add_log(f"Uploading batch {batch_num}/{total_batches} to Qdrant ({len(batch)} points)")
                                db_status.markdown(f"⏳ **Step 4/4:** Storing in database (batch {batch_num}/{total_batches})...")
                                progress_value = 80 + (batch_num / total_batches) * 20
                                progress_bar.progress(int(progress_value))
                                
                                st.session_state.db_manager.client.upsert(
                                    collection_name=st.session_state.db_manager.collection_name,
                                    points=batch
                                )
                            
                            db_status.markdown("✅ **Step 4/4:** Storage in vector database completed")
                            progress_bar.progress(100)
                            
                            # Success message
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                            add_log(f"Complete processing pipeline finished successfully for {uploaded_file.name}")
                            st.session_state.documents_processed += 1
                            
                            # Add to processed document list
                            if metadata.get('filename') not in st.session_state.processed_document_list:
                                st.session_state.processed_document_list.append(metadata.get('filename'))
                            
                        except Exception as e:
                            error_msg = str(e)
                            add_log(f"ERROR: {error_msg}")
                            st.error(f"❌ Error processing {uploaded_file.name}: {error_msg}")
                            progress_bar.progress(100)  # Complete the progress bar
                        
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_path)
                            add_log(f"Temporary file {tmp_path} removed")
                        except Exception as e:
                            add_log(f"Warning: Could not remove temporary file: {e}")
                    
                    # Add a separator between files
                    st.divider()
                
                st.success(f"Finished processing {len(uploaded_files)} files.")
    
    # Tab 2: Ask Questions
    with tab2:
        st.header("Ask Questions")
        st.markdown("Ask questions about your documents and get answers based on their content.")
        
        # Get document stats
        db_stats = st.session_state.db_manager.get_collection_stats()
        doc_count = db_stats.get("unique_documents", 0)
        chunk_count = db_stats.get("total_chunks", 0)
        
        if doc_count == 0 and st.session_state.documents_processed == 0:
            st.warning("⚠️ No documents have been processed yet. Please upload and process some documents first.")
        else:
            st.info(f"📊 You have {doc_count} documents with {chunk_count} total chunks in the database.")
            
            # Document filter
            doc_list = db_stats.get("document_list", [])
            doc_filter = st.multiselect(
                "Filter by document (optional):", 
                options=["All Documents"] + doc_list,
                default=["All Documents"]
            )
            
            # Number of chunks to retrieve
            top_k = st.slider("Number of chunks to retrieve:", 1, 10, 3)
            
            # Query input
            query = st.text_input("Your question:", placeholder="What information can I find in these documents?")
            
            # Search button
            search_button = st.button("Search", type="primary", key="search_button")
            
            # Create context for answers
            if search_button and query:
                with st.spinner("Searching for relevant information..."):
                    # Prepare filter if needed
                    filter_condition = None
                    if "All Documents" not in doc_filter and doc_filter:
                        filter_condition = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_metadata.filename",
                                    match=models.MatchAny(any=doc_filter)
                                )
                            ]
                        )
                    
                    # Perform semantic search
                    results = st.session_state.db_manager.semantic_search(
                        query,
                        top_k=top_k,
                        filter_condition=filter_condition
                    )
                    
                    if results:
                        # Display search results
                        st.success(f"Found {len(results)} relevant chunks.")
                        
                        # Prepare context for LLM
                        context = "\n\n".join([f"[Document: {r['metadata']['document']}, Chunk: {r['metadata']['chunk_num']}]\n{r['text']}" for r in results])
                        
                        # Add collapsible display for context used
                        with st.expander("View context chunks used for answer"):
                            for i, result in enumerate(results):
                                st.markdown(f"##### Chunk {i+1} (Score: {result['score']:.4f})")
                                st.markdown(f"**Document:** {result['metadata']['document']}, **Chunk:** {result['metadata']['chunk_num']}")
                                st.text(result['text'])
                                st.divider()
                        
                        # Prepare prompt for LLM
                        prompt = f"""
                        Answer the question based on the following context:
                        
                        CONTEXT:
                        {context}
                        
                        QUESTION:
                        {query}
                        """
                        
                        # Get response from LLM
                        with st.spinner("Generating answer..."):
                            response = abc_response(prompt)
                        
                        # Display answer
                        st.markdown("### Answer")
                        st.markdown(response)
                        
                        # Show raw context and scores for debugging
                        with st.expander("Debug Information"):
                            st.markdown("#### Raw Context and Relevance Scores")
                            
                            # Create a DataFrame for better display
                            debug_data = []
                            for r in results:
                                debug_data.append({
                                    "Document": r['metadata']['document'],
                                    "Chunk": r['metadata']['chunk_num'],
                                    "Score": r['score'],
                                    "Text Length": len(r['text'])
                                })
                            
                            st.dataframe(pd.DataFrame(debug_data))
                            
                            # Print full prompt
                            st.markdown("#### Full Prompt Sent to LLM")
                            st.text(prompt)
                    else:
                        st.warning("No relevant information found. Try reformulating your question.")
    
    # Tab 3: System Info
    with tab3:
        st.header("System Information")
        st.markdown("View information about the system and the documents in the database.")
        
        # Refresh button
        refresh_button = st.button("Refresh Information", type="primary")
        
        if refresh_button or not 'db_stats' in locals():
            with st.spinner("Fetching system information..."):
                db_stats = st.session_state.db_manager.get_collection_stats()
        
        # Display system info
        st.subheader("Vector Database Stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", db_stats.get("unique_documents", 0))
        with col2:
            st.metric("Total Chunks", db_stats.get("total_chunks", 0))
        with col3:
            st.metric("Vector Dimension", db_stats.get("vector_size", 0))
        
        # Document list
        if db_stats.get("document_list"):
            st.subheader("Document List")
            
            # Display documents as a dataframe
            doc_data = []
            for doc in db_stats.get("document_list", []):
                # Get metadata for the first chunk of this document
                metadata_list = st.session_state.db_manager.get_document_metadata(doc)
                metadata = metadata_list[0] if metadata_list else {}
                
                doc_data.append({
                    "Filename": doc,
                    "Title": metadata.get("title", "Unknown"),
                    "Pages": metadata.get("page_count", "Unknown"),
                    "Added on": metadata.get("extraction_time", "Unknown")
                })
            
            st.dataframe(pd.DataFrame(doc_data), use_container_width=True)
        else:
            st.info("No documents have been processed yet.")
        
        # System configuration
        st.subheader("System Configuration")
        st.json({
            "Qdrant Storage Path": LOCAL_QDRANT_PATH,
            "Embedding Model Path": LOCAL_MODEL_PATH,
            "Upload Folder": UPLOAD_FOLDER,
            "Chunk Size": CHUNK_SIZE,
            "Chunk Overlap": CHUNK_OVERLAP
        })

if __name__ == "__main__":
    main()
