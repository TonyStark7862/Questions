import os
import time
import json
import uuid
import logging
import tempfile
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from pathlib import Path
from datetime import datetime

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Qdrant
import qdrant_client
from qdrant_client.http import models as qdrant_models

# PDF processing
import fitz  # PyMuPDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTPage, LTTable, LTRect, LTLine
import re

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/backend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Class to handle PDF document processing with advanced techniques to maintain context.
    Extracts text while preserving structure, tables, and hierarchical organization.
    """
    
    def __init__(self):
        """Initialize the PDF processor."""
        logger.info("Initializing PDF processor")
    
    def extract_text_with_pymupdf(
        self, 
        file_path: str, 
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text from PDF using PyMuPDF while maintaining document structure.
        
        Args:
            file_path: Path to the PDF file
            progress_callback: Callback function to report progress
            
        Returns:
            List of document chunks with metadata
        """
        logger.info(f"Extracting text from {file_path} using PyMuPDF")
        
        # Open the PDF
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        total_pages = len(doc)
        
        # Extract document title or use filename
        doc_title = doc.metadata.get("title", "") or os.path.splitext(filename)[0]
        
        # Initialize structure for extraction results
        extracted_data = []
        
        # Parse TOC for section structure if available
        toc = doc.get_toc()
        toc_by_page = {}
        current_section = {"title": "Introduction", "level": 1}
        
        # Process TOC to get section information by page
        if toc:
            for item in toc:
                level, title, page = item
                toc_by_page[page] = {"title": title, "level": level}
        
        # Process each page
        for page_num in range(total_pages):
            if progress_callback:
                progress_callback(page_num, total_pages, f"Processing page {page_num+1}/{total_pages}")
                
            page = doc[page_num]
            
            # Update current section if we have TOC information
            if page_num+1 in toc_by_page:
                current_section = toc_by_page[page_num+1]
            
            # Get blocks which can help identify paragraphs, tables, etc.
            blocks = page.get_text("dict")["blocks"]
            
            # Extract tables separately to maintain their structure
            tables = self._extract_tables_from_page(page)
            
            # Extract text blocks, tracking their positions
            text_blocks = []
            for block in blocks:
                if block["type"] == 0:  # Text block
                    lines = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        lines.append(line_text)
                    
                    if lines:
                        text_blocks.append({
                            "text": "\n".join(lines),
                            "bbox": block["bbox"],
                            "type": "text"
                        })
            
            # Merge tables with text blocks and sort by position
            all_blocks = text_blocks + tables
            all_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))  # Sort by y, then x
            
            # Process text by combining adjacent blocks
            for block in all_blocks:
                # Skip empty blocks
                if not block.get("text", "").strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source": filename,
                    "page": page_num + 1,
                    "section": current_section["title"],
                    "section_level": current_section["level"],
                    "title": doc_title,
                    "type": block.get("type", "text"),
                    "bbox": block.get("bbox", [0, 0, 0, 0])
                }
                
                # Add to extracted data
                extracted_data.append({
                    "content": block.get("text", ""),
                    "metadata": metadata
                })
        
        doc.close()
        logger.info(f"Extracted {len(extracted_data)} text blocks from {filename}")
        return extracted_data
    
    def _extract_tables_from_page(self, page) -> List[Dict[str, Any]]:
        """
        Extract tables from a page using PyMuPDF.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of extracted tables with their text and bounding boxes
        """
        tables = []
        
        # Find table structures based on lines and rectangles
        rect_tolerance = 3.0  # Tolerance for rectangle detection
        
        # Get all lines and rectangles
        lines = page.get_drawings()
        
        # Find potential table regions
        rects = []
        for drawing in lines:
            if drawing["type"] == "r":  # rectangle
                rects.append(drawing["rect"])
        
        # Identify tables based on aligned rectangles
        if rects:
            # Group rectangles by alignment (potential tables)
            horizontal_aligned = {}
            for i, rect in enumerate(rects):
                y_key = round(rect[1] / rect_tolerance) * rect_tolerance
                if y_key not in horizontal_aligned:
                    horizontal_aligned[y_key] = []
                horizontal_aligned[y_key].append((i, rect))
            
            # Process potential table regions
            for y_key, aligned_rects in horizontal_aligned.items():
                if len(aligned_rects) >= 2:  # At least 2 cells horizontally
                    # Define the table region
                    left = min(r[0] for _, r in aligned_rects)
                    top = min(r[1] for _, r in aligned_rects)
                    right = max(r[2] for _, r in aligned_rects)
                    bottom = max(r[3] for _, r in aligned_rects)
                    
                    # Extract text within table region
                    table_text = page.get_text("text", clip=[left, top, right, bottom])
                    
                    if table_text.strip():
                        tables.append({
                            "text": "TABLE: " + table_text,
                            "bbox": [left, top, right, bottom],
                            "type": "table"
                        })
        
        return tables
    
    def process_document(
        self, 
        file_path: str, 
        progress_callbacks: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Any]:
        """
        Process a document by extracting text and maintaining structure.
        
        Args:
            file_path: Path to the PDF file
            progress_callbacks: Dict of callback functions for different processing stages
            
        Returns:
            Dict containing processing results and metadata
        """
        start_time = time.time()
        logger.info(f"Processing document: {file_path}")
        
        # Extract callback functions
        extraction_callback = progress_callbacks.get("extraction") if progress_callbacks else None
        
        # Process the PDF
        try:
            # Extract text with PyMuPDF while maintaining structure
            extracted_blocks = self.extract_text_with_pymupdf(file_path, extraction_callback)
            
            # Add document ID
            doc_id = str(uuid.uuid4())
            
            # Create document result
            document_result = {
                "id": doc_id,
                "name": os.path.basename(file_path),
                "path": file_path,
                "timestamp": datetime.now().isoformat(),
                "blocks": extracted_blocks,
                "status": "success",
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Successfully processed {file_path} in {document_result['processing_time']:.2f}s")
            return document_result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}", exc_info=True)
            return {
                "id": str(uuid.uuid4()),
                "name": os.path.basename(file_path),
                "path": file_path,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def chunk_document(
        self, 
        document: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Chunk the document while preserving context.
        
        Args:
            document: Processed document dict
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Chunking document {document['name']} with chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # Skip if document processing failed
        if document.get("status") != "success" or "blocks" not in document:
            logger.warning(f"Cannot chunk document {document['name']} due to processing error")
            return []
        
        result_chunks = []
        
        # Group blocks by section to maintain context
        sections = {}
        for block in document["blocks"]:
            section_key = f"{block['metadata']['section']}_{block['metadata']['page']}"
            if section_key not in sections:
                sections[section_key] = []
            sections[section_key].append(block)
        
        # Process each section to create contextual chunks
        for section_key, blocks in sections.items():
            section_text = ""
            metadata = None
            
            # Combine blocks within the same section
            for block in blocks:
                if metadata is None:
                    metadata = block["metadata"].copy()
                section_text += block["content"] + "\n\n"
            
            # Skip empty sections
            if not section_text.strip():
                continue
                
            # Create chunks that respect section boundaries
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            
            # Split the text
            texts = text_splitter.split_text(section_text)
            
            # Create LangChain documents with enhanced metadata
            for i, text in enumerate(texts):
                if metadata:
                    # Update metadata to indicate chunk number within section
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk"] = i
                    chunk_metadata["chunk_total"] = len(texts)
                    chunk_metadata["doc_id"] = document["id"]
                    
                    # Add context about section and document position
                    if i == 0:
                        chunk_metadata["position"] = "section_start"
                    elif i == len(texts) - 1:
                        chunk_metadata["position"] = "section_end"
                    else:
                        chunk_metadata["position"] = "section_middle"
                    
                    # Create Document object
                    chunk_doc = Document(page_content=text, metadata=chunk_metadata)
                    result_chunks.append(chunk_doc)
        
        logger.info(f"Created {len(result_chunks)} chunks for document {document['name']}")
        return result_chunks


class RAGManager:
    """
    Main class for managing the RAG (Retrieval Augmented Generation) system.
    Handles document processing, indexing, and querying.
    """
    
    def __init__(self, models_dir="./models", data_dir="./data"):
        """
        Initialize the RAG Manager with specified directories.
        
        Args:
            models_dir: Directory to store/load models
            data_dir: Directory to store document data and vector database
        """
        logger.info("Initializing RAG Manager")
        
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.docs_dir = self.data_dir / "documents"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.vectordb_path = self.data_dir / "qdrant_db"
        
        # Ensure directories exist
        for directory in [self.models_dir, self.data_dir, self.docs_dir, 
                          self.embeddings_dir, self.vectordb_path]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()
        
        # Initialize embeddings model (with cache to avoid redundant processing)
        self.embeddings = self._initialize_embeddings()
        
        # Initialize vector database
        self.vector_db = self._initialize_vectordb()
        
        # Document index
        self.document_index = self._load_document_index()
        
        logger.info("RAG Manager initialized successfully")
    
    def _initialize_embeddings(self) -> CacheBackedEmbeddings:
        """
        Initialize the embeddings model with caching.
        
        Returns:
            CacheBackedEmbeddings object
        """
        logger.info("Initializing embeddings model")
        
        try:
            # Use a local HuggingFace embeddings model
            hf_embed_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=str(self.models_dir)
            )
            
            # Use cached embeddings to avoid redundant processing
            fs = LocalFileStore(str(self.embeddings_dir))
            
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                hf_embed_model,
                fs,
                namespace=hf_embed_model.model_name
            )
            
            logger.info(f"Embeddings model initialized: sentence-transformers/all-MiniLM-L6-v2")
            return cached_embeddings
            
        except Exception as e:
            logger.error(f"Error initializing embeddings model: {str(e)}", exc_info=True)
            raise
    
    def _initialize_vectordb(self) -> Qdrant:
        """
        Initialize the vector database.
        
        Returns:
            Qdrant vector store
        """
        logger.info("Initializing Qdrant vector database")
        
        try:
            # Initialize Qdrant client with local storage
            client = qdrant_client.QdrantClient(
                path=str(self.vectordb_path)
            )
            
            # Create Qdrant vector store
            vector_db = Qdrant(
                client=client,
                collection_name="document_chunks",
                embeddings=self.embeddings
            )
            
            logger.info("Vector database initialized successfully")
            return vector_db
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}", exc_info=True)
            raise
    
    def _load_document_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the document index from disk.
        
        Returns:
            Dict mapping document IDs to document metadata
        """
        index_path = self.data_dir / "document_index.json"
        
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading document index: {str(e)}", exc_info=True)
                return {}
        else:
            logger.info("Document index does not exist, creating new")
            return {}
    
    def _save_document_index(self):
        """Save the document index to disk."""
        index_path = self.data_dir / "document_index.json"
        
        try:
            with open(index_path, "w") as f:
                json.dump(self.document_index, f, indent=2)
            logger.info("Document index saved successfully")
        except Exception as e:
            logger.error(f"Error saving document index: {str(e)}", exc_info=True)
    
    def process_pdf_file(
        self, 
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        summarize_chunks: bool = False,
        progress_callbacks: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF file and add it to the vector database.
        
        Args:
            file_path: Path to the PDF file
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            summarize_chunks: Whether to summarize chunks (requires abc_response to be defined externally)
            progress_callbacks: Dict of callback functions for different processing stages
            
        Returns:
            Dict containing processing results
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        # Extract callback functions
        extraction_callback = progress_callbacks.get("extraction") if progress_callbacks else None
        embedding_callback = progress_callbacks.get("embedding") if progress_callbacks else None
        
        # Process the document
        processed_doc = self.pdf_processor.process_document(
            file_path, 
            {"extraction": extraction_callback}
        )
        
        # Check if processing was successful
        if processed_doc.get("status") != "success":
            logger.error(f"Processing failed for {file_path}: {processed_doc.get('error', 'Unknown error')}")
            return processed_doc
        
        # Chunk the document
        chunks = self.pdf_processor.chunk_document(
            processed_doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if not chunks:
            processed_doc["status"] = "error"
            processed_doc["error"] = "No text chunks were extracted from the document"
            logger.error(f"No chunks extracted from {file_path}")
            return processed_doc
        
        # Summarize chunks if requested and abc_response function is available
        if summarize_chunks and 'abc_response' in globals():
            for i, chunk in enumerate(chunks):
                try:
                    if embedding_callback:
                        embedding_callback(i, len(chunks), f"Summarizing chunk {i+1}/{len(chunks)}")
                    
                    # Generate summary
                    summary_prompt = f"""
                    Please provide a short, concise summary of the following text.
                    Focus on the key information and main points.
                    
                    TEXT:
                    {chunk.page_content}
                    
                    SUMMARY:
                    """
                    
                    summary = abc_response(summary_prompt)
                    chunk.metadata["summary"] = summary
                    
                except Exception as e:
                    logger.warning(f"Error summarizing chunk {i}: {str(e)}")
                    # Continue even if summarization fails
        
        # Add chunks to vector database
        try:
            # Log the operation
            logger.info(f"Adding {len(chunks)} chunks to vector database for {file_path}")
            
            # Add chunks with progress tracking
            for i, chunk in enumerate(chunks):
                if embedding_callback:
                    embedding_callback(i, len(chunks), f"Embedding chunk {i+1}/{len(chunks)}")
                
                # Add document to vector store
                self.vector_db.add_documents([chunk])
            
            # Save document metadata to index
            doc_metadata = {
                "id": processed_doc["id"],
                "name": processed_doc["name"],
                "path": processed_doc["path"],
                "timestamp": processed_doc["timestamp"],
                "num_chunks": len(chunks),
                "status": "indexed"
            }
            
            self.document_index[processed_doc["id"]] = doc_metadata
            self._save_document_index()
            
            logger.info(f"Successfully indexed {file_path}")
            
            # Include metadata in result
            processed_doc["indexed"] = True
            processed_doc["num_chunks"] = len(chunks)
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error adding document to vector database: {str(e)}", exc_info=True)
            processed_doc["status"] = "error"
            processed_doc["error"] = f"Error during indexing: {str(e)}"
            return processed_doc
    
    def get_available_documents(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available documents.
        
        Returns:
            List of document metadata dicts
        """
        return list(self.document_index.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the system.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Deleting document with ID {doc_id}")
        
        if doc_id not in self.document_index:
            logger.warning(f"Document {doc_id} not found in index")
            return False
        
        try:
            # Get document metadata
            doc_metadata = self.document_index[doc_id]
            
            # Delete from vector store by filtering out chunks with this doc_id
            client = self.vector_db._client
            client.delete(
                collection_name=self.vector_db._collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="metadata.doc_id",
                                match=qdrant_models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
            
            # Remove from document index
            del self.document_index[doc_id]
            self._save_document_index()
            
            logger.info(f"Successfully deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}", exc_info=True)
            return False
    
    def query(
        self, 
        query: str, 
        file_ids: Optional[List[str]] = None,
        num_results: int = 5,
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query: User query
            file_ids: List of document IDs to search (None for all documents)
            num_results: Number of results to retrieve
            use_reranking: Whether to use reranking for better results
            
        Returns:
            Dict containing answer and sources
        """
        logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        try:
            # Create document filter if specific files are requested
            filter_condition = None
            if file_ids:
                filter_condition = {"filter": {"metadata": {"doc_id": {"$in": file_ids}}}}
            
            # Get search results with enhanced filtering
            # First, implement a much smarter retrieval strategy
            retriever = self._get_advanced_retriever(filter_condition)
            
            # Retrieve contexts based on the query
            contexts = retriever.get_relevant_documents(query)
            
            if not contexts:
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            # Format contexts for use in prompt
            formatted_contexts = []
            sources = []
            
            for i, doc in enumerate(contexts[:num_results]):
                # Add to formatted contexts
                formatted_contexts.append(f"Context {i+1}:\n{doc.page_content}")
                
                # Add to sources
                source = {
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "file": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "section": doc.metadata.get("section", "Unknown"),
                }
                sources.append(source)
            
            # Generate an answer
            # For the abc_response function:
            # If abc_response function exists, use it
            # If not, we'll generate a placeholder response
            
            contexts_text = "\n\n".join(formatted_contexts)
            
            prompt = f"""
            You are an intelligent assistant answering questions based on provided document contexts.
            Use only the information in the contexts to answer the question.
            If the contexts don't contain the information needed, say "I don't have enough information to answer this question based on the documents."
            Be concise, accurate, and provide specific details from the contexts when relevant.
            
            CONTEXTS:
            {contexts_text}
            
            QUESTION: {query}
            
            ANSWER:
            """
            
            # Use the abc_response function if it exists, otherwise provide a placeholder
            if 'abc_response' in globals():
                answer = abc_response(prompt)
            else:
                logger.warning("abc_response function not defined, using placeholder answer")
                answer = "To generate actual responses, please ensure the 'abc_response' function is defined for LLM integration."
            
            result = {
                "answer": answer,
                "sources": sources,
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Query processed in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "processing_time": time.time() - start_time
            }
    
    def _get_advanced_retriever(self, filter_condition=None):
        """
        Create an advanced retriever with multiple enhancement techniques.
        
        Args:
            filter_condition: Optional filter for retrieval
            
        Returns:
            Enhanced document retriever
        """
        # Base retriever with filtered search
        base_retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
                **({} if filter_condition is None else filter_condition)
            }
        )
        
        # Add embedding-based reranking
        embeddings_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.7)
        
        # Create retrieval pipeline with compression
        pipeline_compressor = DocumentCompressorPipeline(transformers=[embeddings_filter])
        
        # Create compressed retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
