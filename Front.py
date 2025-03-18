import os
import time
import streamlit as st
import tempfile
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import base64
from datetime import datetime

# Import the backend module
from backend import RAGManager

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
for directory in ["./data", "./logs"]:
    os.makedirs(directory, exist_ok=True)


# Initialize session state variables
def init_session_state():
    """Initialize session state variables."""
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = RAGManager()
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
        
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
        
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = {}
        
    if "available_documents" not in st.session_state:
        st.session_state.available_documents = []
        
    if "selected_documents" not in st.session_state:
        st.session_state.selected_documents = []
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# Custom CSS
def load_css():
    """Load custom CSS."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #43A047;
    }
    
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #FFA000;
    }
    
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #E53935;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.user {
        background-color: #E3F2FD;
        border: 1px solid #90CAF9;
        margin-left: 2rem;
    }
    
    .chat-message.bot {
        background-color: #F5F5F5;
        border: 1px solid #E0E0E0;
        margin-right: 2rem;
    }
    
    .chat-message .message-content {
        margin-bottom: 0.5rem;
    }
    
    .chat-message .message-metadata {
        font-size: 0.8rem;
        color: #757575;
        align-self: flex-end;
    }
    
    .source-accordion {
        margin-top: 0.5rem;
        background-color: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    
    .source-item {
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background-color: #F5F5F5;
        border-radius: 0.3rem;
    }
    
    .metadata-pill {
        background-color: #E1F5FE;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        margin-right: 0.3rem;
        color: #0277BD;
    }
    
    .relevance-pill {
        background-color: #E8F5E9;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        margin-right: 0.3rem;
        color: #2E7D32;
    }
    
    .status-pill {
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .status-pill.success {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    
    .status-pill.processing {
        background-color: #FFF9C4;
        color: #F57F17;
    }
    
    .status-pill.error {
        background-color: #FFCDD2;
        color: #C62828;
    }
    
    .processing-step {
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    
    .progress-container {
        flex-grow: 1;
        margin-left: 1rem;
    }
    
    .document-card {
        border: 1px solid #E0E0E0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #FAFAFA;
    }
    
    .document-card:hover {
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .document-card .document-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1976D2;
        margin-bottom: 0.5rem;
    }
    
    .document-card .document-metadata {
        font-size: 0.8rem;
        color: #757575;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


# Custom components
def header():
    """Display the header."""
    st.markdown('<div class="main-header">📚 PDF RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        "A powerful Retrieval-Augmented Generation system for querying PDF documents. "
        "Upload your PDFs and ask questions to get accurate answers based on the document content."
    )
    st.divider()


def subheader(text):
    """Display a subheader."""
    st.markdown(f'<div class="sub-header">{text}</div>', unsafe_allow_html=True)


def info_box(text):
    """Display an info box."""
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def success_box(text):
    """Display a success box."""
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)


def warning_box(text):
    """Display a warning box."""
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


def error_box(text):
    """Display an error box."""
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)


def get_document_details():
    """Get available document details from the system."""
    docs = st.session_state.rag_manager.get_available_documents()
    st.session_state.available_documents = docs
    return docs


# File processing functions
def process_uploaded_files():
    """Process the uploaded files."""
    if not st.session_state.uploaded_files:
        return
    
    for file in st.session_state.uploaded_files:
        # Check if file is already processed
        if any(p["name"] == file.name for p in st.session_state.processed_files):
            continue
            
        # Add file to processing queue
        st.session_state.processed_files.append({
            "name": file.name,
            "status": "processing",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Initialize processing status
        file_key = file.name
        st.session_state.processing_status[file_key] = {
            "extraction": {"current": 0, "total": 0},
            "embedding": {"current": 0, "total": 0},
            "storage": {"current": 0, "total": 0}
        }
        
        # Define progress callback functions
        def extraction_callback(current, total):
            st.session_state.processing_status[file_key]["extraction"] = {
                "current": current,
                "total": total
            }
            
        def embedding_callback(current, total):
            st.session_state.processing_status[file_key]["embedding"] = {
                "current": current,
                "total": total
            }
            
        def storage_callback(current, total):
            st.session_state.processing_status[file_key]["storage"] = {
                "current": current,
                "total": total
            }
            
        # Process the file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
            
        try:
            # Process the file
            success = st.session_state.rag_manager.process_pdf_file(
                tmp_path,
                progress_callbacks={
                    "extraction": extraction_callback,
                    "embedding": embedding_callback,
                    "storage": storage_callback
                }
            )
            
            # Update processing status
            for idx, pf in enumerate(st.session_state.processed_files):
                if pf["name"] == file.name:
                    st.session_state.processed_files[idx]["status"] = "success" if success else "error"
                    break
                    
        except Exception as e:
            # Update processing status on error
            for idx, pf in enumerate(st.session_state.processed_files):
                if pf["name"] == file.name:
                    st.session_state.processed_files[idx]["status"] = "error"
                    st.session_state.processed_files[idx]["error"] = str(e)
                    break
                    
            logger.error(f"Error processing file {file.name}: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.error(f"Error removing temp file: {e}")
    
    # Refresh available documents
    get_document_details()
    
    # Clear uploaded files
    st.session_state.uploaded_files = []


def upload_section():
    """Display the file upload section."""
    subheader("Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze and query."
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        # Process button
        if st.button("Process Documents", type="primary"):
            process_uploaded_files()
    
    # Display processing status
    if st.session_state.processed_files:
        subheader("Processing Status")
        
        for doc in st.session_state.processed_files:
            status_class = "success" if doc["status"] == "success" else (
                "processing" if doc["status"] == "processing" else "error"
            )
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(
                        f'<div>{doc["name"]} <span class="status-pill {status_class}">{doc["status"]}</span></div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(f'<div>{doc["timestamp"]}</div>', unsafe_allow_html=True)
                
                # Display progress bars for processing file
                if doc["status"] == "processing":
                    file_key = doc["name"]
                    if file_key in st.session_state.processing_status:
                        status = st.session_state.processing_status[file_key]
                        
                        # Text extraction progress
                        extraction = status["extraction"]
                        if extraction["total"] > 0:
                            st.markdown('<div class="processing-step">Text extraction:</div>', unsafe_allow_html=True)
                            progress = min(1.0, extraction["current"] / max(1, extraction["total"]))
                            st.progress(progress)
                            
                        # Embedding progress
                        embedding = status["embedding"]
                        if embedding["total"] > 0:
                            st.markdown('<div class="processing-step">Generating embeddings:</div>', unsafe_allow_html=True)
                            progress = min(1.0, embedding["current"] / max(1, embedding["total"]))
                            st.progress(progress)
                            
                        # Storage progress
                        storage = status["storage"]
                        if storage["total"] > 0:
                            st.markdown('<div class="processing-step">Storing in database:</div>', unsafe_allow_html=True)
                            progress = min(1.0, storage["current"] / max(1, storage["total"]))
                            st.progress(progress)


def document_management_section():
    """Display the document management section."""
    subheader("Document Management")
    
    # Get available documents
    docs = get_document_details()
    
    if not docs:
        info_box("No documents available. Upload and process documents to get started.")
        return
        
    # Display documents
    st.write(f"Found {len(docs)} documents in the system.")
    
    # Select documents for querying
    selected = st.multiselect(
        "Select documents to query",
        options=[doc["name"] for doc in docs],
        default=[doc["name"] for doc in docs],
        help="Select which documents to include in your query."
    )
    
    # Store selected document IDs in session
    st.session_state.selected_documents = [
        doc["id"] for doc in docs if doc["name"] in selected
    ]
    
    # Delete document button
    if st.button("Delete Selected Documents", type="secondary"):
        for doc in docs:
            if doc["name"] in selected:
                success = st.session_state.rag_manager.delete_document(doc["id"])
                if success:
                    st.success(f"Deleted document: {doc['name']}")
                else:
                    st.error(f"Failed to delete document: {doc['name']}")
        
        # Refresh available documents
        get_document_details()


def chat_section():
    """Display the chat interface."""
    subheader("Ask Questions")
    
    # Check if documents are available
    if not st.session_state.available_documents:
        warning_box("No documents available. Please upload and process some documents first.")
        return
        
    # Check if documents are selected
    if not st.session_state.selected_documents:
        warning_box("No documents selected. Please select at least one document to query.")
        return
        
    # Query input
    query = st.text_input(
        "Your question",
        key="query_input",
        placeholder="What would you like to know about the documents?",
        help="Ask a question related to the content of the selected documents."
    )
    
    # Submit button
    if st.button("Ask", type="primary") and query:
        with st.spinner("Searching for answers..."):
            # Query the RAG system
            response = st.session_state.rag_manager.query(
                query,
                file_ids=st.session_state.selected_documents
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "type": "user",
                "content": query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.session_state.chat_history.append({
                "type": "bot",
                "content": response["answer"],
                "sources": response["sources"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        
        # Loop through chat history in reverse order (newest first)
        for message in reversed(st.session_state.chat_history):
            if message["type"] == "user":
                # User message
                st.markdown(
                    f'<div class="chat-message user">'
                    f'<div class="message-content">{message["content"]}</div>'
                    f'<div class="message-metadata">{message["timestamp"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                # Bot message
                st.markdown(
                    f'<div class="chat-message bot">'
                    f'<div class="message-content">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(
                                f'<div class="source-item">'
                                f'<div><strong>Source {i+1}:</strong> {source["text"]}</div>'
                                f'<div>'
                                f'<span class="metadata-pill">File: {source["file"]}</span>'
                                f'<span class="metadata-pill">Page: {source["page"]}</span>'
                                f'<span class="metadata-pill">Section: {source["section"]}</span>'
                                f'<span class="relevance-pill">Relevance: {source["score"]}</span>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                
                st.markdown(
                    f'<div class="message-metadata">{message["timestamp"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )


def main():
    """Main function to run the Streamlit app."""
    # Setup page configuration
    st.set_page_config(
        page_title="PDF RAG Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Load custom CSS
    load_css()
    
    # Sidebar
    with st.sidebar:
        st.title("PDF RAG Assistant")
        st.divider()
        
        st.subheader("Navigation")
        page = st.radio(
            "Go to",
            options=["Upload Documents", "Chat with Documents", "Settings"],
            help="Navigate to different sections of the application."
        )
        
        st.divider()
        
        # About section
        st.subheader("About")
        st.markdown(
            "This application allows you to upload PDF documents, process them using "
            "advanced techniques, and query their content using natural language."
        )
        st.markdown(
            "The system uses semantic search to retrieve relevant information "
            "from your documents and generates accurate responses based on the content."
        )
        
        st.divider()
        
        # Document statistics
        st.subheader("Document Statistics")
        docs = st.session_state.available_documents
        st.write(f"Total Documents: {len(docs)}")
    
    # Main content
    header()
    
    if page == "Upload Documents":
        upload_section()
        document_management_section()
    elif page == "Chat with Documents":
        chat_section()
    elif page == "Settings":
        st.subheader("Settings")
        st.write("This section is for future expansion.")
        
        # Debug option (for development)
        if st.checkbox("Debug Mode"):
            st.subheader("Debug Information")
            st.write("Session State:")
            st.json({
                "available_documents": st.session_state.available_documents,
                "selected_documents": st.session_state.selected_documents,
                "processed_files": st.session_state.processed_files
            })


if __name__ == "__main__":
    main()
