# frontend.py
import streamlit as st
import time
import os
from typing import Optional, Callable
from backend import RAGManager, PDFProcessor

# Configure page settings
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>div>input {
            padding: 0.5rem;
            border-radius: 5px;
        }
        .stProgress .st-bo {
            background-color: #4CAF50;
        }
        .answer-box {
            padding: 1.5rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .source-card {
            padding: 1rem;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

def initialize_rag_manager():
    """Initialize RAG Manager in session state"""
    if 'rag_manager' not in st.session_state:
        st.session_state.rag_manager = RAGManager()
        st.session_state.processed_docs = st.session_state.rag_manager.get_available_documents()
        st.session_state.active_docs = [doc['id'] for doc in st.session_state.processed_docs]

def progress_callback(current: int, total: int, message: str):
    """Handle progress updates for file processing"""
    progress_bar.progress(current / total, text=message)

# Initialize RAG Manager
initialize_rag_manager()

# Sidebar Navigation
with st.sidebar:
    st.title("PDF RAG System")
    nav_option = st.radio(
        "Navigation",
        ["📤 Upload Documents", "📂 Manage Documents", "❓ Ask Questions"],
        label_visibility="collapsed"
    )

# File Upload Section
if nav_option == "📤 Upload Documents":
    st.header("Upload and Process PDF Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Select one or more PDF documents to process"
    )
    
    if uploaded_files:
        with st.expander("Processing Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("Chunk Size", 500, 2000, 1000, help="Size of text chunks in characters")
            with col2:
                chunk_overlap = st.number_input("Chunk Overlap", 100, 500, 200, help="Overlap between chunks")
        
        process_btn = st.button("Process Documents")
        
        if process_btn:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process the document
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    global progress_bar
                    progress_bar = st.progress(0, text="Starting processing...")
                    
                    try:
                        result = st.session_state.rag_manager.process_pdf_file(
                            tmp_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            progress_callbacks={
                                "extraction": progress_callback,
                                "embedding": progress_callback
                            }
                        )
                        
                        if result['status'] == 'success':
                            st.success(f"Successfully processed {uploaded_file.name}")
                            st.session_state.processed_docs = st.session_state.rag_manager.get_available_documents()
                            st.session_state.active_docs = [doc['id'] for doc in st.session_state.processed_docs]
                        else:
                            st.error(f"Failed to process {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
                        progress_bar.empty()

# Document Management Section
elif nav_option == "📂 Manage Documents":
    st.header("Manage Processed Documents")
    
    if not st.session_state.processed_docs:
        st.info("No documents have been processed yet.")
    else:
        # Document selection and management
        selected_docs = st.multiselect(
            "Select documents to manage",
            options=[doc['name'] for doc in st.session_state.processed_docs],
            format_func=lambda x: x
        )
        
        # Show document details
        with st.expander("Document Details", expanded=True):
            for doc in st.session_state.processed_docs:
                if doc['name'] in selected_docs:
                    cols = st.columns([2, 3, 1])
                    cols[0].subheader(doc['name'])
                    cols[1].markdown(f"""
                        **ID:** {doc['id']}  
                        **Processed:** {doc['timestamp']}  
                        **Chunks:** {doc['num_chunks']}
                    """)
                    if cols[2].button("Delete", key=f"del_{doc['id']}"):
                        if st.session_state.rag_manager.delete_document(doc['id']):
                            st.success(f"Deleted {doc['name']}")
                            st.session_state.processed_docs = st.session_state.rag_manager.get_available_documents()
                            st.rerun()

# Query Section
elif nav_option == "❓ Ask Questions":
    st.header("Ask Questions About Your Documents")
    
    if not st.session_state.processed_docs:
        st.warning("Please process some documents first before asking questions.")
    else:
        # Query input and options
        query = st.text_input("Enter your question:", placeholder="Ask me anything about the documents...")
        
        with st.expander("Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                num_results = st.slider("Number of results", 1, 10, 5)
            with col2:
                selected_docs = st.multiselect(
                    "Filter by documents:",
                    options=[doc['id'] for doc in st.session_state.processed_docs],
                    default=st.session_state.active_docs,
                    format_func=lambda x: next(d['name'] for d in st.session_state.processed_docs if d['id'] == x)
        
        if query:
            with st.spinner("Analyzing documents..."):
                start_time = time.time()
                try:
                    response = st.session_state.rag_manager.query(
                        query,
                        file_ids=selected_docs,
                        num_results=num_results
                    )
                    
                    # Display results
                    with st.container():
                        # Answer section
                        st.markdown(f"<div class='answer-box'><h3>Answer</h3>{response['answer']}</div>", unsafe_allow_html=True)
                        
                        # Sources section
                        st.subheader("Sources")
                        for i, source in enumerate(response['sources']):
                            with st.expander(f"Source {i+1} - {source['file']} (Page {source['page']})"):
                                st.markdown(f"""
                                    **Section:** {source['section']}  
                                    **Content:**  
                                    {source['text']}
                                """)
                    
                    st.caption(f"Query processed in {response['processing_time']:.2f} seconds")
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>PDF RAG System v1.0</div>", unsafe_allow_html=True)
