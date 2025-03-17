# Add these modifications to your PDF processing section in the Streamlit app

# In your main.py file, modify the Tab 1 processing section:

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
                    
                    # Extract text with better error handling and timeout
                    try:
                        # This would be the modified extract_text_from_pdf function
                        # You would need to modify it to accept a callback
                        # For demonstration, we'll simulate the steps
                        add_log("Opening PDF document")
                        extraction_status.markdown("⏳ **Step 1/4:** Opening PDF document...")
                        time.sleep(1)  # Simulate processing time
                        
                        add_log("Reading document structure")
                        extraction_status.markdown("⏳ **Step 1/4:** Reading document structure...")
                        progress_bar.progress(15)
                        time.sleep(1)  # Simulate processing time
                        
                        add_log("Extracting text from pages")
                        extraction_status.markdown("⏳ **Step 1/4:** Extracting text from pages...")
                        progress_bar.progress(20)
                        # Here you would call your actual text extraction function
                        # text = PDFProcessor.extract_text_from_pdf(tmp_path)
                        
                        # For demo purposes
                        add_log("Text extraction completed successfully")
                        extraction_status.markdown("✅ **Step 1/4:** Text extracted successfully")
                        progress_bar.progress(30)
                        
                        # Process metadata
                        add_log("Extracting document metadata")
                        # metadata = PDFProcessor.extract_metadata_from_pdf(tmp_path)
                        add_log("Metadata extraction completed")
                        progress_bar.progress(35)
                        
                        # Chunking
                        chunking_status.markdown("⏳ **Step 2/4:** Chunking document text...")
                        add_log("Starting text chunking process")
                        
                        # Here you would call your actual chunking function
                        # chunks = PDFProcessor.intelligent_chunking(text)
                        
                        # For demo purposes
                        for chunk_progress in range(5):
                            time.sleep(0.5)  # Simulate chunking steps
                            add_log(f"Processed chunk section {chunk_progress+1}/5")
                            chunking_status.markdown(f"⏳ **Step 2/4:** Chunking document (section {chunk_progress+1}/5)...")
                            progress_bar.progress(35 + chunk_progress * 3)
                        
                        add_log("Document chunking completed")
                        chunking_status.markdown("✅ **Step 2/4:** Document chunked successfully")
                        progress_bar.progress(50)
                        
                        # Embedding
                        embedding_status.markdown("⏳ **Step 3/4:** Generating embeddings...")
                        add_log("Loading embedding model")
                        
                        # For demo purposes
                        for emb_progress in range(10):
                            time.sleep(0.3)  # Simulate embedding generation
                            add_log(f"Generated embeddings for chunk batch {emb_progress+1}/10")
                            embedding_status.markdown(f"⏳ **Step 3/4:** Generating embeddings (batch {emb_progress+1}/10)...")
                            progress_bar.progress(50 + emb_progress * 3)
                        
                        add_log("Embedding generation completed")
                        embedding_status.markdown("✅ **Step 3/4:** Embeddings generated successfully")
                        progress_bar.progress(80)
                        
                        # Database upload
                        db_status.markdown("⏳ **Step 4/4:** Storing in vector database...")
                        add_log("Preparing database points")
                        
                        # For demo purposes
                        for db_progress in range(5):
                            time.sleep(0.4)  # Simulate database operations
                            add_log(f"Uploaded batch {db_progress+1}/5 to vector database")
                            db_status.markdown(f"⏳ **Step 4/4:** Storing in database (batch {db_progress+1}/5)...")
                            progress_bar.progress(80 + db_progress * 4)
                        
                        add_log("Database storage completed")
                        db_status.markdown("✅ **Step 4/4:** Storage in vector database completed")
                        progress_bar.progress(100)
                        
                        # Success message
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                        add_log(f"Complete processing pipeline finished successfully for {uploaded_file.name}")
                        st.session_state.documents_processed += 1
                        
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
