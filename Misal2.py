# At the beginning of your app, initialize these session state variables
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = 0
    st.session_state.processed_document_list = []

# After successful document processing, update both variables
if success:
    st.success(message)
    st.session_state.documents_processed += 1
    st.session_state.processed_document_list.append(metadata.get('filename'))
