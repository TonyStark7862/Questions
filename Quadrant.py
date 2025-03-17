import streamlit as st
import numpy as np
import os
import time
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Configuration for local paths - CHANGE THESE TO YOUR PREFERRED LOCATIONS
LOCAL_QDRANT_PATH = "./qdrant_qna_data"
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"

# Ensure directories exist
os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

# Sample paragraphs for QnA system - EXAMPLE DATA
SAMPLE_PARAGRAPHS = [
    {
        "title": "Python Programming Language",
        "content": "Python is a high-level, interpreted programming language known for its readability and simplicity. Created by Guido van Rossum and released in 1991, Python emphasizes code readability with its notable use of whitespace indentation. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python has a comprehensive standard library and a large ecosystem of third-party packages, making it suitable for various applications such as web development, data analysis, artificial intelligence, scientific computing, and automation."
    },
    {
        "title": "Vector Databases",
        "content": "Vector databases are specialized database systems designed to store, manage, and search vector embeddings efficiently. These embeddings represent high-dimensional data points that capture semantic meaning. Unlike traditional databases that excel at exact matching, vector databases perform similarity searches using distance metrics like cosine similarity or Euclidean distance. They employ approximate nearest neighbor (ANN) algorithms for fast retrieval at scale. Popular vector databases include Qdrant, Pinecone, Milvus, and Weaviate. They're particularly useful for applications involving natural language processing, image recognition, recommendation systems, and other machine learning applications."
    },
    {
        "title": "Natural Language Processing",
        "content": "Natural Language Processing (NLP) is a field of artificial intelligence focused on enabling computers to understand, interpret, and generate human language. It bridges the gap between human communication and computer understanding. Key tasks in NLP include sentiment analysis, named entity recognition, machine translation, question answering, and text summarization. Modern NLP systems typically use deep learning approaches, particularly transformer-based models like BERT, GPT, and T5. These models are trained on vast amounts of text data and can capture complex linguistic patterns and semantic relationships."
    },
    {
        "title": "Climate Change",
        "content": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases. The effects of climate change include rising global temperatures, melting ice caps and glaciers, rising sea levels, more frequent and severe weather events, and disruptions to ecosystems. Addressing climate change requires both mitigation strategies to reduce greenhouse gas emissions and adaptation measures to adjust to the changes that are already occurring."
    },
    {
        "title": "Artificial Intelligence",
        "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines programmed to think and learn like humans. AI encompasses various subfields, including machine learning, deep learning, natural language processing, computer vision, and robotics. AI systems can be categorized as narrow (designed for specific tasks) or general (capable of performing any intellectual task that a human can do). The development of AI raises important ethical considerations, including privacy concerns, algorithmic bias, job displacement, and questions about accountability and control. Despite these challenges, AI continues to advance rapidly and has the potential to transform numerous sectors, from healthcare and transportation to education and entertainment."
    },
    {
        "title": "Renewable Energy",
        "content": "Renewable energy comes from naturally replenishing sources that are virtually inexhaustible, such as sunlight, wind, rain, tides, waves, and geothermal heat. Unlike fossil fuels, renewable energy sources don't deplete over time and typically produce little to no greenhouse gas emissions. Solar energy harnesses the power of the sun through photovoltaic panels or concentrated solar power systems. Wind energy converts kinetic energy from wind into electricity using turbines. Hydropower captures energy from flowing water, while geothermal energy utilizes heat from within the Earth. Biomass energy is derived from organic materials. The transition to renewable energy is crucial for addressing climate change and building a sustainable future."
    },
    {
        "title": "Quantum Computing",
        "content": "Quantum computing leverages the principles of quantum mechanics to process information in fundamentally different ways than classical computers. While classical computers use bits (0s and 1s), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously through superposition. Qubits can also be entangled, allowing changes to one qubit to instantaneously affect another, regardless of distance. These properties enable quantum computers to solve certain complex problems exponentially faster than classical computers. Potential applications include cryptography, optimization, drug discovery, materials science, and artificial intelligence. However, quantum computers face challenges such as maintaining quantum coherence and error correction."
    },
    {
        "title": "Blockchain Technology",
        "content": "Blockchain is a distributed ledger technology that records transactions across multiple computers in a way that ensures data cannot be altered retroactively. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data, creating a chain of blocks. The decentralized nature of blockchain eliminates the need for a central authority, providing transparency, security, and immutability. Initially developed for Bitcoin, blockchain technology has expanded to various applications beyond cryptocurrencies, including supply chain management, voting systems, identity verification, smart contracts, and decentralized finance (DeFi). Its potential to transform traditional business models and processes continues to drive innovation across industries."
    },
    {
        "title": "Space Exploration",
        "content": "Space exploration involves the discovery and exploration of outer space through advancing technologies and space programs. Since the first satellite, Sputnik 1, launched in 1957, humans have made remarkable achievements in space, including landing on the Moon, deploying space telescopes, and establishing the International Space Station. Current initiatives include Mars rovers and plans for human missions to Mars, asteroid mining research, and the search for exoplanets that could support life. Private companies like SpaceX, Blue Origin, and Virgin Galactic are increasingly contributing to space exploration, making space more accessible. Space exploration drives scientific discovery, technological innovation, and addresses fundamental questions about our place in the universe."
    },
    {
        "title": "Internet of Things",
        "content": "The Internet of Things (IoT) refers to the network of physical objects embedded with sensors, software, and connectivity that enables them to connect and exchange data over the internet. These connected devices range from ordinary household objects to sophisticated industrial tools. IoT enables seamless communication between people, processes, and things. Smart homes with connected thermostats, lighting systems, and appliances represent consumer IoT applications. In industry, IoT facilitates predictive maintenance, asset tracking, and optimization of manufacturing processes. While IoT offers convenience and efficiency, it also raises concerns about privacy, security, and the potential for systems to become dependent on functioning networks."
    }
]

@st.cache_resource
def load_sentence_transformer():
    """Initialize the embedding model from local path or download if not present."""
    try:
        st.info(f"Loading embedding model from {LOCAL_MODEL_PATH}...")
        model = SentenceTransformer(LOCAL_MODEL_PATH)
        st.success("✅ Model loaded from local path")
    except Exception as e:
        st.warning(f"Model not found locally or error loading: {e}. Downloading model (this may take a moment)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Save the model for future use
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model.save(LOCAL_MODEL_PATH)
        st.success("✅ Model downloaded and saved locally")
    return model

@st.cache_resource
def setup_qdrant_client():
    """Setup Qdrant client with local persistence."""
    try:
        client = QdrantClient(path=LOCAL_QDRANT_PATH)
        st.success("✅ Connected to local Qdrant database")
        return client
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return None

def create_collection(client, collection_name, vector_size=384):
    """Create a new collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            st.success(f"✅ Collection '{collection_name}' created")
        else:
            st.info(f"Collection '{collection_name}' already exists")
    except Exception as e:
        st.error(f"Error creating collection: {e}")

def add_paragraphs(client, collection_name, model, paragraphs):
    """Add paragraphs to the collection."""
    try:
        # Extract paragraph contents
        contents = [p["content"] for p in paragraphs]
        
        # Generate embeddings
        with st.spinner("Generating embeddings for paragraphs..."):
            embeddings = model.encode(contents)
        
        # Get collection info to check if points already exist
        collection_info = client.get_collection(collection_name)
        existing_count = collection_info.points_count
        
        # Skip if paragraphs are already added
        if existing_count >= len(paragraphs):
            st.info(f"Paragraphs already added to collection (found {existing_count} points)")
            return
        
        # Generate IDs starting after existing points
        starting_id = existing_count
        
        # Prepare points for upload
        points = [
            models.PointStruct(
                id=starting_id + idx,
                vector=embedding.tolist(),
                payload={
                    "title": paragraph["title"],
                    "content": paragraph["content"]
                }
            )
            for idx, (embedding, paragraph) in enumerate(zip(embeddings, paragraphs))
        ]
        
        # Upload to collection
        with st.spinner(f"Adding {len(points)} paragraphs to collection..."):
            client.upsert(
                collection_name=collection_name,
                points=points
            )
        
        st.success(f"✅ Added {len(points)} paragraphs to collection")
    except Exception as e:
        st.error(f"Error adding paragraphs: {e}")

def search_paragraphs(client, collection_name, model, query_text, limit=3):
    """Search for paragraphs similar to the query using the non-deprecated query_points method."""
    try:
        # Generate embedding for query
        query_embedding = model.encode([query_text])[0]
        
        # Search in collection using query_points instead of search
        search_results = client.query_points(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        return search_results
    except Exception as e:
        st.error(f"Error searching paragraphs: {e}")
        return []

def main():
    st.set_page_config(
        page_title="Qdrant QnA Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Semantic Question Answering with Qdrant")
    st.markdown("""
    This demo showcases a simple Question Answering system using Qdrant vector database.
    Ask questions about various topics and the system will find the most relevant paragraphs.
    
    ### Features:
    - Uses local storage for both the database and embedding model
    - Semantic search finds information based on meaning, not just keywords
    - Simple and efficient vector similarity search
    """)
    
    # Initialize model and client
    model = load_sentence_transformer()
    client = setup_qdrant_client()
    
    if not client:
        st.error("Failed to initialize Qdrant client. Please check the configuration.")
        return
    
    # Setup collection
    collection_name = "paragraphs"
    create_collection(
        client, 
        collection_name,
        vector_size=model.get_sentence_embedding_dimension()
    )
    
    # Add sample paragraphs
    add_paragraphs(client, collection_name, model, SAMPLE_PARAGRAPHS)
    
    st.divider()
    
    # Sidebar for settings and info
    with st.sidebar:
        st.subheader("About")
        st.markdown("""
        This application demonstrates how vector databases can be used for semantic search
        and question answering. It uses:
        
        - **Sentence Transformers** to convert text to vector embeddings
        - **Qdrant** vector database to store and search these embeddings
        - **Streamlit** for the user interface
        
        All data is stored locally for persistence.
        """)
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        num_results = st.slider("Number of results to show", 1, 5, 3)
        
        st.divider()
        
        # Display collection info
        st.subheader("Collection Info")
        if st.button("Refresh Collection Info"):
            try:
                collection_info = client.get_collection(collection_name)
                st.write(f"Collection Name: {collection_name}")
                st.write(f"Number of Documents: {collection_info.points_count}")
                st.write(f"Vector Size: {collection_info.config.params.vectors.size}")
            except Exception as e:
                st.error(f"Error fetching collection info: {e}")
    
    # Main area - QnA interface
    st.header("Ask a Question")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Question Answering", "View Knowledge Base"])
    
    with tab1:
        # Query input
        query = st.text_input(
            "Your Question",
            placeholder="What is artificial intelligence?",
            help="Ask any question related to the topics in the knowledge base"
        )
        
        # Search button
        search_col1, search_col2 = st.columns([1, 5])
        with search_col1:
            search_button = st.button("Search", type="primary")
        with search_col2:
            st.markdown("")  # Empty space
        
        # Execute search when button is clicked
        if search_button and query:
            with st.spinner("Searching for answers..."):
                # Add a small delay to make the spinner visible
                time.sleep(0.5)
                results = search_paragraphs(client, collection_name, model, query, limit=num_results)
            
            # Display results
            if results:
                st.success(f"Found {len(results)} relevant answers")
                
                for i, result in enumerate(results):
                    with st.container():
                        st.subheader(result.payload["title"])
                        st.markdown(result.payload["content"])
                        st.caption(f"Relevance score: {result.score:.4f}")
                        st.divider()
            else:
                st.warning("No relevant information found. Try rephrasing your question.")
        
        # Sample questions section
        st.subheader("Sample Questions to Try")
        
        sample_questions = [
            "What is Python used for?",
            "How do vector databases work?",
            "What are the effects of climate change?",
            "What is quantum computing?",
            "How is AI being used today?",
            "What are renewable energy sources?",
            "Explain blockchain technology",
            "What is the current state of space exploration?",
            "How does the Internet of Things work?",
            "What is NLP used for in AI?"
        ]
        
        # Display sample questions as buttons in two columns
        question_cols = st.columns(2)
        for i, sample_question in enumerate(sample_questions):
            with question_cols[i % 2]:
                if st.button(f"🔍 {sample_question}", key=f"sample_{i}"):
                    # Execute search for this question
                    with st.spinner(f"Searching: {sample_question}"):
                        # Add a small delay to make the spinner visible
                        time.sleep(0.5)
                        results = search_paragraphs(client, collection_name, model, sample_question, limit=num_results)
                    
                    # Display results
                    if results:
                        st.success(f"Found {len(results)} relevant answers")
                        
                        for j, result in enumerate(results):
                            with st.container():
                                st.subheader(result.payload["title"])
                                st.markdown(result.payload["content"])
                                st.caption(f"Relevance score: {result.score:.4f}")
                                st.divider()
                    else:
                        st.warning("No relevant information found.")
    
    with tab2:
        # Display the knowledge base
        st.subheader("Knowledge Base Content")
        
        # Create a DataFrame for display
        kb_data = [
            {"Title": p["title"], "Content": p["content"][:100] + "..."}
            for p in SAMPLE_PARAGRAPHS
        ]
        kb_df = pd.DataFrame(kb_data)
        
        st.dataframe(kb_df, use_container_width=True)
        
        # Expandable section to view full paragraphs
        for i, paragraph in enumerate(SAMPLE_PARAGRAPHS):
            with st.expander(f"View full text: {paragraph['title']}"):
                st.markdown(paragraph["content"])

if __name__ == "__main__":
    main()
