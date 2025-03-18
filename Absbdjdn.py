import os
from typing import List, Dict, Any, Optional, Tuple
import uuid
import json
from datetime import datetime

# PDF Processing
import fitz  # PyMuPDF
import re
from pathlib import Path

# Vector DB and Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# RAG Components
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import numpy as np

# Session Management
class Session:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.chat_history = []
        self.created_at = datetime.now().isoformat()
    
    def add_message(self, role: str, content: str):
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        return self.chat_history[-count:] if len(self.chat_history) > 0 else []
    
    def save(self, directory: str = "sessions"):
        os.makedirs(directory, exist_ok=True)
        with open(f"{directory}/{self.session_id}.json", "w") as f:
            json.dump({
                "session_id": self.session_id,
                "created_at": self.created_at,
                "chat_history": self.chat_history
            }, f)
    
    @classmethod
    def load(cls, session_id: str, directory: str = "sessions") -> "Session":
        try:
            with open(f"{directory}/{session_id}.json", "r") as f:
                data = json.load(f)
                session = cls(session_id=data["session_id"])
                session.created_at = data["created_at"]
                session.chat_history = data["chat_history"]
                return session
        except FileNotFoundError:
            return cls(session_id=session_id)

# PDF Processing with Context Preservation
class EnhancedPDFProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = fitz.open(file_path)
        self.metadata = self._extract_metadata()
        
    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract basic metadata from the PDF"""
        return {
            "title": self.doc.metadata.get("title", Path(self.file_path).stem),
            "author": self.doc.metadata.get("author", "Unknown"),
            "subject": self.doc.metadata.get("subject", ""),
            "page_count": len(self.doc),
            "file_path": self.file_path
        }
    
    def _detect_structure(self, page) -> Dict[str, Any]:
        """Detect various structural elements in a page"""
        blocks = page.get_text("dict")["blocks"]
        
        headers = []
        paragraphs = []
        tables = []
        
        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    if line["spans"]:
                        # Detect headers based on font size
                        span = line["spans"][0]
                        text = "".join([s["text"] for s in line["spans"]])
                        if span["size"] > 12 and len(text.strip()) < 100:  # Likely a header
                            headers.append({
                                "text": text,
                                "font_size": span["size"],
                                "bbox": line["bbox"]
                            })
                        else:
                            paragraphs.append(text)
            elif b.get("type") == 1:  # Image block, could be a table
                tables.append({
                    "bbox": b["bbox"],
                    "type": "image/table"
                })
                
        return {
            "headers": headers,
            "paragraphs": paragraphs,
            "tables": tables
        }
        
    def _is_toc_page(self, page_text: str) -> bool:
        """Detect if a page is a table of contents"""
        patterns = [
            r"^\s*table\s+of\s+contents\s*$",
            r"^\s*contents\s*$",
            r"^\s*index\s*$"
        ]
        
        lines = page_text.lower().split("\n")
        # Check if first few lines match TOC patterns
        for line in lines[:5]:
            if any(re.match(pattern, line.strip()) for pattern in patterns):
                return True
                
        # Check for page number patterns (common in TOC)
        page_num_pattern = r"\d+\s*$"
        page_num_lines = [line for line in lines if re.search(page_num_pattern, line.strip())]
        
        return len(page_num_lines) > len(lines) * 0.5  # If >50% lines have page numbers
    
    def _is_reference_page(self, page_text: str) -> bool:
        """Detect if a page is a references/bibliography page"""
        references_patterns = [
            r"^\s*references\s*$",
            r"^\s*bibliography\s*$",
            r"^\s*works\s+cited\s*$"
        ]
        
        lines = page_text.lower().split("\n")
        # Check if first few lines match reference patterns
        for line in lines[:5]:
            if any(re.match(pattern, line.strip()) for pattern in references_patterns):
                return True
                
        # Check for citation patterns
        citation_patterns = [
            r"\(\d{4}\)",  # (2023)
            r"\[\d+\]",    # [1]
            r"^\s*\d+\.\s+"  # 1. 
        ]
        
        citation_lines = 0
        for line in lines:
            if any(re.search(pattern, line) for pattern in citation_patterns):
                citation_lines += 1
                
        # If many lines have citation patterns, likely a reference page
        return citation_lines > len(lines) * 0.3
        
    def extract_text_with_context(self) -> List[Dict[str, Any]]:
        """
        Extract text while preserving structural context from the PDF
        Returns a list of document chunks with metadata
        """
        documents = []
        
        for page_num, page in enumerate(self.doc):
            page_text = page.get_text()
            
            # Skip processing if it's TOC or References, but still record them
            is_toc = self._is_toc_page(page_text)
            is_reference = self._is_reference_page(page_text)
            
            structure = self._detect_structure(page)
            
            # Get tables as text if possible
            try:
                tables_text = page.get_text("blocks", flags=fitz.TEXT_PRESERVE_IMAGES)
                tables_content = "\n".join([t for t in tables_text if isinstance(t, str)])
            except:
                tables_content = ""
            
            # Construct hierarchical context
            current_section = ""
            current_subsection = ""
            
            # Sort headers by their y-coordinate to maintain order
            sorted_headers = sorted(structure["headers"], key=lambda h: h["bbox"][1])
            
            for header in sorted_headers:
                header_text = header["text"].strip()
                if not header_text:
                    continue
                    
                if header["font_size"] > 14:  # Main section
                    current_section = header_text
                    current_subsection = ""
                else:  # Subsection
                    current_subsection = header_text
            
            # Combine paragraphs with context
            text_with_context = "\n".join(structure["paragraphs"])
            
            # Create document with detailed metadata
            doc_metadata = {
                **self.metadata,
                "page_num": page_num + 1,
                "section": current_section,
                "subsection": current_subsection,
                "is_toc": is_toc,
                "is_reference": is_reference,
                "has_tables": len(structure["tables"]) > 0,
                "tables_count": len(structure["tables"])
            }
            
            # Don't skip TOC and references - include with specific metadata
            documents.append({
                "text": text_with_context,
                "metadata": doc_metadata
            })
            
            # If page has tables, add them as separate chunks with table context
            if tables_content and len(structure["tables"]) > 0:
                documents.append({
                    "text": f"[TABLE CONTENT]\n{tables_content}\n[/TABLE CONTENT]",
                    "metadata": {**doc_metadata, "content_type": "table"}
                })
        
        return documents

# Enhanced Chunking for RAG
class EnhancedTextSplitter:
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200, 
        preserve_structure: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_structure = preserve_structure
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Split documents while preserving context"""
        results = []
        
        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            
            # Special handling for TOC and References pages
            if metadata.get("is_toc") or metadata.get("is_reference"):
                # Use smaller chunks for these pages
                splits = self.text_splitter.create_documents(
                    texts=[text], 
                    metadatas=[metadata]
                )
                results.extend(splits)
                continue
                
            # Special handling for tables
            if metadata.get("content_type") == "table":
                # Use larger chunks for tables to preserve context
                table_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size * 2,  # Larger chunks for tables
                    chunk_overlap=self.chunk_overlap * 2
                )
                splits = table_splitter.create_documents(
                    texts=[text], 
                    metadatas=[{**metadata, "chunk_type": "table"}]
                )
                results.extend(splits)
                continue
            
            # Add section/subsection to chunk if available
            context_prefix = ""
            section = metadata.get("section", "")
            subsection = metadata.get("subsection", "")
            
            if section and self.preserve_structure:
                context_prefix += f"Section: {section}\n"
            if subsection and self.preserve_structure:
                context_prefix += f"Subsection: {subsection}\n"
                
            # Regular content chunks
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                # Add context prefix to first chunk or all chunks if specified
                if i == 0 or self.preserve_structure:
                    chunk_with_context = f"{context_prefix}{chunk}"
                else:
                    chunk_with_context = chunk
                    
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_count": len(chunks)
                }
                
                results.append(Document(
                    page_content=chunk_with_context,
                    metadata=chunk_metadata
                ))
                
        return results

# Enhanced RAG Retriever with Hybrid Search
class HybridRetriever(BaseRetriever):
    def __init__(
        self, 
        qdrant_client: QdrantClient,
        collection_name: str,
        embedding_model: SentenceTransformer,
        top_k: int = 5,
        use_reranker: bool = False,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        super().__init__()
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.bm25_index = None
        self.documents = []
        self.reranker = None
        
        # Initialize reranker if specified
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model_name)
    
    def build_bm25_index(self, documents: List[Document]):
        """Build BM25 index for sparse retrieval"""
        texts = [doc.page_content for doc in documents]
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.documents = documents
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Vector search
        query_vector = self.embedding_model.encode(query)
        vector_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.top_k * 2  # Get more for reranking
        )
        
        # Get Document objects from vector search
        ids = [hit.id for hit in vector_results]
        scores = [hit.score for hit in vector_results]
        
        # Build document result map
        vector_docs = []
        for id, score in zip(ids, scores):
            # Get point from Qdrant
            points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[id]
            )
            if points:
                payload = points[0].payload
                vector_docs.append(Document(
                    page_content=payload.get("text", ""),
                    metadata={
                        **payload.get("metadata", {}),
                        "score": score,
                        "retrieval_method": "vector"
                    }
                ))
        
        # If we have a BM25 index, use hybrid search
        hybrid_docs = vector_docs
        if self.bm25_index is not None:
            # BM25 search
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Normalize BM25 scores
            max_bm25 = max(bm25_scores)
            if max_bm25 > 0:
                bm25_scores = [score/max_bm25 for score in bm25_scores]
            
            # Get top BM25 results
            bm25_indices = np.argsort(bm25_scores)[-self.top_k*2:][::-1]
            bm25_docs = [
                Document(
                    page_content=self.documents[idx].page_content,
                    metadata={
                        **self.documents[idx].metadata,
                        "score": bm25_scores[idx],
                        "retrieval_method": "bm25" 
                    }
                )
                for idx in bm25_indices
            ]
            
            # Combine results
            hybrid_docs = vector_docs + bm25_docs
        
        # Apply reranking if specified
        if self.use_reranker and self.reranker and hybrid_docs:
            pairs = [(query, doc.page_content) for doc in hybrid_docs]
            rerank_scores = self.reranker.predict(pairs)
            
            # Sort by reranker scores
            scored_results = list(zip(hybrid_docs, rerank_scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Update scores and return top k
            reranked_docs = []
            for doc, score in scored_results[:self.top_k]:
                doc.metadata["reranker_score"] = float(score)
                reranked_docs.append(doc)
            
            return reranked_docs
            
        # Return top k unique documents
        seen_content = set()
        unique_docs = []
        
        for doc in sorted(hybrid_docs, key=lambda d: d.metadata.get("score", 0), reverse=True):
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
                if len(unique_docs) >= self.top_k:
                    break
                    
        return unique_docs

# Main RAG Application
class PDFChatApp:
    def __init__(
        self, 
        vector_db_path: str = "./qdrant_db",
        collection_name: str = "pdf_collection",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        use_reranker: bool = False
    ):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(path=vector_db_path)
        self.collection_name = collection_name
        
        # Make sure collection exists
        try:
            self.qdrant_client.get_collection(collection_name)
        except:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            qdrant_client=self.qdrant_client,
            collection_name=collection_name,
            embedding_model=self.embedding_model,
            top_k=5,
            use_reranker=use_reranker
        )
        
        # Session management
        self.sessions = {}
        
    def add_document(self, file_path: str) -> int:
        """Process and add a PDF document to the vector store"""
        # Process PDF
        pdf_processor = EnhancedPDFProcessor(file_path)
        raw_documents = pdf_processor.extract_text_with_context()
        
        # Split into chunks
        splitter = EnhancedTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            preserve_structure=True
        )
        documents = splitter.split_documents(raw_documents)
        
        # Add to BM25 index
        self.retriever.build_bm25_index(documents)
        
        # Add to vector database
        points = []
        for i, doc in enumerate(documents):
            # Create unique ID
            point_id = str(uuid.uuid4())
            
            # Get embedding
            embedding = self.embedding_model.encode(doc.page_content)
            
            # Create payload
            payload = {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            
            # Add to points
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
        
        # Insert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        return len(documents)
    
    def create_session(self) -> str:
        """Create a new chat session"""
        session = Session()
        self.sessions[session.session_id] = session
        return session.session_id
    
    def get_session(self, session_id: str) -> Session:
        """Get or create a session"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try to load from disk
        session = Session.load(session_id)
        self.sessions[session_id] = session
        return session
    
    def format_sources(self, documents: List[Document]) -> str:
        """Format source information for display"""
        sources = []
        
        for i, doc in enumerate(documents):
            metadata = doc.metadata
            source_info = f"Source {i+1}: "
            
            if "title" in metadata:
                source_info += f"{metadata['title']} • "
            
            source_info += f"Page {metadata.get('page_num', 'unknown')}"
            
            if "section" in metadata and metadata["section"]:
                source_info += f" • Section: {metadata['section']}"
                
            if "subsection" in metadata and metadata["subsection"]:
                source_info += f" • Subsection: {metadata['subsection']}"
                
            score = metadata.get("score", 0)
            source_info += f" • Relevance: {score:.2f}"
            
            sources.append(source_info)
            
        return "\n".join(sources)
        
    def chat(
        self, 
        query: str, 
        session_id: str = None,
        use_reranker: bool = False
    ) -> Dict[str, Any]:
        """Process a query and return response with sources"""
        # Get or create session
        if not session_id:
            session_id = self.create_session()
        
        session = self.get_session(session_id)
        
        # Add user message to history
        session.add_message("user", query)
        
        # Update reranker setting if needed
        if self.retriever.use_reranker != use_reranker:
            self.retriever.use_reranker = use_reranker
        
        # Get relevant documents
        documents = self.retriever._get_relevant_documents(query)
        
        # Format context for LLM
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Get recent chat history for context
        recent_messages = session.get_recent_messages(3)
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
        
        # Construct prompt with context
        prompt = f"""
        Chat History:
        {chat_history}
        
        Context from documents:
        {context}
        
        User Query: {query}
        
        Please respond to the query based on the context provided. Only use information from the context.
        """
        
        # Get LLM response
        try:
            # This is a placeholder - implement your actual local LLM call here
            response = abc_response(prompt)
        except Exception as e:
            response = f"Error processing query: {str(e)}"
        
        # Format sources
        sources_text = self.format_sources(documents)
        
        # Add response to history
        session.add_message("assistant", response)
        
        # Save session
        session.save()
        
        return {
            "response": response,
            "sources": sources_text,
            "session_id": session_id
        }

# Example usage
if __name__ == "__main__":
    # Initialize app
    chat_app = PDFChatApp(
        vector_db_path="./qdrant_db",
        collection_name="pdf_documents",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        use_reranker=False
    )
    
    # Add document
    num_chunks = chat_app.add_document("path/to/your/document.pdf")
    print(f"Processed document into {num_chunks} chunks")
    
    # Create session
    session_id = chat_app.create_session()
    
    # Chat example
    result = chat_app.chat("What is the main topic of this document?", session_id)
    
    print(result["response"])
    print("\nSources:")
    print(result["sources"])
