import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from components.pdf_processor import PDFProcessor
from components.embeddings import EmbeddingHandler
from components.llm_handler import LLMHandler

# Page configuration
st.set_page_config(
    page_title="StudyMate - AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .upload-section {
        border: 2px dashed #ccc;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processor" not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if "embedding_handler" not in st.session_state:
    st.session_state.embedding_handler = EmbeddingHandler()
if "llm_handler" not in st.session_state:
    st.session_state.llm_handler = LLMHandler()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 StudyMate - AI Study Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for file upload and management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload your textbooks, lecture notes, or research papers"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Show processed files
        if st.session_state.processed_files:
            st.subheader("✅ Processed Documents")
            for i, file_name in enumerate(st.session_state.processed_files):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file_name}")
                with col2:
                    if st.button("🗑", key=f"delete_{i}", help="Remove document"):
                        remove_document(i)
                        st.rerun()
        
        # Clear all button
        if st.session_state.processed_files:
            if st.button("🗑 Clear All Documents", type="secondary"):
                clear_all_documents()
                st.rerun()
        
        st.markdown("---")
        
        # Instructions
        st.subheader("ℹ How to Use")
        st.markdown("""
        1. *Upload PDFs*: Click 'Browse files' to upload your study materials
        2. *Wait for Processing*: Let StudyMate read and understand your documents
        3. *Ask Questions*: Type any question about your materials
        4. *Get Answers*: Receive detailed answers with source references
        """)
        
        # Statistics
        if st.session_state.text_chunks:
            st.subheader("📊 Statistics")
            st.metric("Documents Processed", len(st.session_state.processed_files))
            st.metric("Text Chunks", len(st.session_state.text_chunks))
    
    # Main chat interface
    main_container = st.container()
    
    with main_container:
        if not st.session_state.processed_files:
            # Welcome screen when no documents are uploaded
            st.markdown("""
            <div class="upload-section">
                <h2>🎓 Welcome to StudyMate!</h2>
                <p>Upload your PDF study materials to get started.</p>
                <p>StudyMate will help you:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>📖 Understand complex concepts</li>
                    <li>🔍 Find specific information quickly</li>
                    <li>💡 Get explanations with source references</li>
                    <li>🎯 Study more efficiently</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Chat interface
            st.subheader("💬 Ask StudyMate")
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Sources"):
                            for source in message["sources"]:
                                st.markdown(f"{source['file']}** - Page {source['page']}")
                                st.markdown(f"{source['text'][:200]}...")
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your documents..."):
                handle_user_query(prompt)

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files"""
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Extract text from PDF
                    text_data = st.session_state.pdf_processor.extract_text_from_pdf(
                        tmp_file_path, uploaded_file.name
                    )
                    
                    if text_data:
                        # Add to text chunks
                        st.session_state.text_chunks.extend(text_data)
                        
                        # Update embeddings
                        st.session_state.embedding_handler.update_embeddings(text_data)
                        
                        # Add to processed files
                        st.session_state.processed_files.append(uploaded_file.name)
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                    else:
                        st.error(f"❌ Could not extract text from {uploaded_file.name}")
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

def handle_user_query(query):
    """Handle user questions"""
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Search for relevant context
                search_results = st.session_state.embedding_handler.semantic_search(query)
                
                if search_results:
                    # Generate answer using LLM
                    context = "\n".join([result['text'] for result in search_results])
                    answer = st.session_state.llm_handler.generate_answer(query, context)
                    
                    # Prepare sources
                    sources = [
                        {
                            "file": result['file'],
                            "page": result['page'],
                            "text": result['text']
                        }
                        for result in search_results[:3]  # Top 3 sources
                    ]
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.markdown(f"{source['file']}** - Page {source['page']}")
                            st.markdown(f"{source['text'][:200]}...")
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                else:
                    error_msg = "I couldn't find relevant information in your documents to answer this question. Please try rephrasing or upload more relevant materials."
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

def remove_document(index):
    """Remove a processed document"""
    if 0 <= index < len(st.session_state.processed_files):
        removed_file = st.session_state.processed_files.pop(index)
        
        # Remove text chunks from this file
        st.session_state.text_chunks = [
            chunk for chunk in st.session_state.text_chunks 
            if chunk['file'] != removed_file
        ]
        
        # Update embeddings
        if st.session_state.text_chunks:
            st.session_state.embedding_handler.update_embeddings(st.session_state.text_chunks)
        else:
            st.session_state.embedding_handler = EmbeddingHandler()
        
        st.success(f"Removed {removed_file}")

def clear_all_documents():
    """Clear all processed documents"""
    st.session_state.processed_files = []
    st.session_state.text_chunks = []
    st.session_state.messages = []
    st.session_state.embedding_handler = EmbeddingHandler()
    st.success("All documents cleared!")

if _name_ == "_main_":
    main()
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
import streamlit as st

class EmbeddingHandler:
    def _init_(self):
        self.model_name = 'all-MiniLM-L6-v2'  # Fast and efficient model
        self.model = None
        self.index = None
        self.text_chunks = []
        self.embeddings = None
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            with st.spinner("Loading AI model..."):
                self.model = SentenceTransformer(self.model_name)
            st.success("✅ AI model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            self.model = None
    
    def create_embeddings(self, text_chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks
        
        Args:
            text_chunks (List[str]): List of text chunks
            
        Returns:
            np.ndarray: Embeddings matrix
        """
        if not self.model:
            st.error("Model not loaded!")
            return np.array([])
        
        try:
            with st.spinner("Creating embeddings..."):
                embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return np.array([])
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for similarity search
        
        Args:
            embeddings (np.ndarray): Embeddings matrix
        """
        try:
            if embeddings.size == 0:
                return
                
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
        except Exception as e:
            st.error(f"Error building search index: {str(e)}")
    
    def update_embeddings(self, text_data: List[Dict]):
        """
        Update embeddings with new text data
        
        Args:
            text_data (List[Dict]): List of text chunks with metadata
        """
        if not self.model:
            st.error("Model not loaded!")
            return
        
        try:
            # Store text data
            self.text_chunks = text_data
            
            # Extract just the text for embedding
            texts = [chunk['text'] for chunk in text_data]
            
            if texts:
                # Create embeddings
                self.embeddings = self.create_embeddings(texts)
                
                if self.embeddings.size > 0:
                    # Build FAISS index
                    self.build_faiss_index(self.embeddings)
                    st.success(f"✅ Created embeddings for {len(texts)} text chunks")
                else:
                    st.error("❌ Failed to create embeddings")
            
        except Exception as e:
            st.error(f"Error updating embeddings: {str(e)}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for most relevant text chunks using semantic similarity
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant text chunks with scores
        """
        if not self.model or not self.index or not self.text_chunks:
            return []
        
        try:
            # Create embedding for the query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.text_chunks):  # Valid index
                    result = self.text_chunks[idx].copy()
                    result['similarity_score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Error in semantic search: {str(e)}")
            return []
    
    def get_similar_chunks(self, text: str, top_k: int = 3) -> List[Dict]:
        """
        Find chunks similar to given text
        
        Args:
            text (str): Reference text
            top_k (int): Number of similar chunks to return
            
        Returns:
            List[Dict]: Similar text chunks
        """
        return self.semantic_search(text, top_k)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the embedding handler
        
        Returns:
            Dict: Statistics
        """
        return {
            'model_name': self.model_name,
            'total_chunks': len(self.text_chunks),
            'index_ready': self.index is not None,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None
        }
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Simple keyword-based search as fallback
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: Matching text chunks
        """
        if not self.text_chunks:
            return []
        
        query_words = query.lower().split()
        results = []
        
        for chunk in self.text_chunks:
            text_lower = chunk['text'].lower()
            matches = sum(1 for word in query_words if word in text_lower)
            
            if matches > 0:
                chunk_copy = chunk.copy()
                chunk_copy['keyword_matches'] = matches
                chunk_copy['match_ratio'] = matches / len(query_words)
                results.append(chunk_copy)
        
        # Sort by number of matches
        results.sort(key=lambda x: x['keyword_matches'], reverse=True)
        
        return results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Combine semantic and keyword search for better results
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: Combined search results
        """
        # Get semantic search results
        semantic_results = self.semantic_search(query, top_k)
        
        # Get keyword search results
        keyword_results = self.keyword_search(query, top_k)
        
        # Combine and deduplicate results
        combined = {}
        
        # Add semantic results with higher weight
        for result in semantic_results:
            chunk_id = result['chunk_id']
            result['search_type'] = 'semantic'
            combined[chunk_id] = result
        
        # Add keyword results (if not already present)
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id not in combined:
                result['search_type'] = 'keyword'
                combined[chunk_id] = result
        
        # Convert back to list and sort by relevance
        final_results = list(combined.values())
        
        # Sort by semantic score if available, otherwise by keyword matches
        final_results.sort(key=lambda x: x.get('similarity_score', x.get('match_ratio', 0)), reverse=True)
        
        return final_results[:top_k]
import streamlit as st
from typing import Dict, List
import requests
import json

class LLMHandler:
    def _init_(self):
        """Initialize the LLM handler"""
        self.model_options = {
            "OpenAI GPT": self._call_openai,
            "Hugging Face": self._call_huggingface,
            "Local Model": self._call_local_model,
            "Mock Response": self._generate_mock_response  # For demo purposes
        }
        
        # Default to mock response for demo
        self.current_model = "Mock Response"
        
        # System prompt for the AI
        self.system_prompt = """You are StudyMate, an AI assistant that helps students learn from their study materials.

Your job is to:
1. Answer questions based ONLY on the provided context from the student's documents
2. Provide clear, educational explanations
3. Always mention which document/page your information comes from
4. If the context doesn't contain the answer, say so clearly
5. Be encouraging and helpful in your tone

Guidelines:
- Use simple, clear language that students can understand
- Break down complex concepts into smaller parts
- Give examples when helpful
- Always cite your sources from the provided context
- If you're not sure about something, say so"""

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer based on the query and context
        
        Args:
            query (str): User's question
            context (str): Relevant text from documents
            
        Returns:
            str: Generated answer
        """
        # Create the prompt
        prompt = f"""Context from student's documents:
{context}

Student's question: {query}

Please provide a clear, helpful answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        
        # Use the selected model
        try:
            response = self.model_options[self.current_model](prompt)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response for demonstration purposes
        
        Args:
            prompt (str): The input prompt
            
        Returns:
            str: Mock response
        """
        # Extract question from prompt
        lines = prompt.split('\n')
        question = ""
        context = ""
        
        for i, line in enumerate(lines):
            if "Student's question:" in line:
                question = line.replace("Student's question:", "").strip()
            elif "Context from student's documents:" in line:
                # Get context (everything between this line and the question)
                context_start = i + 1
                for j in range(context_start, len(lines)):
                    if "Student's question:" in lines[j]:
                        break
                    context += lines[j] + " "
        
        # Generate a contextual response
        if not context.strip():
            return "I don't have any relevant information in your uploaded documents to answer this question. Please make sure you've uploaded the right materials or try rephrasing your question."
        
        # Simple response generation based on keywords
        response = f"Based on your study materials, here's what I found:\n\n"
        
        # Add context-based response
        context_snippet = context.strip()[:200] + "..." if len(context.strip()) > 200 else context.strip()
        
        if any(word in question.lower() for word in ['what', 'define', 'definition']):
            response += f"The definition appears to be related to: {context_snippet}\n\n"
        elif any(word in question.lower() for word in ['how', 'process', 'steps']):
            response += f"The process involves: {context_snippet}\n\n"
        elif any(word in question.lower() for word in ['why', 'reason', 'because']):
            response += f"The reason is explained as: {context_snippet}\n\n"
        else:
            response += f"According to your documents: {context_snippet}\n\n"
        
        response += "This information was found in your uploaded study materials. Would you like me to elaborate on any part of this explanation?"
        
        return response
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API (requires API key)
        
        Args:
            prompt (str): The input prompt
            
        Returns:
            str: API response
        """
        api_key = st.secrets.get("openai_api_key", "")
        
        if not api_key:
            return "OpenAI API key not configured. Please add it to your secrets or use the demo mode."
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(