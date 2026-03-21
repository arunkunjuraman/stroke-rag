import streamlit as st
import os
import sys
from typing import List

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from brain import stroke_rag_app
except ImportError:
    # If brain.py is not in current dir, try parent
    sys.path.append(os.getcwd())
    from src.brain import stroke_rag_app

# Page Configuration
st.set_page_config(
    page_title="Stroke-RAG Assistant",
    page_icon="🧠",
    layout="wide",
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    .header-container {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border-left: 5px solid #007bff;
    }
    
    .header-title {
        color: #1a202c;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #4a5568;
        font-size: 1.1rem;
    }

    /* Message styling */
    .stChatMessage {
        background-color: white !important;
        border-radius: 1rem !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
        margin-bottom: 1rem !important;
        border: 1px solid #edf2f7 !important;
    }
    
    .stChatMessage[data-testimonial="user"] {
        border-left: 4px solid #007bff !important;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    /* Source Badge */
    .source-badge {
        display: inline-block;
        background: #e2e8f0;
        color: #4a5568;
        padding: 0.2rem 0.6rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        margin: 0.2rem;
        font-weight: 600;
    }
    
    /* Expander styling */
    .stExpander {
        border: none !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491299.png", width=100)
    st.title("Settings & Info")
    st.info("""
    **Models Used:**
    - LLM: GPT-4o
    - Embedding: text-embedding-3-large
    - Re-ranker: Cohere v3.0
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### Guideline Versions")
    st.caption("Currently using: AHA/ASA Stroke Guidelines (v2 Comparative)")

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🧠 Stroke-RAG Clinical Assistant</div>
    <div class="header-subtitle">Intelligent Retrieval-Augmented Guidance for Clinical Stroke Protocols.</div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Supporting Evidence"):
                cols = st.columns(2)
                for i, source in enumerate(message["sources"]):
                    col_idx = i % 2
                    cols[col_idx].markdown(f"""
                    <div class="source-badge">
                        📄 {source['file']} | Pg. {source['page']}
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a clinical question (e.g., BP targets for acute stroke)..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Analyzing guidelines and retrieving clinical evidence..."):
            try:
                # Invoke RAG Graph
                response = stroke_rag_app.invoke({"question": prompt})
                
                answer = response["generation"]
                documents = response.get("documents", [])
                
                # Format sources
                sources = []
                for d in documents:
                    source_file = os.path.basename(d.metadata.get('source', 'Unknown'))
                    page_num = d.metadata.get('page', 'Unknown')
                    sources.append({"file": source_file, "page": page_num})
                
                # Display Answer
                if response.get("is_cached"):
                    st.caption("⚡ *Cached Answer*")
                response_placeholder.markdown(answer)

                
                # Display Sources
                if sources:
                    with st.expander("📚 View Supporting Evidence"):
                        cols = st.columns(2)
                        for i, source in enumerate(sources):
                            col_idx = i % 2
                            cols[col_idx].markdown(f"""
                            <div class="source-badge">
                                📄 {source['file']} | Pg. {source['page']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}"
                })
