"""
standalone_rag_app.py
=======================
A standalone Streamlit web application that uses the underlying modules
of the RAG Benchmarking Framework to provide a working Chat UI.

How to run:
    .venv\\Scripts\\activate
    streamlit run standalone_rag_app.py
"""

import sys
from pathlib import Path

# Ensure the framework modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from loguru import logger

from data_ingestion.document_loader import DocumentLoader
from text_processing.text_cleaner import TextCleaner
from text_processing.chunker import get_chunker
from embedding_layer.embedder import get_embedder
from vector_database.vector_store import get_vector_store
from retrieval_system.retrievers import get_retriever
from llm_generation.llm_interface import get_llm
from llm_generation.prompt_builder import PromptBuilder
from llm_generation.answer_generator import AnswerGenerator


st.set_page_config(page_title="My Local RAG System", page_icon="🤖", layout="wide")

# ── 1. Setup Session State (Run Once) ──────────────────────────────────────────
@st.cache_resource
def init_rag_system():
    """Initialize the AI models and Vector DB."""
    logger.info("Initializing Local RAG System...")
    
    # You can change these to the best combination you found in your benchmarking!
    chunker = get_chunker("recursive", chunk_size=512, chunk_overlap=50)
    embedder = get_embedder("minilm", device="cpu")
    
    # NEW: We are now providing a persist_dir so the database is saved to disk!
    vector_store = get_vector_store("chroma", persist_dir="results_storage/main_app_db") 
    llm = get_llm("mistral")  # Uses your local Ollama
    
    retriever = get_retriever("basic", embedder=embedder, store=vector_store)
    prompt_builder = PromptBuilder(language="en")
    
    generator = AnswerGenerator(
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder,
        top_k=5
    )
    
    return chunker, embedder, vector_store, generator

# ── 2. Data Ingestion Function ─────────────────────────────────────────────────
def ingest_documents(data_path: str, chunker, embedder, vector_store):
    """Load PDFs, clean them, chunk them, and save to Vector DB."""
    with st.spinner(f"Ingesting documents from {data_path}..."):
        loader = DocumentLoader()
        cleaner = TextCleaner()
        
        raw_texts = []
        for doc in loader.load_directory(Path(data_path), recursive=True):
            if doc["type"] in ["text", "pdf"]:
                clean_text = cleaner.clean(doc["content"])
                raw_texts.append(clean_text)
                
        if not raw_texts:
            st.error("No text found in your documents!")
            return False
            
        full_text = "\n\n".join(raw_texts)
        chunks = chunker.split(full_text)
        
        st.info(f"Created {len(chunks)} chunks. Generating embeddings...")
        chunk_vecs = embedder.embed(chunks)
        
        vector_store.clear()
        vector_store.index(chunks, chunk_vecs)
        return True


# ── 3. Web App UI ──────────────────────────────────────────────────────────────
def main():
    st.title("🤖 My Local RAG Chatbot")
    st.caption("Powered by my PFE Framework Modules & Local Ollama")
    
    # Initialize the engine
    chunker, embedder, vector_store, generator = init_rag_system()
    
    # Check if documents are already in the DB
    doc_count = vector_store.get_chunk_count()
    if doc_count > 0:
        st.session_state["ready"] = True
    
    # Sidebar for controls
    with st.sidebar:
        st.header("1. Knowledge Base")
        
        if st.session_state.get("ready"):
            st.success(f"✅ Database loaded with {doc_count} chunks.")
            st.info("You can start chatting, or re-ingest below if you added new files.")
        else:
            st.warning("No documents loaded yet.")
            
        doc_folder = st.text_input("Folder Path", value="data/documents")
        
        if st.button("Ingest / Re-scan Documents"):
            success = ingest_documents(doc_folder, chunker, embedder, vector_store)
            if success:
                st.success("✅ Documents completely embedded and indexed!")
                st.session_state["ready"] = True
                st.rerun()
                
        st.divider()
        st.write("Current Settings:")
        st.write("- **LLM:** Mistral (Ollama)")
        st.write("- **Embeddings:** MiniLM")
        st.write("- **DB:** Chroma")
    
    # Chat Interface
    st.header("2. Chat with your PDFs")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am ready. Ask me anything about your documents."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question about the PDFs..."):
        if not st.session_state.get("ready"):
            st.warning("Please click 'Ingest Documents' in the sidebar first!")
            return
            
        # Add user format to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking..."):
                result = generator.answer_batch([prompt])[0]
                answer = result["answer"]
                
                # Show the answer
                st.write(answer)
                
                # Show the retrieved context as expandable proof
                with st.expander("Show Retrieved Context"):
                    for i, chunk in enumerate(result["retrieved_chunks"]):
                        st.markdown(f"**Chunk {i+1}:**\\n {chunk}")
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
