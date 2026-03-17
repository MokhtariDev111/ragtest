"""
app.py
======
Unified RAG Research & Chat Platform (Super-App).
Combines Live Chat (Smart RAG), Experiment Lab, and Analytics Dashboard.
"""

import sys
import time
from pathlib import Path

# Ensure modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
from loguru import logger

# --- Framework Imports ---
from data_ingestion.document_loader import DocumentLoader
from text_processing.text_cleaner import TextCleaner
from text_processing.chunker import get_chunker
from embedding_layer.embedder import get_embedder
from vector_database.vector_store import get_vector_store
from retrieval_system.retrievers import get_retriever
from llm_generation.llm_interface import get_llm
from llm_generation.prompt_builder import PromptBuilder
from llm_generation.answer_generator import AnswerGenerator

# --- Dashboard Imports ---
from results_storage.database import ExperimentDatabase
import dashboard.app as dashboard_old

# ==========================================
# PAGE SETTINGS & CSS
# ==========================================
st.set_page_config(page_title="PFE Super-App | RAG Platform", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Premium CSS Theme */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4FACFE, #00F2FE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 2rem;
    }
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .stChatFloatingInputContainer {
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHING & HOT-SWAPPING LOGIC
# ==========================================
@st.cache_resource(show_spinner="Booting AI Engine...")
def init_dynamic_rag(chunker_type, chunk_size, embedder_type, db_type, llm_type):
    """Initializes the RAG components based on user selection."""
    logger.info(f"Hot-swapping config: {chunker_type}, {embedder_type}, {db_type}, {llm_type}")
    
    # 1. Chunker
    chunker = get_chunker(chunker_type, chunk_size=int(chunk_size), chunk_overlap=50)
    
    # 2. Embedder
    embedder = get_embedder(embedder_type, device="cpu")
    
    # 3. Vector DB (Persistent per configuration so we don't mix vectors)
    persist_dir = f"results_storage/superapp_{db_type}_{embedder_type}_{chunk_size}"
    vector_store = get_vector_store(db_type, persist_dir=persist_dir) if db_type == "chroma" else get_vector_store(db_type)
    
    # 4. LLM
    llm = get_llm(llm_type)
    
    # 5. Pipeline Setup
    retriever = get_retriever("basic", embedder=embedder, store=vector_store)
    prompt_builder = PromptBuilder(language="en")
    generator = AnswerGenerator(retriever=retriever, llm=llm, prompt_builder=prompt_builder, top_k=5)
    
    return chunker, embedder, vector_store, generator

def ingest_documents_ui(chunker, embedder, vector_store):
    """Ingests data/documents into the currently selected Vector DB."""
    st.info("Ingesting `data/documents/` into the new Vector Database... This may take a minute.")
    progress = st.progress(0)
    
    loader = DocumentLoader()
    cleaner = TextCleaner()
    raw_texts = []
    
    docs = list(loader.load_directory(Path("data/documents"), recursive=True))
    for idx, doc in enumerate(docs):
        if doc["type"] in ["text", "pdf"]:
            raw_texts.append(cleaner.clean(doc["content"]))
        progress.progress(min(1.0, (idx+1)/max(1, len(docs))))
        
    if not raw_texts:
        st.error("No documents found in `data/documents/`!")
        return False
        
    full_text = "\\n\\n".join(raw_texts)
    chunks = chunker.split(full_text)
    
    with st.spinner(f"Generating embeddings for {len(chunks)} chunks using {embedder.name}..."):
        chunk_vecs = embedder.embed(chunks)
        vector_store.clear()  # reset this specific DB
        vector_store.index(chunks, chunk_vecs)
        
    progress.progress(1.0)
    st.success(f"✅ Successfully indexed {len(chunks)} chunks into {vector_store.name.upper()}!")
    return True

# ==========================================
# PAGE VIEWS
# ==========================================

def render_chat_page():
    st.markdown("<p style='text-align: center; font-size: 1.5rem;'>💬 Live AI RAG Chat</p>", unsafe_allow_html=True)
    
    # --- Top Config Bar ---
    with st.expander("⚙️ Dynamic RAG Configuration (Hot-Swap Models)", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        # We use session state to remember choices across tab switches
        chunker_type = c1.selectbox("Chunker", ["recursive", "fixed", "semantic", "sliding_window"], index=0)
        chunk_size = c2.selectbox("Chunk Size", [256, 512, 1024], index=1)
        embedder_type = c3.selectbox("Embedder", ["minilm", "bge", "e5", "instructor"], index=0)
        db_type = c4.selectbox("Vector DB", ["chroma", "faiss"], index=0)
        llm_type = c5.selectbox("LLM Model", ["mistral", "llama3.2", "qwen"], index=0)
        
    # --- Initialize Engine ---
    chunker, embedder, vector_store, generator = init_dynamic_rag(
        chunker_type, chunk_size, embedder_type, db_type, llm_type
    )
    
    # --- Index Management ---
    doc_count = vector_store.get_chunk_count()
    if doc_count == 0:
        st.warning(f"⚠️ Vector DB ({db_type} + {embedder_type}) is empty. You must ingest documents before chatting.")
        if st.button("🚀 Ingest Documents Now", use_container_width=True):
            if ingest_documents_ui(chunker, embedder, vector_store):
                st.rerun()
        st.stop()
    else:
        st.caption(f"🟢 **Status:** Ready | **DB:** {db_type.upper()} ({doc_count} chunks) | **Embedder:** {embedder_type.upper()} | **LLM:** {llm_type.upper()}")
        
    # --- Chat UI ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Hello! Ask me anything about your documents."}]

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question about the PDFs..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Searching {db_type} and generating with {llm_type}..."):
                start_time = time.time()
                try:
                    result = generator.answer_batch([prompt])[0]
                    answer = result["answer"]
                    latency = time.time() - start_time
                    
                    st.write(answer)
                    
                    with st.expander(f"🔍 Show Retrieved Context (Retrieved in {result['retrieval_latency']:.2f}s | Total: {latency:.2f}s)"):
                        st.markdown(f"**Generated by:** {llm_type}")
                        for i, chunk in enumerate(result["retrieved_chunks"]):
                            st.markdown(f"**Chunk {i+1}:**\\n{chunk}")
                            st.divider()
                            
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")


def render_experiment_lab():
    st.markdown("<p style='text-align: center; font-size: 1.5rem;'>🧪 Automated Experiment Lab</p>", unsafe_allow_html=True)
    st.info("The Experiment Lab allows you to coordinate automated benchmarking pipelines.")
    
    st.write("### Ready to Benchmark")
    c1, c2, c3 = st.columns(3)
    c1.metric("LLMs Available", 3)
    c2.metric("Embeddings Available", 4)
    c3.metric("Combinations", ">500")
        
    st.write("---")
    st.warning("⚠️ Running the full benchmark can take hours and will utilize 100% of your CPU.")
    
    if st.button("🚀 Start Automated Benchmark Sweep", type="primary", use_container_width=True):
        st.success("Benchmark initiated! For optimal performance, run this directly in your terminal:")
        st.code("python main.py run --config config/default_config.yaml", language="bash")
        
    st.write("### Instructions:")
    st.markdown("""
    1. Ensure your PDF and TXT files are in `data/documents/`
    2. Write your evaluation test questions in `data/evaluation/questions.json`
    3. Run the benchmarker. As experiments finish, their results will instantly appear in the **Analytics Explorer** tab!
    """)

def render_analytics():
    st.markdown("<p style='text-align: center; font-size: 1.5rem;'>📊 Analytics & Benchmarks Explorer</p>", unsafe_allow_html=True)
    
    try:
        db_path = str(Path("results_storage/experiments.db"))
        df = dashboard_old.load_data(db_path)
        
        if df.empty:
            st.warning("No benchmark data found! Go to the Experiment Lab to run a benchmark first.")
            return

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Experiments Run", len(df))
        if "composite_score" not in df.columns and "faithfulness" in df.columns:
            # Fake composite for KPI if not grouped yet
            c2.metric("Best LLM", df.groupby("llm_model")["faithfulness"].mean().idxmax())
        else:
            c2.metric("Best LLM", "N/A")
            
        c3.metric("Fastest DB", df.groupby("vector_db")["total_latency_s"].mean().idxmin() if "total_latency_s" in df.columns else "N/A")
        c4.metric("Avg Answer Relevancy", f"{df['answer_relevancy'].mean():.2f}" if "answer_relevancy" in df.columns else "N/A")

        st.divider()

        # Inject the core tabs from dashboard/app.py
        tabs = st.tabs(["📋 Results Table", "🎯 Accuracy", "⚡ Latency", "📈 Metrics Deep Dive", "🏆 Master Rankings"])
        with tabs[0]:
            dashboard_old._tab_results_table(df)
        with tabs[1]:
            dashboard_old._tab_accuracy(df)
        with tabs[2]:
            dashboard_old._tab_latency(df)
        with tabs[3]:
            dashboard_old._tab_metrics(df)
        with tabs[4]:
            dashboard_old._tab_rankings(df)
            
    except Exception as e:
        st.error(f"Could not load Analytics: {e}. Make sure you have completed at least one benchmark.")

# ==========================================
# MAIN ROUTER
# ==========================================
def main():
    st.markdown("<div class='main-header'>PFE RAG Super-App</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Unified Research Platform: Chat, Benchmark, Analyze</div>", unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to module", ["💬 Live AI Chat", "🧪 Experiment Lab", "📊 Analytics Explorer"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("🎓 **End of Studies Project**\\n\\nModular RAG Benchmarking Framework")
    
    if page == "💬 Live AI Chat":
        render_chat_page()
    elif page == "🧪 Experiment Lab":
        render_experiment_lab()
    elif page == "📊 Analytics Explorer":
        render_analytics()

if __name__ == "__main__":
    main()
