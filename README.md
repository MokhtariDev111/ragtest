# RAG Benchmarking Framework

A modular Python research platform that **automatically benchmarks every stage of a RAG pipeline** and produces structured experimental results for analysis and reporting.

---

## Architecture Overview

```
documents → OCR → cleaning → chunking → embeddings → vector DB → retrieval → LLM → evaluation → SQLite/CSV → Dashboard
```

### Modules

| Module | Description |
|---|---|
| `data_ingestion/` | Document loading (PDF, TXT, images) + OCR (Tesseract, EasyOCR, PaddleOCR, DocTR) |
| `text_processing/` | Text cleaning + 4 chunking strategies (Fixed, Recursive, Sliding Window, Semantic) |
| `embedding_layer/` | HuggingFace embedders: BGE, E5, Instructor, MiniLM |
| `vector_database/` | FAISS, Chroma, Qdrant with unified interface |
| `retrieval_system/` | Basic, Reranking (cross-encoder), Hybrid (BM25+vector), Multi-query |
| `llm_generation/` | Ollama LLM client (Mistral, Qwen, Llama 3.2) + prompt builder |
| `evaluation_framework/` | P@K, R@K, MRR, NDCG + Faithfulness, Answer Relevancy, Context Precision/Recall |
| `experiment_runner/` | Automated cartesian-product experiment loop |
| `results_storage/` | SQLite database + CSV export |
| `dashboard/` | Streamlit web dashboard with Plotly charts |

---

## Quick Start

### 1. Install Prerequisites

**Python 3.10+** is required.

**Tesseract OCR** (required for OCR benchmarking):
```bash
# Windows (with Chocolatey)
choco install tesseract
```

**Ollama** (required for LLM generation):
```bash
# Install from https://ollama.ai
ollama pull mistral
ollama pull qwen
ollama pull llama3.2
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

> **Note on OCR dependencies:**  
> - PaddleOCR: `pip install paddlepaddle paddleocr`  
> - DocTR: `pip install "python-doctr[torch]"`  
> These can be installed selectively based on which engines you want to use.

### 3. Add Your Documents

Place PDF or TXT files in `data/documents/`.  
Place scanned images in `data/images/`.

The sample file `data/documents/rag_overview.txt` is included for testing.

### 4. Run a Quick Experiment

```bash
# Fast smoke-test (only 1 variant per stage)
python main.py run --config config/experiment_config.yaml

# Full benchmark (all enabled combinations)
python main.py run --config config/default_config.yaml
```

### 5. View Results

```bash
# Launch the dashboard
python main.py dashboard

# Export to CSV
python main.py export
```

---

## Configuration

Edit `config/default_config.yaml` to control which techniques are benchmarked.  
Use `config/experiment_config.yaml` for a quick minimal test (1 option per stage).

Key sections:
- `ocr.engines` — enable/disable OCR engines
- `text_processing.chunking.strategies` — enable/disable chunking strategies
- `text_processing.chunking.chunk_sizes` — list of chunk sizes to test
- `embeddings.models` — enable/disable embedding models
- `vector_databases.stores` — enable/disable vector DBs
- `retrieval.strategies` — enable/disable retrieval strategies
- `llm.models` — enable/disable LLM models

---

## Evaluation Dataset

`data/evaluation/questions.json` contains Q&A pairs used to evaluate pipeline quality.

Format:
```json
{
  "questions": [
    {
      "id": "q001",
      "question": "...",
      "expected_answer": "...",
      "source_document": "..."
    }
  ]
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dashboard

The Streamlit dashboard reads from `results_storage/experiments.db` and shows:
- **Results Table** — filterable experiment log  
- **Accuracy Comparison** — bar charts + embedding × vector DB heatmap  
- **Latency Analysis** — latency breakdowns + box plots  
- **Metrics Deep Dive** — radar charts per technique  
- **Model Rankings** — composite score leaderboard  

---

## Adding New Techniques

### New OCR Engine
Subclass `data_ingestion.ocr_pipeline.BaseOCREngine` and register with `register_ocr_engine()`.

### New Chunker
Subclass `text_processing.chunker.BaseChunker` and register with `register_chunker()`.

### New Embedder
Subclass `embedding_layer.embedder.BaseEmbedder` and register with `register_embedder()`.

### New Vector Store
Subclass `vector_database.vector_store.BaseVectorStore` and register with `register_vector_store()`.

### New Retriever
Subclass `retrieval_system.retrievers.BaseRetriever` and add to the registry in `retrievers.py`.

### New LLM
Subclass `llm_generation.llm_interface.BaseLLM` and register with `register_llm()`.

---

## Project Structure

```
rag_benchmark/
├── config/
│   ├── default_config.yaml
│   └── experiment_config.yaml
├── data/
│   ├── documents/         ← add your PDFs and TXT files here
│   ├── images/            ← add scanned images here
│   └── evaluation/
│       └── questions.json
├── data_ingestion/
│   ├── document_loader.py
│   ├── ocr_pipeline.py
│   └── ocr_benchmark.py
├── text_processing/
│   ├── text_cleaner.py
│   └── chunker.py
├── embedding_layer/
│   ├── embedder.py
│   └── embedding_benchmark.py
├── vector_database/
│   ├── vector_store.py
│   └── vector_db_benchmark.py
├── retrieval_system/
│   └── retrievers.py
├── llm_generation/
│   ├── llm_interface.py
│   ├── prompt_builder.py
│   └── answer_generator.py
├── evaluation_framework/
│   ├── retrieval_metrics.py
│   └── rag_metrics.py
├── experiment_runner/
│   └── runner.py
├── results_storage/
│   ├── database.py
│   ├── exporter.py
│   └── exports/
├── dashboard/
│   └── app.py
├── tests/
│   ├── test_text_processing.py
│   └── test_evaluation.py
├── main.py
├── requirements.txt
└── README.md
```
