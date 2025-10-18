# GraphRAG Contract Analyzer

Reference application that combines Retrieval-Augmented Generation (RAG) with knowledge graphs to answer questions about PDF contracts. Designed as a starting point for developers building document-analysis solutions with Streamlit, LangChain, NetworkX, FAISS, and OpenAI.

## Overview

- Upload contracts in PDF with automatic extraction of up to 20 pages.
- Generate chunks, vector embeddings, and a knowledge graph in parallel.
- Query engine blends vector search and graph traversal with answer completeness checks.
- Streamlit interface optimized for chat-style interactions.
- Modular codebase that accelerates reuse and experimentation.

## High-Level Architecture

1. **Document ingestion** (`graphrag/document_processor.py`)  
   - Loads PDFs via `PyPDFLoader`.  
   - Splits text with `RecursiveCharacterTextSplitter`.  
   - Builds OpenAI embeddings and indexes them in FAISS.
2. **Knowledge graph** (`graphrag/knowledge_graph.py`)  
   - Extracts concepts and entities with the LLM.  
   - Creates a weighted NetworkX graph using similarity and shared concepts.
3. **Query engine** (`graphrag/query_engine.py`)  
   - Retrieves top chunks from the vector index.  
   - Expands context by traversing the graph and checking completeness.  
   - Produces the final answer through the LLM.
4. **Streamlit app** (`app.py`)  
   - Manages session state and asynchronous execution with `ThreadPoolExecutor`.  
   - Displays chat history with `streamlit-chat`.

## Prerequisites

- Python 3.12+
- OpenAI API key with access to embedding and chat models
- Conda or virtualenv (recommended)

## Environment Setup

```bash
conda create --name graphrag-contracts python=3.12
conda activate graphrag-contracts
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Add a Streamlit secrets file at `.streamlit/secrets.toml`:

```toml
[general]
API_KEY = "your-openai-api-key"
```

## Run the App

```bash
streamlit run app.py
```

After uploading a PDF, type your question in the input box. The main panel keeps the conversation history.

## Project Structure

```
.
├── app.py                       # Streamlit interface
├── graphrag
│   ├── document_processor.py    # Chunking + embedding + FAISS pipeline
│   ├── graph_rag.py             # Orchestrates the GraphRAG flow
│   ├── knowledge_graph.py       # Knowledge graph construction
│   └── query_engine.py          # Hybrid search and reasoning engine
├── requirements.txt
├── README.md
└── LEIAME.txt                   # Quickstart guide (English)
```

## Customization Tips

- Tune `chunk_size` and `chunk_overlap` in `DocumentProcessor` to adjust chunking granularity.
- Modify `edges_threshold` in `KnowledgeGraph` to control graph density.
- Swap embedding or chat models by updating parameters in `OpenAIEmbedding`.
- Adjust `self.max_content_length` in `QueryEngine` to constrain the context sent to the LLM.

## Suggested Next Steps

1. Persist FAISS indexes to disk for faster reloads.
2. Add authentication to the Streamlit UI.
3. Build automated tests to cover the critical components.

---

Feel free to adapt this project and reference it in posts or presentations. It is designed to accelerate GraphRAG proofs of concept for document analysis.
