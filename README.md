
# RAG-01-Project
# Multi-Document RAG Assistant

## Features

- PDF ingestion
- Text chunking
- SentenceTransformer embeddings
- FAISS vector search
- Ollama integration
- Local LLM inference

## Tech Stack

- Python
- SentenceTransformers
- FAISS
- Ollama
- Llama3 / Phi3

## Usage

1. Put PDFs into data/
2. Run ingestion
3. Start chatbot

```bash
uv run python -m src.ingest
uv run python main.py
