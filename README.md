# Mahabharata RAG

A retrieval-augmented generation (RAG) system for querying the complete Mahabharata (KMG English translation) using natural language.

## Overview
- **Corpus**: 18 Parvas, 2110 sections, ~1.8M words (Kisari Mohan Ganguli translation, public domain)
- **Stack**: Python, LangChain, ChromaDB, Sentence Transformers, Ollama
- **Status**: In progress

## Project Structure
ingestion/   # Parsing and embedding pipeline
retrieval/   # Query and reranking logic
generation/  # Prompt templates and LLM integration
evaluation/  # RAGAS eval harness
api/         # FastAPI wrapper
ui/          # Chat frontend

## Setup
```bash
git clone https://github.com/PrajjwalDewangan/mahabharata-rag
cd mahabharata-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```