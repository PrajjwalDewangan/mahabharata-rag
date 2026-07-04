# Mahabharata RAG

A retrieval-augmented generation system over the Kisari Mohan Ganguli (KMG) English translation of the Mahabharata, built to answer questions grounded strictly in the source text with citations.

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/PrajjwalDewangan/mahabharata-rag)

## Overview

This project ingests the full 18-parva KMG translation, chunks it into retrievable passages, and serves a conversational Q&A interface backed by hybrid retrieval and an open-weight LLM. It also includes a custom evaluation harness used to measure and improve retrieval and generation quality over multiple iterations.

## Pipeline

### 1. Ingestion (`parse.py`, `clean.py`, `chunk.py`)
- Parses raw per-parva text files into structured sections, handling two different heading formats (Roman numeral `SECTION`s vs. numeric headings depending on parva).
- Strips formatting artifacts (footnotes, "continued" markers, end-of-parva markers) using a set of regex-based rules.
- Chunks cleaned sections into **13,800+ passages** (40–300 words), merging small paragraphs and splitting oversized ones at sentence boundaries to keep chunks semantically coherent.

### 2. Embedding & Storage (`embed.py`)
- Embeds all chunks using `all-mpnet-base-v2` (Hugging Face `sentence-transformers`).
- Stores vectors and metadata (parva, section, chunk index) in a persistent **ChromaDB** collection (cosine similarity).

### 3. Retrieval (`retriever.py`)
- **Hybrid search:** dense retrieval (ChromaDB) + BM25 lexical search, merged via **Reciprocal Rank Fusion (RRF)**.
- **Reranking:** top RRF candidates reranked with a `ms-marco-MiniLM-L-6-v2` cross-encoder.
- **Multi-query expansion:** the production LLM (Groq) generates alternate phrasings of the query, with awareness of character epithets (e.g., Arjuna = Dhananjaya = Partha) — and results across all variants are merged via RRF before reranking.

### 4. Generation (`generator.py`)
- Answers are generated with **Llama-3.1-8B-Instant** via **Groq** in production, **Ollama** (`gemma3:4b`) for local dev.
- A system prompt enforces strict grounding: no outside knowledge, mandatory `[Parva Name, Section X]` citations, explicit refusal when passages are insufficient, and no fabricated quotes.
- **Multi-turn support:** a sliding window of the last 3 conversation turns is included in every prompt, enabling coherent follow-up questions.

### 5. Deployment
- FastAPI backend, containerized with Docker, deployed on Hugging Face Spaces.
- The ChromaDB vector store is persisted to the Space using Git LFS.

## Evaluation

Two custom harnesses (`eval_retrieval.py`, `eval_generation.py`) run against a curated 19-query test set spanning categories: factual events, character deaths, descriptive, motivational, encyclopedic, and ambiguous queries.

**Retrieval — Recall@k**
Multi-query expansion improved **Recall@5 from 55.6% to 66.7%**.

**Generation — LLM-as-judge (Groq, temperature 0)**
Scored on a 1–5 scale for:
- **Faithfulness** — every claim traceable to retrieved passages
- **Answer relevance** — does the answer address what was asked

| Metric | Score |
|---|---|
| Avg. Faithfulness | 4.06 / 5 |
| Avg. Relevance | 4.11 / 5 |

### Notable findings from evaluation
- **Hallucination caught:** on "How did Bhima die?" the model fabricated multiple unsupported death scenarios (e.g., inverting a passage where Bhima *kills* a warrior into a claim that the warrior poisoned him): scored **1/5 faithfulness**, the lowest in the test set.
- **Faithful-but-unhelpful refusals:** on queries like "How did Arjuna die?" and "Ashwatthama the elephant," the model correctly refused to answer when the right chunk wasn't retrieved (high faithfulness) but was scored **1/5 relevance** for not answering: a real tension between grounded caution and being useful, not a bug in either metric.
- **Late-parva retrieval gaps:** queries about deaths narrated in the epic's final parvas (Mahaprasthanika, Mausala) were more likely to fail retrieval, plausibly due to lower representation of those parvas relative to the war-heavy middle parvas.
- **Judge blind spot:** a genuinely wrong answer (Kunti listed as a Pandava instead of Bhima) still scored faithfulness 4, since the incorrect names were individually grounded in other retrieved passages, a limitation of faithfulness-only judging for enumeration-style questions.

## Known Limitations
- Enumeration-style questions (e.g., "list all parvas") are a poor fit for RAG without a structured lookup layer: the model tends to conflate major parvas with sub-parvas because retrieved chunks list them together.
- Some KMG translation sections contain near-duplicate passages, which can skew retrieved context.
- Retrieval quality is not uniform across all parvas; later, shorter parvas are more failure-prone.

## Tech Stack
Python · ChromaDB · Sentence-Transformers (`all-mpnet-base-v2`) · `rank_bm25` · Cross-Encoder reranking · Groq (Llama-3.1-8B-Instant) · Ollama · FastAPI · Docker · Hugging Face Spaces
