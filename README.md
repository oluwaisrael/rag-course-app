# UniRAG: Advanced Layout-Aware Hybrid RAG System

A production-grade Retrieval-Augmented Generation (RAG) application built with Python and Streamlit, fully containerized using Docker. This system implements a dual-index architecture with intelligent layout-aware document chunking and cross-encoder re-ranking to deliver highly precise context matching.

## Architecture Overview



* **Layout-Aware Document Ingestion:** Custom parsing pipeline that maintains document hierarchy and contextual proximity rather than naive character splitting.
* **Dual-Index Hybrid Retrieval:** * **Dense Retrieval:** FAISS index utilizing semantic vector embeddings via `sentence-transformers`.
* **Sparse Retrieval:** BM25 lexical keyword matching to capture exact terminology, course codes, and phrases.
* **Cross-Encoder Re-ranking:** Merges and scores candidate chunks using a deep learning cross-encoder model to maximize relevance before context window delivery.
* **LLM Generation:** Integrates with the Google GenAI SDK using Gemini models for grounded, factual reasoning.

## Tech Stack
* **Language & UI:** Python, Streamlit
* **Vector Search & NLP:** FAISS, SentenceTransformers, Rank-BM25
* **LLM API:** Google GenAI SDK (Gemini)
* **DevOps & Containerization:** Docker (Multi-stage layer architecture optimized for ARM64/Apple Silicon CPU)

## Getting Started with Docker

### Prerequisites
* Docker Desktop installed
* Gemini API Key

### Build the Image
```bash
docker build -t unirag-app .