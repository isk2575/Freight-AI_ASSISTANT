#  Freight AI Assistant

An AI-powered freight and logistics assistant built with Claude API, RAG, and Gradio.

## Features

-  **Chat Mode** — Ask any freight and logistics question
-  **RAG Search** — Ask questions about saved freight documents
-  **PDF Upload** — Upload any freight PDF and extract key information

## Tech Stack

- **Claude API** (Anthropic) — AI language model
- **LlamaIndex** — RAG pipeline for document search
- **PyMuPDF** — PDF reading and text extraction
- **Gradio** — Web UI
- **Python** — Backend

## How It Works

1. **Chat** — Ask Claude any freight question directly
2. **Search Documents** — LlamaIndex searches saved documents and Claude answers based on them
3. **Upload PDF** — PyMuPDF extracts text from any PDF, Claude analyzes it

## Setup

1. Clone the repo
2. Install dependencies:(pip install anthropic llama-index llama-index-llms-anthropic llama-index-embeddings-huggingface pymupdf gradio python-dotenv)
3. Create a `.env` file:(ANTHROPIC_API_KEY=your-key-here)
4.  Run the app:
