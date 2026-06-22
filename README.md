# Retrieval-Augmented-Generation-Chatbot

A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on document context.

## How it works
1. Documents are loaded and chunked
2. Relevant chunks are retrieved based on the user query
3. An LLM generates a response using the retrieved context

## Setup
```bash
pip install -r requirements.txt
python app.py
```
