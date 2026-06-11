# AI Research Assistant

A Retrieval-Augmented Generation (RAG) application built using FastAPI, LangChain, ChromaDB, and Mistral AI.

This project allows users to upload PDF documents and ask questions about their content. The system retrieves the most relevant document chunks from a vector database and generates grounded answers using Mistral AI.

---

## Features

* PDF document ingestion
* Automatic text chunking
* Mistral Embeddings
* ChromaDB vector storage
* Semantic search and retrieval
* Mistral-powered answer generation
* Source citations
* FastAPI REST API
* Interactive Swagger documentation

---

## Tech Stack

### Backend

* FastAPI
* Uvicorn

### RAG Framework

* LangChain

### Embeddings

* Mistral AI Embeddings

### Vector Database

* ChromaDB

### Large Language Model

* Mistral Small

### Document Processing

* PyPDF

---

## Project Structure

```text
├── app
│   ├── ingest.py
│   ├── main.py
│   ├── rag.py
│   └── schemas.py
│
├── data
│   └── deeplearning.pdf
│
├── vectorstore
│   └── chroma.sqlite3
│
├── README.md
└── requirements.txt
```

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>

cd AI-Research-Assistant
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
MISTRAL_API_KEY=your_mistral_api_key
```

---

## Build Vector Database

Place a PDF inside the `data` directory.

Example:

```text
data/
└── deeplearning.pdf
```

Run:

```python
from app.ingest import create_vectorstore

create_vectorstore("data/deeplearning.pdf")
```

This process:

1. Loads the PDF
2. Splits text into chunks
3. Generates embeddings
4. Stores vectors in ChromaDB

---

## Run the Application

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

Server URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Home

```http
GET /
```

Response:

```json
{
  "message": "Welcome to AI Research Assistant"
}
```

---

### Upload PDF

```http
POST /upload
```

Upload a PDF file to create embeddings and store them in ChromaDB.

Response:

```json
{
  "message": "Document uploaded",
  "chunks": 125
}
```

---

### Ask Questions

```http
POST /ask
```

Request:

```json
{
  "question": "What is backpropagation?"
}
```

Response:

```json
{
  "answer": "Backpropagation is a training algorithm used to update neural network weights by propagating errors backward through the network.",
  "sources": [
    {
      "page": 12,
      "source": "data/deeplearning.pdf",
      "preview": "Backpropagation computes gradients..."
    }
  ]
}
```

---

## RAG Pipeline

```text
PDF
 ↓
PyPDFLoader
 ↓
RecursiveCharacterTextSplitter
 ↓
Mistral Embeddings
 ↓
ChromaDB
 ↓
Retriever
 ↓
Prompt Template
 ↓
Mistral Small
 ↓
Answer + Sources
```

---

## Example Workflow

1. Upload a PDF document
2. Generate embeddings
3. Store vectors in ChromaDB
4. Ask questions
5. Retrieve relevant chunks
6. Generate grounded answers
7. Return answer with citations

---

## Future Improvements

* Multi-document support
* Chat history
* Streaming responses
* Hybrid Search (BM25 + Vector Search)
* Reranking
* LangGraph integration
* Docker deployment
* PostgreSQL + PGVector
* User authentication

---

## Known Issues

* Large PDFs may take time to embed.
* Mistral Embeddings require a valid API key.
* Network interruptions may cause embedding failures during ingestion.

---
