# AI Research Assistant

A Retrieval-Augmented Generation (RAG) application built with FastAPI, ChromaDB, LangChain, HuggingFace Embeddings, and Gemini.

The system allows users to ask questions about PDF documents by retrieving relevant document chunks and generating grounded answers using an LLM.

---

## Features

* PDF document ingestion
* Automatic text chunking
* Local embeddings using MiniLM
* ChromaDB vector storage
* Semantic document retrieval
* Gemini-powered answer generation
* Source page citations
* FastAPI REST API
* Swagger API documentation

---

## Tech Stack

### Backend

* FastAPI
* Uvicorn

### RAG Framework

* LangChain

### Embeddings

* all-MiniLM-L6-v2
* HuggingFace Embeddings

### Vector Database

* ChromaDB

### LLM

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
├── Readme.md
└── requirements.txt
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/AI-Research-Assistant.git

cd AI-Research-Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
MISTRAL_API_KEY=your_api_key_here
```

Get a Gemini API key from Google AI Studio.

---

## Building the Vector Database

Place your PDF inside the `data` folder.

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

This will:

1. Load the PDF
2. Split text into chunks
3. Generate embeddings
4. Store vectors in ChromaDB

Output:

```text
Vector store created successfully.
```

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Server:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Usage

### Ask Question

Endpoint:

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
      "source": "data/deeplearning.pdf"
    }
  ]
}
```

---

## RAG Pipeline

```text
PDF
 ↓
Document Loader
 ↓
Text Splitter
 ↓
MiniLM Embeddings
 ↓
ChromaDB
 ↓
Retriever
 ↓
Prompt Template
 ↓
Mistral
 ↓
Answer + Sources
```

---

## Future Improvements

* Multiple PDF support
* PDF upload endpoint
* Conversation memory
* Hybrid search (BM25 + Vector Search)
* Reranking
* Streaming responses
* LangGraph integration
* Docker deployment
* PostgreSQL + PGVector

---

## Example Workflow

1. Add a PDF to the data folder
2. Run document ingestion
3. Start FastAPI server
4. Open Swagger UI
5. Ask questions about the document
6. Receive grounded answers with citations

---

## License

This project is licensed under the MIT License.

---
