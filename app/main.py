from app.schemas import QuestionRequest
from app.rag import generate_answer

from app.ingest import create_vectorstore

create_vectorstore("data/paper.pdf")

from fastapi import FastAPI

app = FastAPI(
    Title = "AI Reseacher Asistant"
)

@app.get("/")
def home():
    return {
        "message": "Wellcome to AI Research Assistant"
    }

@app.get("/ask")
def ask(question: QuestionRequest):
    
    return generate_answer(
        question
    )