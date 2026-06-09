from app.schemas import QuestionRequest
from app.rag import generate_answer

from fastapi import FastAPI

app = FastAPI(
    title = "AI Reseacher Asistant"
)

@app.get("/")
def home():
    return {
        "message": "Wellcome to AI Research Assistant"
    }

@app.post("/ask")
def ask(request: QuestionRequest):
    
    return generate_answer(
        request.question
    )