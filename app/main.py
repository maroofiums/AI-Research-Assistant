import shutil
from fastapi import UploadFile, File, FastAPI
from app.schemas import QuestionRequest
from app.rag import generate_answer
from app.ingest import create_vectorstore

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title = "AI Reseacher Asistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    path = f"data/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = create_vectorstore(path)

    return {
        "message": "Document uploaded",
        "chunks": chunks
    }