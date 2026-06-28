import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.rag import ask_question
from src.ingest import ingest_pdf

app = FastAPI()

# Create uploads folder if it doesn't exist
os.makedirs("uploads", exist_ok=True)


class Query(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "RAG API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    filepath = os.path.join(
        "uploads",
        file.filename
    )

    with open(filepath, "wb") as f:
        f.write(await file.read())

    try:
        chunks_added = ingest_pdf(filepath)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    return {
        "message": "PDF indexed successfully",
        "filename": file.filename,
        "chunks_added": chunks_added
    }



@app.post("/ask")
def ask(query: Query):

    result = ask_question(query.question)

    return result