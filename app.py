from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import ask_question

app = FastAPI()


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


@app.post("/ask")
def ask(query: Query):

    result = ask_question(
        query.question
    )

    return result