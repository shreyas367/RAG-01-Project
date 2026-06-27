import pickle
import faiss

from src.embedder import model
from src.retriever import search
from src.llm import generate_answer


# Load once when the application starts
index = faiss.read_index("vectorstore/dbms.index")

with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def ask_question(question: str):

    # Create query embedding
    query_embedding = model.encode(question)

    # Retrieve relevant chunks
    results = search(
        index,
        query_embedding,
        k=3
    )

    context = ""

    sources = []

    for idx in results:

        context += chunks[idx]
        context += "\n\n"

        sources.append(int(idx))

    prompt = f"""
You are a DBMS tutor.

Answer ONLY using the context below.

If the answer is not present, say:

"I could not find that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": sources
    }