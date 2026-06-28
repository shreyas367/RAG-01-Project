import pickle
import faiss

from src.embedder import model
from src.retriever import search
from src.llm import generate_answer


# -----------------------------
# Load Vector Store
# -----------------------------
index = faiss.read_index("vectorstore/dbms.index")

with open("vectorstore/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


# -----------------------------
# Ask Question
# -----------------------------
def ask_question(question: str):

    # Create query embedding
    query_embedding = model.encode(question)

    # Retrieve top chunks
    results = search(
        index,
        query_embedding,
        k=5
    )

    context = ""
    sources = []

    print("\nRetrieved Chunks\n")

    for idx in results:

        chunk = metadata[idx]

        print("=" * 60)
        print(f"Document : {chunk['document']}")
        print(f"Page     : {chunk['page']}")
        print("=" * 60)
        print(chunk["text"][:500])
        print()

        context += chunk["text"]
        context += "\n\n"

        sources.append(
            {
                "document": chunk["document"],
                "page": chunk["page"]
            }
        )

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = f"""
You are an expert tutor.

Answer ONLY using the provided context.

If the information is not available in the context, reply exactly:

"I could not find that information in the provided documents."

Instructions:
- Give a detailed explanation.
- Use bullet points where appropriate.
- Include examples if available.
- Do not hallucinate.

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