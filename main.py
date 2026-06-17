import pypdf

from src.chunker import chunk_text
from src.embedder import create_embeddings, model
from src.vector_store import create_index
from src.retriever import search
from src.llm import generate_answer


# -----------------------------
# 1. Load PDF
# -----------------------------
reader = pypdf.PdfReader("data/dbms_notes.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"

print("Pages:", len(reader.pages))
print("Characters:", len(full_text))


# -----------------------------
# 2. Chunk Text
# -----------------------------
chunks = chunk_text(full_text)

print("Chunks:", len(chunks))


# -----------------------------
# 3. Create Embeddings
# -----------------------------
embeddings = create_embeddings(chunks)

print("Embedding Shape:", embeddings.shape)
print("First Vector Length:", len(embeddings[0]))


# -----------------------------
# 4. Create FAISS Index
# -----------------------------
index = create_index(embeddings)

print("Vectors Stored:", index.ntotal)


# -----------------------------
# 5. User Query
# -----------------------------
questions = [
    "What is DBMS?",
    "What is relational algebra?",
    "What is relational calculus?",
    "What are procedural query languages?",
    "What are non procedural query languages?",
    "Explain select operation.",
    "Explain project operation.",
    "What is tuple relational calculus?",
    "What is domain relational calculus?",
    "Explain EF Codd's 12 rules."
]
for query in questions:

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(query)

    # Embed Query
    query_embedding = model.encode(query)

    # Retrieve Chunks
    results = search(
        index,
        query_embedding,
        k=3
    )

    context = ""

    print("\nTOP MATCHING CHUNKS:\n")

    for idx in results:

        print("=" * 60)
        print(chunks[idx][:300])
        print()

        context += chunks[idx]
        context += "\n\n"

    # Build Prompt
    prompt = f"""
You are a DBMS tutor.

Answer ONLY from the provided context.

If the answer is not present in the context, say:
"I could not find that information in the provided documents."

Context:
{context}

Question:
{query}

Provide a clear and concise answer.

Answer:
"""

    # Generate Answer
    answer = generate_answer(prompt)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)




