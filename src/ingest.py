import os
import pickle
import pypdf
import faiss

from src.chunker import chunk_text
from src.embedder import create_embeddings
from src.vector_store import create_index


reader = pypdf.PdfReader(
    "data/dbms_notes.pdf"
)

full_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:
        full_text += text + "\n"


chunks = chunk_text(
    full_text
)

embeddings = create_embeddings(
    chunks
)

index = create_index(
    embeddings
)

os.makedirs(
    "vectorstore",
    exist_ok=True
)

with open(
    "vectorstore/chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )

print("Ingestion Complete")
print("Chunks:", len(chunks))