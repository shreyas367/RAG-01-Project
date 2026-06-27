import os
import pickle
import pypdf
import faiss

from src.chunker import chunk_text
from src.embedder import create_embeddings
from src.vector_store import create_index

# -----------------------------
# Load PDF
# -----------------------------
reader = pypdf.PdfReader("data/dbms_notes.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"

# -----------------------------
# Chunk Text
# -----------------------------
chunks = chunk_text(full_text)

# -----------------------------
# Create Embeddings
# -----------------------------
embeddings = create_embeddings(chunks)

# -----------------------------
# Create FAISS Index
# -----------------------------
index = create_index(embeddings)

# -----------------------------
# Create vectorstore folder
# -----------------------------
os.makedirs("vectorstore", exist_ok=True)

# -----------------------------
# Save FAISS Index
# -----------------------------
faiss.write_index(
    index,
    "vectorstore/dbms.index"
)

# -----------------------------
# Save Chunks
# -----------------------------
with open(
    "vectorstore/chunks.pkl",
    "wb"
) as f:
    pickle.dump(chunks, f)

print("\n✅ Ingestion Complete")
print("Chunks:", len(chunks))
print("FAISS Index Saved")
print("Chunks Saved")