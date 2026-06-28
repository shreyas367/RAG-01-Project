import os
import pickle
from pydoc import text
import pypdf
import faiss
import numpy as np

from src.chunker import chunk_text
from src.embedder import create_embeddings
from src.vector_store import create_index


DATA_FOLDER = "data"
VECTORSTORE = "vectorstore"

INDEX_PATH = os.path.join(VECTORSTORE, "dbms.index")
METADATA_PATH = os.path.join(VECTORSTORE, "metadata.pkl")


# =====================================================
# FULL REBUILD
# =====================================================
def rebuild_index():
    """
    Reads all PDFs from the data folder,
    creates embeddings,
    builds a brand-new FAISS index,
    and saves metadata.
    """

    all_metadata = []

    print("\nRebuilding Vector Database...\n")

    for filename in os.listdir(DATA_FOLDER):

        if not filename.endswith(".pdf"):
            continue

        print(f"Processing: {filename}")

        reader = pypdf.PdfReader(
            os.path.join(DATA_FOLDER, filename)
        )

        for page_no, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if not text:
                continue

            chunks = chunk_text(
                text=text,
                page_number=page_no,
                document_name=filename
            )

            all_metadata.extend(chunks)

    texts = [item["text"] for item in all_metadata]

    embeddings = create_embeddings(texts)

    index = create_index(embeddings)

    os.makedirs(VECTORSTORE, exist_ok=True)

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_metadata, f)

    print("\n✅ Full Index Rebuilt")
    print("Chunks:", len(all_metadata))
    print("Vectors:", index.ntotal)


# =====================================================
# ADD ONE NEW PDF
# =====================================================
def ingest_pdf(filepath):
    """
    Reads one uploaded PDF,
    creates embeddings,
    appends vectors to FAISS,
    updates metadata.pkl.
    """

    filename = os.path.basename(filepath)

    reader = pypdf.PdfReader(filepath)
    print("Pages:", len(reader.pages))

    new_metadata = []

    print(f"\nIndexing: {filename}")

    for page_no, page in enumerate(reader.pages, start=1):

        text = page.extract_text()
        print("Page:", page_no)
        print("Extracted:", text[:200] if text else "NO TEXT")

        if not text:
            continue

        chunks = chunk_text(
            text=text,
            page_number=page_no,
            document_name=filename
        )

        new_metadata.extend(chunks)

    if not new_metadata:
        return 0

    texts = [item["text"] for item in new_metadata]

    embeddings = create_embeddings(texts)

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(embeddings)

    index = faiss.read_index(INDEX_PATH)

    index.add(embeddings)

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    metadata.extend(new_metadata)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Added {len(new_metadata)} chunks")

    return len(new_metadata)