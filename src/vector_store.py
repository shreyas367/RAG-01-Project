import faiss
import numpy as np


def create_index(embeddings):

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index