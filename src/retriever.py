import faiss
import numpy as np

def search(index, query_embedding, k=3):

    query_embedding = np.array(
        [query_embedding],
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        k
    )

    return indices[0]