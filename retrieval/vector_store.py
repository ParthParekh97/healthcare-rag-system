import os
import json
import numpy as np
import faiss

import config
from ingestion.chunker import Chunk


class VectorStore:
    """FAISS-based vector store for semantic retrieval."""

    def __init__(self, dimension: int = None):
        self.dimension = dimension or config.EMBEDDING_DIMENSION
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunk count and embedding count must match")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = None) -> list[dict]:
        top_k = top_k or config.TOP_K_RETRIEVAL
        top_k = min(top_k, self.index.ntotal)

        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk": chunk,
                "score": float(score),
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })

        return results

    def save(self, path: str = None):
        path = path or config.INDEX_DIR
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "faiss.index"))

        chunk_data = []
        for chunk in self.chunks:
            chunk_data.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
            })
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(chunk_data, f, indent=2)

    def load(self, path: str = None) -> bool:
        path = path or config.INDEX_DIR
        index_path = os.path.join(path, "faiss.index")
        chunks_path = os.path.join(path, "chunks.json")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return False

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "r") as f:
            chunk_data = json.load(f)

        self.chunks = []
        for item in chunk_data:
            self.chunks.append(Chunk(
                text=item["text"],
                metadata=item["metadata"],
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
            ))
        return True

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
