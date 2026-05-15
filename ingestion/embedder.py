import numpy as np
from sentence_transformers import SentenceTransformer

import config


class Embedder:
    """Generates vector embeddings using sentence-transformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
