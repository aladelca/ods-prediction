"""Reusable feature transformers for text experiments."""

from __future__ import annotations

import re

import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin


class Word2VecEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Train Word2Vec on training corpus and return mean embeddings per document."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 20,
        workers: int = 1,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9]+", str(text).lower())

    def fit(self, X, y=None):  # noqa: N803
        tokenized = [self._tokenize(t) for t in X]
        self.model_ = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
        )
        return self

    def transform(self, X):  # noqa: N803
        embeddings = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, text in enumerate(X):
            tokens = self._tokenize(text)
            vectors = [self.model_.wv[t] for t in tokens if t in self.model_.wv]
            if vectors:
                embeddings[i] = np.mean(vectors, axis=0)
        return embeddings
