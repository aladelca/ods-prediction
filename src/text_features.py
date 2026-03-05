"""Reusable text preprocessing and embedding transformers for notebooks/pipelines."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
except Exception:  # pragma: no cover - optional at import time
    nltk = None
    stopwords = None
    SnowballStemmer = None

try:
    import spacy
except Exception:  # pragma: no cover - optional at import time
    spacy = None


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Configurable text preprocessor for Spanish NLP tasks."""

    def __init__(
        self,
        lowercase: bool = True,
        strip_accents: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        stem: bool = False,
        min_token_len: int = 2,
        stopword_language: str = "spanish",
        spacy_model_name: str = "es_core_news_sm",
        nltk_data_dir: str = "nltk_data",
    ) -> None:
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_token_len = min_token_len
        self.stopword_language = stopword_language
        self.spacy_model_name = spacy_model_name
        self.nltk_data_dir = nltk_data_dir

    def fit(self, X, y=None):  # noqa: N803
        if nltk is None or stopwords is None or SnowballStemmer is None:
            raise ImportError(
                "nltk is required for TextPreprocessor.fit. Install with: pip install nltk"
            )

        data_dir = Path(self.nltk_data_dir).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        if str(data_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(data_dir))

        nltk.download("stopwords", download_dir=str(data_dir), quiet=True)
        self._stopwords = set(stopwords.words(self.stopword_language))
        self._stemmer = SnowballStemmer(self.stopword_language)

        if self.lemmatize:
            if spacy is None:
                raise ImportError(
                    "spacy is required when lemmatize=True. Install with: pip install spacy"
                )
            try:
                self._nlp = spacy.load(self.spacy_model_name, disable=["parser", "ner"])
            except OSError:
                from spacy.cli import download

                download(self.spacy_model_name)
                self._nlp = spacy.load(self.spacy_model_name, disable=["parser", "ner"])
        else:
            self._nlp = None

        return self

    def _normalize(self, text: str) -> str:
        text = str(text)
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = "".join(
                c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
            )
        return text

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-ZñÑüÜ0-9]+", text)

    def transform(self, X):  # noqa: N803
        processed = []
        for text in X:
            text = self._normalize(text)
            toks = self._tokenize(text)
            toks = [t for t in toks if len(t) >= self.min_token_len]

            if self.remove_stopwords:
                toks = [t for t in toks if t not in self._stopwords]

            if self.lemmatize and self._nlp is not None:
                doc = self._nlp(" ".join(toks))
                toks = [tok.lemma_.strip() for tok in doc if tok.lemma_.strip()]

            if self.stem:
                toks = [self._stemmer.stem(t) for t in toks]

            processed.append(" ".join(toks))
        return processed


class MeanGensimEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Train Word2Vec/FastText and return mean embedding per document."""

    def __init__(
        self,
        method: str = "word2vec",
        vector_size: int = 150,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 20,
        workers: int = 1,
        seed: int = 42,
    ) -> None:
        self.method = method
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

    def fit(self, X, y=None):  # noqa: N803
        try:
            from gensim.models import FastText, Word2Vec
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "gensim is required for MeanGensimEmbeddingTransformer. Install with: pip install gensim"
            ) from exc

        tokenized = [str(x).split() for x in X]
        if self.method == "fasttext":
            self.model_ = FastText(
                sentences=tokenized,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs,
                workers=self.workers,
                seed=self.seed,
            )
        else:
            self.model_ = Word2Vec(
                sentences=tokenized,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs,
                workers=self.workers,
                seed=self.seed,
            )
        return self

    def transform(self, X):  # noqa: N803
        vectors = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, text in enumerate(X):
            toks = str(text).split()
            vecs = [self.model_.wv[t] for t in toks if t in self.model_.wv]
            if vecs:
                vectors[i] = np.mean(vecs, axis=0)
        return vectors


class HFEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Sentence-transformers embedding extractor (PyTorch backend)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        use_cache: bool = True,
        cache_dir: str = "cache/hf_embeddings",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.use_cache = use_cache
        self.cache_dir = cache_dir

    def fit(self, X, y=None):  # noqa: N803
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for HFEmbeddingTransformer. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model_ = SentenceTransformer(self.model_name, device=self.device)
        return self

    def _build_cache_path(self, X):  # noqa: N803
        hasher = hashlib.sha1()
        hasher.update(self.model_name.encode("utf-8"))
        hasher.update(str(self.normalize_embeddings).encode("utf-8"))
        hasher.update(str(len(X)).encode("utf-8"))
        for text in X:
            s = str(text)
            hasher.update(s.encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
        filename = f"{hasher.hexdigest()}.npy"
        return Path(self.cache_dir) / filename

    def transform(self, X):  # noqa: N803
        texts = list(map(str, X))

        if self.use_cache:
            cache_path = self._build_cache_path(texts)
            if cache_path.exists():
                return np.load(cache_path).astype(np.float32)

        if not hasattr(self, "model_"):
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "sentence-transformers is required for HFEmbeddingTransformer. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self.model_ = SentenceTransformer(self.model_name, device=self.device)

        emb = self.model_.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        emb = emb.astype(np.float32)

        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, emb)

        return emb
