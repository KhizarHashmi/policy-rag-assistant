"""Embedding backends behind one small interface.

GeminiEmbedder is used in production. HashEmbedder is a deterministic, offline stand-in so
tests and CI run without an API key or network access (it is not meant to be good).
"""
import hashlib
import math
import re
from typing import Protocol

from .config import settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Asymmetric retrieval embeddings: documents and queries use different task types."""

    def __init__(self, model: str = settings.embedding_model, dim: int = settings.embedding_dim):
        from google import genai
        from google.genai import types

        self._client, self._types, self.model, self.dim = genai.Client(), types, model, dim

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 50):  # batch to stay under request limits
            resp = self._client.models.embed_content(
                model=self.model, contents=texts[i:i + 50],
                config=self._types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim))
            out += [_normalise(e.values) for e in resp.embeddings]  # truncated dims must be re-normalised
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class HashEmbedder:
    """Feature-hashed bag of words. Offline, deterministic, for tests only."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _normalise(v)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _normalise(v) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]
