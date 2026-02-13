import os
from typing import List

import numpy as np
import requests

from config import EMBEDDING_MODEL


def _fallback_embed(texts: List[str], dim: int = 512) -> np.ndarray:
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.lower().split():
            idx = hash(token) % dim
            vectors[i, idx] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return (vectors / norms).astype(np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        return _fallback_embed(texts)

    url = "https://api.jina.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": texts}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        arr = np.array([row["embedding"] for row in data], dtype=np.float32)
        return arr
    except Exception:
        return _fallback_embed(texts)
