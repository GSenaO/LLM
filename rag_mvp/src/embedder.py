"""
embedder.py
-----------
Responsável por converter chunks de texto em vetores numéricos (embeddings).

Modelo usado: paraphrase-multilingual-MiniLM-L12-v2
- Suporta português nativamente
- Gera vetores de 384 dimensões
- Leve o suficiente para rodar em CPU sem demora
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# Modelo carregado uma única vez e reutilizado em todas as chamadas
_modelo = None


def _get_modelo() -> SentenceTransformer:
    """Carrega o modelo apenas na primeira chamada (lazy loading)."""
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _modelo


def gerar_embeddings(textos: list[str]) -> np.ndarray:
    """
    Converte uma lista de textos em uma matriz de embeddings.

    Retorna:
        np.ndarray de shape (n_textos, 384)
    """
    modelo = _get_modelo()
    return modelo.encode(textos, show_progress_bar=True, convert_to_numpy=True)


def gerar_embedding_query(query: str) -> np.ndarray:
    """
    Converte uma query em um único vetor de embedding.

    Retorna:
        np.ndarray de shape (384,)
    """
    modelo = _get_modelo()
    return modelo.encode([query], convert_to_numpy=True)[0]
