"""
retriever.py
------------
Responsável por indexar embeddings no FAISS e realizar busca semântica.

O que é FAISS?
- Biblioteca do Meta para busca eficiente em vetores de alta dimensão
- Em produção seria substituído por Pinecone, Chroma ou Weaviate
- Para o MVP local, é perfeito: rápido, sem custo, sem servidor

Fluxo:
    1. Indexar: recebe embeddings dos chunks e salva no índice FAISS
    2. Buscar: recebe embedding da query e retorna os chunks mais similares
"""

import faiss
import numpy as np
import pickle
from pathlib import Path


STORAGE_DIR = Path("storage")
INDEX_PATH  = STORAGE_DIR / "faiss.index"
CHUNKS_PATH = STORAGE_DIR / "chunks.pkl"


def indexar(embeddings: np.ndarray, chunks: list[str]) -> None:
    """
    Cria o índice FAISS e salva junto com os chunks originais.

    Parâmetros:
        embeddings : matriz (n_chunks, 384) gerada pelo embedder
        chunks     : lista de textos originais correspondentes
    """
    STORAGE_DIR.mkdir(exist_ok=True)

    dimensao = embeddings.shape[1]

    # IndexFlatIP: busca por produto interno (equivalente a cosseno para vetores normalizados)
    indice = faiss.IndexFlatIP(dimensao)

    # Normaliza os vetores para que o produto interno = similaridade de cosseno
    faiss.normalize_L2(embeddings)
    indice.add(embeddings)

    # Salva o índice e os chunks no disco
    faiss.write_index(indice, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def buscar(embedding_query: np.ndarray, top_k: int = 3) -> list[dict]:
    """
    Busca os top_k chunks mais similares à query no índice FAISS.

    Retorna:
        Lista de dicts com 'chunk' e 'score' ordenados por relevância
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError("Índice não encontrado. Execute a indexação primeiro.")

    indice = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Normaliza a query para cosseno
    query = embedding_query.reshape(1, -1).astype("float32")
    faiss.normalize_L2(query)

    scores, indices = indice.search(query, top_k)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            resultados.append({
                "chunk": chunks[idx],
                "score": float(score)
            })

    return resultados


def indice_existe() -> bool:
    """Verifica se já existe um índice salvo no disco."""
    return INDEX_PATH.exists() and CHUNKS_PATH.exists()
