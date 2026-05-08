"""
chunker.py
----------
Responsável por carregar e dividir documentos em chunks menores.

Por que fazer chunking?
- Embeddings funcionam melhor em trechos curtos e focados
- Chunks menores = retrieval mais preciso
- LLMs têm limite de contexto — não dá para passar o documento inteiro
"""

from pypdf import PdfReader
from pathlib import Path


def carregar_texto(caminho: str) -> str:
    """Carrega texto de arquivos .txt ou .pdf."""
    path = Path(caminho)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    if path.suffix == ".pdf":
        reader = PdfReader(caminho)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Formato não suportado: {path.suffix}. Use .pdf ou .txt")


def criar_chunks(texto: str, tamanho: int = 500, sobreposicao: int = 50) -> list[str]:
    """
    Divide o texto em chunks com sobreposição.

    Parâmetros:
        tamanho      : número de caracteres por chunk
        sobreposicao : caracteres compartilhados entre chunks adjacentes

    Por que sobreposição?
        Evita que informações importantes sejam cortadas na fronteira
        entre dois chunks. Um chunk sempre "vê" o final do anterior.
    """

    # Divide por parágrafo primeiro
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    
    chunks = []
    chunk_atual = ""
    
    for paragrafo in paragrafos:
        # Se adicionar esse parágrafo não ultrapassa o limite, agrupa
        if len(chunk_atual) + len(paragrafo) <= tamanho:
            chunk_atual += "\n\n" + paragrafo
        else:
            if chunk_atual:
                chunks.append(chunk_atual.strip())
            chunk_atual = paragrafo
    
    if chunk_atual:
        chunks.append(chunk_atual.strip())
    
    return chunks


def processar_documento(caminho: str, tamanho: int = 500, sobreposicao: int = 50) -> list[str]:
    """Pipeline completo: carrega o arquivo e retorna lista de chunks."""
    texto = carregar_texto(caminho)
    chunks = criar_chunks(texto, tamanho=tamanho, sobreposicao=sobreposicao)
    return chunks
