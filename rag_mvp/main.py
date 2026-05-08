"""
main.py
-------
Ponto de entrada do projeto RAG MVP.

Uso:
    # Indexar um documento pela primeira vez
    python main.py --indexar caminho/para/documento.pdf

    # Fazer perguntas sobre o documento já indexado
    python main.py

    # Reindexar um documento diferente
    python main.py --indexar caminho/para/outro.txt
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Adiciona o diretório src ao path
sys.path.append(str(Path(__file__).parent))

from src.chunker  import processar_documento
from src.embedder import gerar_embeddings, gerar_embedding_query
from src.retriever import indexar, buscar, indice_existe
from src.generator import gerar_resposta

console = Console()


def pipeline_indexacao(caminho: str) -> None:
    """Executa o pipeline completo de indexação de um documento."""

    console.print(Panel(
        f"[bold cyan]Indexando documento:[/bold cyan] {caminho}",
        title="RAG MVP — Indexação"
    ))

    # Etapa 1: Chunking
    console.print("\n[yellow]Etapa 1/3 — Dividindo documento em chunks...[/yellow]")
    chunks = processar_documento(caminho, tamanho=1000, sobreposicao=100)#processar_documento(caminho, tamanho=500, sobreposicao=50)
    console.print(f"[green]✓[/green] {len(chunks)} chunks criados")

    # Etapa 2: Embeddings
    console.print("\n[yellow]Etapa 2/3 — Gerando embeddings...[/yellow]")
    embeddings = gerar_embeddings(chunks)
    console.print(f"[green]✓[/green] Embeddings gerados: shape {embeddings.shape}")

    # Etapa 3: Indexação FAISS
    console.print("\n[yellow]Etapa 3/3 — Indexando no FAISS...[/yellow]")
    indexar(embeddings, chunks)
    console.print("[green]✓[/green] Índice salvo em storage/")

    console.print(Panel(
        "[bold green]Documento indexado com sucesso![/bold green]\n"
        "Execute [cyan]python main.py[/cyan] para começar a fazer perguntas.",
        title="Concluído"
    ))


def pipeline_qa() -> None:
    """Loop interativo de perguntas e respostas."""

    console.print(Panel(
        "[bold cyan]RAG MVP[/bold cyan] — Perguntas e Respostas\n"
        "Digite [bold]sair[/bold] para encerrar.",
        title="RAG MVP"
    ))

    while True:
        console.print()
        query = console.input("[bold yellow]Sua pergunta:[/bold yellow] ").strip()

        if query.lower() in ("sair", "exit", "quit"):
            console.print("\n[dim]Encerrando...[/dim]")
            break

        if not query:
            continue

        # Etapa 1: Embedding da query
        console.print("[dim]Buscando trechos relevantes...[/dim]")
        embedding_query = gerar_embedding_query(query)

        # Etapa 2: Retrieval
        chunks_recuperados = buscar(embedding_query, top_k=3)
        # Adiciona temporariamente no main.py após o retrieval
        for i, item in enumerate(chunks_recuperados, 1):
            print(f"\n=== CHUNK COMPLETO {i} ===")
            print(item["chunk"])
            print(f"Tamanho: {len(item['chunk'])} caracteres")

        # Exibe os chunks recuperados
        console.print("\n[bold blue]Trechos recuperados:[/bold blue]")
        for i, item in enumerate(chunks_recuperados, 1):
            preview = item["chunk"][:120].replace("\n", " ")
            console.print(
                f"  [dim][{i}] score={item['score']:.4f}[/dim] → {preview}..."
            )

        # Etapa 3: Geração
        console.print("\n[dim]Gerando resposta com Llama 3.1...[/dim]")
        resposta = gerar_resposta(query, chunks_recuperados)

        # Exibe resposta
        console.print(Panel(
            Text(resposta, style="white"),
            title="[bold green]Resposta[/bold green]",
            border_style="green"
        ))


def main():
    parser = argparse.ArgumentParser(description="RAG MVP com Llama 3.1 local")
    parser.add_argument(
        "--indexar",
        metavar="CAMINHO",
        help="Caminho para o documento (.pdf ou .txt) a ser indexado"
    )
    args = parser.parse_args()

    if args.indexar:
        pipeline_indexacao(args.indexar)
        return

    if not indice_existe():
        console.print(
            "[bold red]Nenhum índice encontrado.[/bold red]\n"
            "Indexe um documento primeiro com:\n"
            "[cyan]python main.py --indexar caminho/para/documento.pdf[/cyan]"
        )
        return

    pipeline_qa()


if __name__ == "__main__":
    main()
