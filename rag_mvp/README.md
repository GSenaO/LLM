# RAG MVP — Retrieval-Augmented Generation Local

Pipeline RAG completo rodando localmente com Llama 3.1, sem custo de API.

## O problema que este projeto resolve

Modelos de linguagem locais tendem a alucinar quando não conhecem a resposta:

```
Pergunta: "O que é RAG?"
Llama sem RAG: "RAG significa Regras de Aprendizado..." ← alucinação
Llama com RAG: resposta baseada no documento fornecido  ← correto
```

## Arquitetura

```
Documento (.pdf / .txt)
        ↓
   chunker.py      → divide em trechos de 500 caracteres com sobreposição
        ↓
   embedder.py     → converte chunks em vetores de 384 dimensões
        ↓
   retriever.py    → indexa vetores no FAISS (vector store local)
        ↓
   [query do usuário]
        ↓
   embedder.py     → converte query em vetor
   retriever.py    → busca os 3 chunks mais similares (similaridade de cosseno)
        ↓
   generator.py    → monta prompt com contexto e envia ao Llama via Ollama
        ↓
   Resposta fundamentada no documento
```

## Stack

| Componente | Tecnologia |
|-----------|------------|
| Embeddings | `sentence-transformers` (MiniLM multilingual) |
| Vector store | `FAISS` (Meta) |
| LLM | `Llama 3.1 8B` via `Ollama` |
| Interface | Terminal com `rich` |

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado com o modelo `llama3.1`

```bash
ollama pull llama3.1
```

## Instalação

```bash
git clone https://github.com/seu-usuario/rag-mvp
cd rag-mvp
pip install -r requirements.txt
```

## Uso

**1. Indexar um documento:**

```bash
python main.py --indexar data/seu_documento.pdf
```

**2. Fazer perguntas:**

```bash
python main.py
```

## Estrutura do projeto

```
rag_mvp/
├── src/
│   ├── chunker.py    # Carregamento e divisão de documentos
│   ├── embedder.py   # Geração de embeddings
│   ├── retriever.py  # Indexação e busca no FAISS
│   └── generator.py  # Integração com Ollama / Llama
├── data/             # Coloque seus documentos aqui
├── storage/          # Índice FAISS gerado automaticamente
├── main.py           # Ponto de entrada
├── requirements.txt
└── README.md
```

## Próximos passos (Projeto 2)

- Interface web com Gradio
- Métricas de qualidade do retrieval
- Suporte a múltiplos documentos simultâneos
