"""
generator.py
------------
Responsável por montar o prompt com os chunks recuperados
e enviar para o Llama gerar a resposta final via Ollama.

O que acontece aqui é o coração do RAG:
- Sem RAG: LLM responde só com o que memorizou no treinamento
- Com RAG: LLM recebe o contexto correto e responde com base nele

Estrutura do prompt:
    [Sistema]  → define o comportamento do modelo
    [Contexto] → chunks recuperados pelo FAISS
    [Pergunta] → query do usuário
"""

import ollama


MODELO = "llama3.1"

SYSTEM_PROMPT = """Você é um assistente especializado em responder perguntas
com base em documentos fornecidos. Responda sempre em português.

Regras importantes:
- Use APENAS as informações do contexto fornecido para responder
- Se a resposta não estiver no contexto, diga claramente que não encontrou a informação
- Seja objetivo e direto
- Nunca invente informações"""


def montar_contexto(chunks_recuperados: list[dict]) -> str:
    """
    Formata os chunks recuperados como bloco de contexto para o prompt.

    Cada chunk recebe seu score de relevância para fins de transparência.
    """
    partes = []
    for i, item in enumerate(chunks_recuperados, 1):
        partes.append(
            f"[Trecho {i} — relevância: {item['score']:.4f}]\n{item['chunk']}"
        )
    return "\n\n".join(partes)


def gerar_resposta(query: str, chunks_recuperados: list[dict]) -> str:
    """
    Monta o prompt completo e envia para o Llama via Ollama.

    Parâmetros:
        query              : pergunta do usuário
        chunks_recuperados : lista de dicts com 'chunk' e 'score'

    Retorna:
        Resposta gerada pelo modelo como string
    """
    contexto = montar_contexto(chunks_recuperados)

    prompt = f"""Use o contexto abaixo para responder à pergunta do usuário.

CONTEXTO:
{contexto}

PERGUNTA:
{query}

RESPOSTA:"""

    resposta = ollama.chat(
        model=MODELO,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={
        "num_predict": 1024,  # aumenta o limite de tokens gerados
        "temperature": 0.3,   # mantém resposta focada
    }
    )

    return resposta["message"]["content"]
