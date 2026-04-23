"""
Axioma HR Chatbot — BM25 retrieval + Claude/OpenAI answer generation.
Usage: python chatbot.py
"""
import json
import os
import pickle
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = Path("viva_data/rag_index")
TOP_K = 5  # chunks to retrieve per query

SYSTEM_PROMPT = """Eres el asistente de Recursos Humanos de Axioma, una empresa de arquitectura y construcción en México.
Tu trabajo es responder preguntas de los empleados sobre prestaciones, políticas, seguros, capacitación y noticias de la empresa.

Responde siempre en español, de forma clara y amigable.
Basa tus respuestas ÚNICAMENTE en la información del contexto proporcionado.
Si la información no está en el contexto, dilo honestamente — no inventes datos.
Cuando sea relevante, menciona la fecha o grupo de origen de la información."""


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ]+", text.lower())


def load_index():
    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    return chunks, bm25


def retrieve(query: str, chunks: list, bm25, top_k: int = TOP_K) -> list[dict]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices if scores[i] > 0]


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[Fuente {i} — {h['category']} | {h['group']} | {h['date']}]\n{h['text']}"
        )
    return "\n\n---\n\n".join(parts)


def get_llm_client():
    """Return (client, model, provider). Tries Anthropic first, then OpenAI."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    # Anthropic keys start with sk-ant-
    if anthropic_key.startswith("sk-ant-"):
        try:
            import anthropic
            return anthropic.Anthropic(api_key=anthropic_key), "claude-haiku-4-5-20251001", "anthropic"
        except ImportError:
            pass

    # OpenAI key
    if openai_key:
        try:
            from openai import OpenAI
            return OpenAI(api_key=openai_key), "gpt-4o-mini", "openai"
        except ImportError:
            pass

    return None, None, None


def answer(query: str, context: str, client, model: str, provider: str) -> str:
    messages = [
        {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"}
    ]
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text
    else:
        from openai import OpenAI
        resp = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        return resp.choices[0].message.content


def main():
    print("Cargando índice...")
    chunks, bm25 = load_index()
    print(f"  {len(chunks)} fragmentos listos.\n")

    client, model, provider = get_llm_client()
    if client:
        print(f"  LLM: {model} ({provider})\n")
    else:
        print("  Sin API LLM — mostrando solo fragmentos relevantes.\n")

    print("=" * 60)
    print("  Asistente HR de Axioma")
    print("  Escribe tu pregunta o 'salir' para terminar.")
    print("=" * 60)

    while True:
        try:
            query = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not query:
            continue
        if query.lower() in {"salir", "exit", "quit"}:
            print("¡Hasta luego!")
            break

        hits = retrieve(query, chunks, bm25)
        if not hits:
            print("\nAsistente: No encontré información relevante sobre ese tema.")
            continue

        if client:
            context = build_context(hits)
            try:
                response = answer(query, context, client, model, provider)
                print(f"\nAsistente: {response}")
            except Exception as e:
                print(f"\n[Error LLM: {e}]")
                print("\nFragmentos más relevantes:")
                for h in hits:
                    print(f"\n  [{h['category']} | {h['date']}] {h['text'][:300]}...")
        else:
            print("\nFragmentos más relevantes:")
            for h in hits:
                print(f"\n  [{h['category']} | {h['group']} | {h['date']}]")
                print(f"  {h['text'][:400]}...")


if __name__ == "__main__":
    main()
