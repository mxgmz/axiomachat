import json
import os
import pickle
import re
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Load BM25 index once at startup ──────────────────────────────────────────
_DIR = Path(__file__).parent.parent / "viva_data" / "rag_index"

with open(_DIR / "chunks.json", encoding="utf-8") as f:
    _CHUNKS = json.load(f)

with open(_DIR / "bm25.pkl", "rb") as f:
    _BM25 = pickle.load(f)

SYSTEM_PROMPT = (
    "Eres el asistente de Recursos Humanos de Axioma, empresa de arquitectura y "
    "construcción en México. Responde preguntas de empleados sobre prestaciones, "
    "políticas, seguros, capacitación y noticias internas. "
    "Responde siempre en español, de forma clara y amigable. "
    "Basa tus respuestas ÚNICAMENTE en el contexto proporcionado. "
    "Si la información no está disponible, dilo honestamente — no inventes datos."
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../public", static_url_path="")
CORS(app)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ]+", text.lower())


def _retrieve(query: str, top_k: int = 5) -> list[dict]:
    scores = _BM25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_CHUNKS[i] for i in top if scores[i] > 0]


def _llm_answer(query: str, hits: list[dict]) -> str:
    context_parts = []
    for i, h in enumerate(hits, 1):
        context_parts.append(
            f"[Fuente {i} — {h['category']} | {h['group']} | {h['date']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    messages = [{"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"}]

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if anthropic_key.startswith("sk-ant-"):
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text

    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        return resp.choices[0].message.content

    return "⚠️ Configura ANTHROPIC_API_KEY o OPENAI_API_KEY en las variables de entorno."


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "query requerido"}), 400

    hits = _retrieve(query)
    if not hits:
        return jsonify({
            "answer": "No encontré información relevante sobre ese tema en nuestra base de conocimiento.",
            "sources": [],
        })

    try:
        answer = _llm_answer(query, hits)
    except Exception as e:
        answer = f"Error al generar respuesta: {e}"

    sources = [
        {
            "category": h["category"],
            "group": h["group"],
            "date": h["date"],
            "summary": h.get("summary", ""),
            "url": h.get("message_url", ""),
            "excerpt": h["text"][:300] + ("…" if len(h["text"]) > 300 else ""),
        }
        for h in hits
    ]

    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True)
