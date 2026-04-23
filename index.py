import json
import os
import numpy as np
import faiss
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
_DIR  = _BASE / "viva_data" / "rag_index"
_HTML = (_BASE / "index.html").read_text(encoding="utf-8")

# ── Load FAISS index + chunks ─────────────────────────────────────────────────
with open(_DIR / "chunks.json", encoding="utf-8") as f:
    _CHUNKS = json.load(f)

_INDEX = faiss.read_index(str(_DIR / "index.faiss"))

SYSTEM_PROMPT = (
    "Eres el asistente de Recursos Humanos de Axioma, empresa de arquitectura y "
    "construcción en México. Responde preguntas de empleados sobre prestaciones, "
    "políticas, seguros, capacitación y noticias internas. "
    "Responde siempre en español, de forma clara y amigable. "
    "Basa tus respuestas ÚNICAMENTE en el contexto proporcionado. "
    "Si la información no está disponible, dilo honestamente — no inventes datos."
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


def _retrieve(query: str, top_k: int = 10) -> list[dict]:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=openai_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    _, indices = _INDEX.search(vec, top_k)
    return [_CHUNKS[i] for i in indices[0] if i >= 0]


def _llm_answer(query: str, hits: list[dict], history: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Fuente {i} — {h['category']} | {h['group']} | {h['date']}]\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    # History (previous turns) + current question with retrieved context
    messages = list(history) + [
        {"role": "user", "content": f"Contexto relevante:\n{context}\n\nPregunta: {query}"}
    ]

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key    = os.environ.get("OPENAI_API_KEY", "")

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
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        return resp.choices[0].message.content

    return "⚠️ Configura ANTHROPIC_API_KEY o OPENAI_API_KEY en las variables de entorno de Vercel."


@app.route("/health")
def health():
    return jsonify({"chunks": len(_CHUNKS), "status": "ok"})


@app.route("/")
def index():
    return Response(_HTML, mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data  = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "query requerido"}), 400

    hits = _retrieve(query)
    if not hits:
        return jsonify({
            "answer": "No encontré información relevante sobre ese tema en nuestra base de conocimiento.",
            "sources": [],
        })

    history = [m for m in (data.get("history") or []) if m.get("role") in ("user", "assistant")][-12:]

    try:
        answer = _llm_answer(query, hits, history)
    except Exception as e:
        answer = f"Error al generar respuesta: {e}"

    sources = [
        {
            "category": h["category"],
            "group":    h["group"],
            "date":     h["date"],
            "summary":  h.get("summary", ""),
            "url":      h.get("message_url", ""),
            "excerpt":  h["text"][:300] + ("…" if len(h["text"]) > 300 else ""),
        }
        for h in hits
    ]
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True)
