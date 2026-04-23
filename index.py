import json
import os
import time
import numpy as np
import faiss
import requests
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE   = Path(__file__).parent
_DIR    = _BASE / "viva_data" / "rag_index"
_HTML   = (_BASE / "index.html").read_text(encoding="utf-8")
_people_path = _BASE / "viva_data" / "people.json"
_PEOPLE = json.loads(_people_path.read_text(encoding="utf-8")) if _people_path.exists() else []

# ── Load FAISS index + chunks ─────────────────────────────────────────────────
with open(_DIR / "chunks.json", encoding="utf-8") as f:
    _CHUNKS = json.load(f)

_INDEX = faiss.read_index(str(_DIR / "index.faiss"))

SYSTEM_PROMPT = (
    "Eres el asistente de Recursos Humanos de Axioma, empresa de arquitectura y "
    "construcción en México. Responde preguntas de empleados sobre prestaciones, "
    "políticas, seguros, capacitación, noticias internas y directorio de empleados. "
    "Responde siempre en español, de forma clara y amigable. "
    "Usa las herramientas disponibles para buscar información antes de responder. "
    "Basa tus respuestas ÚNICAMENTE en la información recuperada. "
    "Si la información no está disponible, dilo honestamente — no inventes datos."
)

# ── Tool definitions ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Busca en la base de conocimiento de Axioma (publicaciones de Viva Engage, "
                "políticas, prestaciones, seguros, capacitación, noticias). Úsala para "
                "preguntas sobre beneficios, eventos, comunicados internos o cualquier "
                "tema general de la empresa."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta de búsqueda en lenguaje natural"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_people",
            "description": (
                "Busca empleados de Axioma por nombre, puesto o departamento. "
                "Úsala cuando pregunten quién es alguien, cómo contactar a una persona, "
                "qué personas trabajan en un área, o información de directorio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nombre, puesto o departamento a buscar"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_viva_posts",
            "description": (
                "Obtiene publicaciones recientes de Viva Engage (Yammer) en tiempo real. "
                "Úsala cuando pregunten por noticias recientes, últimas publicaciones, "
                "qué está pasando esta semana, o información muy actual de la empresa."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Número de días hacia atrás a consultar (default 7, máx 30)",
                        "default": 7
                    },
                    "group_name": {
                        "type": "string",
                        "description": "Nombre del grupo/comunidad de Viva Engage (opcional, omitir para todos)"
                    }
                },
                "required": []
            }
        }
    }
]

# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_search_knowledge_base(query: str, top_k: int = 10) -> str:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=openai_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    _, indices = _INDEX.search(vec, top_k)
    hits = [_CHUNKS[i] for i in indices[0] if i >= 0]
    if not hits:
        return "No se encontró información relevante en la base de conocimiento."
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[Fuente {i} — {h['category']} | {h['group']} | {h['date']}]\n{h['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _tool_search_people(query: str) -> str:
    q = query.lower()
    results = []
    for p in _PEOPLE:
        if not p.get("active"):
            continue
        score = 0
        name  = (p.get("name") or "").lower()
        title = (p.get("job_title") or "").lower()
        dept  = (p.get("department") or "").lower()
        exp   = (p.get("expertise") or "").lower()
        for word in q.split():
            if word in name:  score += 3
            if word in title: score += 2
            if word in dept:  score += 2
            if word in exp:   score += 1
        if score > 0:
            results.append((score, p))

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:8]

    if not top:
        return f"No se encontraron empleados que coincidan con '{query}'."

    lines = []
    for _, p in top:
        info = [f"**{p['name']}**"]
        if p.get("job_title"):   info.append(f"Puesto: {p['job_title']}")
        if p.get("department"):  info.append(f"Departamento: {p['department']}")
        if p.get("email"):       info.append(f"Email: {p['email']}")
        if p.get("mobile"):      info.append(f"Móvil: {p['mobile']}")
        if p.get("work_phone"):  info.append(f"Tel. trabajo: {p['work_phone']}")
        if p.get("location"):    info.append(f"Ubicación: {p['location']}")
        lines.append("\n".join(info))
    return "\n\n".join(lines)


def _tool_get_recent_viva_posts(days: int = 7, group_name: str = "") -> str:
    token = os.environ.get("YAMMER_TOKEN", "")
    if not token:
        return "Token de Viva Engage no configurado."

    days  = min(max(int(days), 1), 30)
    since = int(time.time()) - days * 86400

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        url = "https://www.yammer.com/api/v1/messages.json"
        r   = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return f"Error al consultar Viva Engage: HTTP {r.status_code}"

        data     = r.json()
        messages = data.get("messages", [])

        posts = []
        for m in messages:
            created = m.get("created_at", "")
            body    = (m.get("body") or {}).get("plain", "").strip()
            sender  = m.get("sender_name", "")
            group   = m.get("group_name", "") or ""

            if not body:
                continue
            if group_name and group_name.lower() not in group.lower():
                continue

            date_str = created[:10] if created else ""
            posts.append(f"[{date_str} | {group} | {sender}]\n{body[:400]}")

        if not posts:
            return f"No se encontraron publicaciones recientes en los últimos {days} días."

        return f"Publicaciones recientes de Viva Engage (últimos {days} días):\n\n" + "\n\n---\n\n".join(posts[:10])

    except Exception as e:
        return f"Error al obtener publicaciones: {e}"


def _run_tool(name: str, args: dict) -> str:
    if name == "search_knowledge_base":
        return _tool_search_knowledge_base(args.get("query", ""))
    if name == "search_people":
        return _tool_search_people(args.get("query", ""))
    if name == "get_recent_viva_posts":
        return _tool_get_recent_viva_posts(
            days=args.get("days", 7),
            group_name=args.get("group_name", "")
        )
    return f"Herramienta desconocida: {name}"


# ── LLM with tool-use loop ────────────────────────────────────────────────────

def _llm_answer(query: str, history: list[dict]) -> tuple[str, list[dict]]:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    client     = OpenAI(api_key=openai_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += list(history)
    messages.append({"role": "user", "content": query})

    used_sources: list[dict] = []

    for _ in range(5):  # max 5 tool-call rounds
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )
        msg = resp.choices[0].message

        # No more tool calls — return final answer
        if not msg.tool_calls:
            return msg.content or "", used_sources

        # Append assistant message with tool_calls
        messages.append(msg)

        # Execute each tool call and collect results
        for tc in msg.tool_calls:
            name   = tc.function.name
            args   = json.loads(tc.function.arguments or "{}")
            result = _run_tool(name, args)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

            # Track sources for knowledge base searches
            if name == "search_knowledge_base":
                openai_key2 = os.environ.get("OPENAI_API_KEY", "")
                client2     = OpenAI(api_key=openai_key2)
                vec_resp    = client2.embeddings.create(
                    model="text-embedding-3-small",
                    input=[args.get("query", query)]
                )
                vec = np.array([vec_resp.data[0].embedding], dtype="float32")
                faiss.normalize_L2(vec)
                _, idxs = _INDEX.search(vec, 5)
                for i in idxs[0]:
                    if i >= 0:
                        h = _CHUNKS[i]
                        used_sources.append({
                            "category": h["category"],
                            "group":    h["group"],
                            "date":     h["date"],
                            "summary":  h.get("summary", ""),
                            "url":      h.get("message_url", ""),
                            "excerpt":  h["text"][:300] + ("…" if len(h["text"]) > 300 else ""),
                        })

    # Fallback if loop exhausted
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=messages + [{"role": "user", "content": "Por favor proporciona tu respuesta final."}],
    )
    return final.choices[0].message.content or "", used_sources


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({"chunks": len(_CHUNKS), "people": len(_PEOPLE), "status": "ok"})


@app.route("/")
def index():
    return Response(_HTML, mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data  = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "query requerido"}), 400

    history = [
        m for m in (data.get("history") or [])
        if m.get("role") in ("user", "assistant")
    ][-12:]

    try:
        answer, sources = _llm_answer(query, history)
    except Exception as e:
        answer  = f"Error al generar respuesta: {e}"
        sources = []

    # Deduplicate sources by record_id
    seen, unique_sources = set(), []
    for s in sources:
        key = s.get("url") or s.get("excerpt", "")[:60]
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return jsonify({"answer": answer, "sources": unique_sources[:5]})


if __name__ == "__main__":
    app.run(debug=True)
