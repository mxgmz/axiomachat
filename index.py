import base64
import io
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
_BASE        = Path(__file__).parent
_DIR         = _BASE / "viva_data" / "rag_index"
_HTML        = (_BASE / "index.html").read_text(encoding="utf-8")
_HTML_IMG    = (_BASE / "imagenes.html").read_text(encoding="utf-8")
_people_path = _BASE / "viva_data" / "people.json"
_PEOPLE      = json.loads(_people_path.read_text(encoding="utf-8")) if _people_path.exists() else []
_BANNER_PATH = _BASE / "Axioma images" / "1200x630.png.webp"
_LOGO_PATH   = _BASE / "Axioma images" / "images-3.png"

# ── Load FAISS index + chunks ─────────────────────────────────────────────────
with open(_DIR / "chunks.json", encoding="utf-8") as f:
    _CHUNKS = json.load(f)

_INDEX = faiss.read_index(str(_DIR / "index.faiss"))

SYSTEM_PROMPT = (
    "Eres el asistente de Recursos Humanos de Axioma, empresa de arquitectura y "
    "construcción en México. Responde preguntas de empleados sobre prestaciones, "
    "políticas, seguros, capacitación, noticias internas y directorio de empleados. "
    "Responde siempre en español, de forma clara y amigable.\n\n"
    "REGLAS DE USO DE HERRAMIENTAS — SIEMPRE debes llamar al menos una herramienta antes de responder:\n"
    "• Usa 'search_knowledge_base' para preguntas sobre prestaciones, seguros, "
    "políticas, capacitación, eventos, comunicados o cualquier tema interno de la empresa.\n"
    "• Usa 'search_people' cuando pregunten por una persona específica, cómo contactar "
    "a alguien, quién trabaja en un área o información de directorio.\n"
    "• Usa 'get_recent_viva_posts' cuando pregunten por noticias recientes, "
    "últimas publicaciones o qué ha pasado esta semana/mes en la empresa.\n"
    "• Puedes llamar varias herramientas en paralelo si la pregunta lo requiere.\n\n"
    "Basa tus respuestas ÚNICAMENTE en la información recuperada por las herramientas. "
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


def _tool_get_recent_viva_posts(days: int = 7, group_name: str = "") -> tuple[str, list[dict]]:
    token = os.environ.get("YAMMER_TOKEN", "")
    if not token:
        return "Token de Viva Engage no configurado.", []

    days = min(max(int(days), 1), 30)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        r = requests.get(
            "https://www.yammer.com/api/v1/messages.json",
            headers=headers, timeout=15
        )
        if r.status_code != 200:
            return f"Error al consultar Viva Engage: HTTP {r.status_code}", []

        all_msgs = r.json().get("messages", [])

        # Group by thread_id: first message in each thread = post, rest = replies
        threads: dict[int, dict] = {}
        for m in all_msgs:
            body = (m.get("body") or {}).get("plain", "").strip()
            if not body:
                continue
            grp = m.get("group_name") or ""
            if group_name and group_name.lower() not in grp.lower():
                continue
            tid = m.get("thread_id") or m.get("id")
            if tid not in threads:
                threads[tid] = {
                    "date":    (m.get("created_at") or "")[:10],
                    "group":   grp,
                    "sender":  m.get("sender_name", ""),
                    "body":    body,
                    "url":     m.get("web_url", "") or m.get("url", ""),
                    "replies": [],
                }
            else:
                threads[tid]["replies"].append({
                    "sender": m.get("sender_name", ""),
                    "body":   body[:200],
                })

        if not threads:
            return f"No se encontraron publicaciones recientes en los últimos {days} días.", []

        text_parts, sources = [], []
        for tid, t in list(threads.items())[:10]:
            block = (
                f"[{t['date']} | {t['group']} | {t['sender']}]\n"
                f"{t['body'][:500]}"
            )
            if t["replies"]:
                replies_txt = "\n".join(
                    f"  ↳ {r['sender']}: {r['body']}"
                    for r in t["replies"][:5]
                )
                block += f"\n\nComentarios:\n{replies_txt}"
            text_parts.append(block)

            # Build reply excerpt for the sources panel
            reply_excerpt = ""
            if t["replies"]:
                reply_excerpt = " | Comentarios: " + " / ".join(
                    f"{r['sender']}: {r['body'][:80]}"
                    for r in t["replies"][:3]
                )

            sources.append({
                "category": "viva engage",
                "group":    t["group"],
                "date":     t["date"],
                "summary":  t["sender"],
                "url":      t["url"],
                "excerpt":  t["body"][:250] + ("…" if len(t["body"]) > 250 else "") + reply_excerpt,
            })

        text = (
            f"Publicaciones recientes de Viva Engage (últimos {days} días):\n\n"
            + "\n\n---\n\n".join(text_parts)
        )
        return text, sources

    except Exception as e:
        return f"Error al obtener publicaciones: {e}", []


def _run_tool(name: str, args: dict) -> tuple[str, list[dict]]:
    if name == "search_knowledge_base":
        return _tool_search_knowledge_base(args.get("query", "")), []
    if name == "search_people":
        return _tool_search_people(args.get("query", "")), []
    if name == "get_recent_viva_posts":
        return _tool_get_recent_viva_posts(
            days=args.get("days", 7),
            group_name=args.get("group_name", "")
        )
    return f"Herramienta desconocida: {name}", []


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
            name        = tc.function.name
            args        = json.loads(tc.function.arguments or "{}")
            result, src = _run_tool(name, args)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

            # Sources from live Viva posts come directly from the tool
            used_sources.extend(src)

            # Sources from knowledge base: re-run FAISS to get metadata
            if name == "search_knowledge_base":
                vec_resp = client.embeddings.create(
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


# ── Image generation ─────────────────────────────────────────────────────────

_TYPE_PROMPTS = {
    "cumpleanos":   "birthday celebration card, festive balloons and confetti, warm and joyful mood",
    "aniversario":  "work anniversary milestone card, professional achievement, years of dedication celebrated",
    "bienvenida":   "welcome card for a new team member, warm and inviting, fresh start energy",
    "logro":        "achievement and recognition card, celebrating success and excellence, trophy or star motif",
    "personalizado":"custom corporate communication card",
}

def _find_person_for_image(prompt: str) -> dict | None:
    prompt_lower = prompt.lower()
    best_score, best_person = 0, None
    for p in _PEOPLE:
        if not p.get("active"):
            continue
        name_words = (p.get("name") or "").lower().split()
        score = sum(1 for w in name_words if len(w) > 3 and w in prompt_lower)
        if score > best_score:
            best_score, best_person = score, p
    return best_person if best_score >= 1 else None


def _build_image_prompt(card_type: str, user_prompt: str, person: dict | None) -> str:
    type_desc = _TYPE_PROMPTS.get(card_type, _TYPE_PROMPTS["personalizado"])
    person_line = ""
    if person:
        parts = []
        if person.get("name"):      parts.append(f'name "{person["name"]}"')
        if person.get("job_title"): parts.append(person["job_title"])
        if person.get("department"):parts.append(f'from {person["department"]} department')
        person_line = f" Feature the person: {', '.join(parts)}."

    return (
        f"Create a professional {type_desc} for Axioma, a Mexican architecture and real estate company. "
        f"Use Axioma's brand visual identity from the reference image: bright yellow background (#F5E000), "
        f"bold black sans-serif typography, clean architectural and geometric line art elements in the corners. "
        f"Minimalist corporate design. "
        f"Card message or context: {user_prompt}.{person_line} "
        f"Include 'axioma.' logotype in bold black. "
        f"Square format, print-ready, high quality."
    )


def _generate_image_api(full_prompt: str, user_image_b64: str | None) -> str:
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=openai_key)

    ref_images = []

    if _BANNER_PATH.exists():
        ref_images.append(("banner.webp", _BANNER_PATH.read_bytes(), "image/webp"))
    if _LOGO_PATH.exists():
        ref_images.append(("logo.png", _LOGO_PATH.read_bytes(), "image/png"))
    if user_image_b64:
        ref_images.append(("reference.png", base64.b64decode(user_image_b64), "image/png"))

    if ref_images:
        image_files = [
            (name, io.BytesIO(data), mime)
            for name, data, mime in ref_images
        ]
        resp = client.images.edit(
            model="gpt-image-1",
            image=image_files,
            prompt=full_prompt,
            n=1,
            size="1024x1024",
        )
    else:
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=full_prompt,
            n=1,
            size="1024x1024",
            quality="high",
        )

    return resp.data[0].b64_json


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({"chunks": len(_CHUNKS), "people": len(_PEOPLE), "status": "ok"})


@app.route("/")
def index():
    return Response(_HTML, mimetype="text/html")


@app.route("/imagenes")
def imagenes():
    return Response(_HTML_IMG, mimetype="text/html")


@app.route("/api/generate-image", methods=["POST"])
def generate_image():
    data      = request.get_json(force=True, silent=True) or {}
    card_type = data.get("type", "personalizado")
    prompt    = (data.get("prompt") or "").strip()
    user_img  = data.get("image")  # base64 or null

    if not prompt:
        return jsonify({"error": "prompt requerido"}), 400

    person      = _find_person_for_image(prompt)
    full_prompt = _build_image_prompt(card_type, prompt, person)

    try:
        b64 = _generate_image_api(full_prompt, user_img)
        return jsonify({
            "image":  b64,
            "person": person.get("name") if person else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
