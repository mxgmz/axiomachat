from http.server import BaseHTTPRequestHandler
import json
import os
import pickle
import re
from pathlib import Path

# ── Load index once at module level (cached across warm invocations) ──────────
_DIR = Path(__file__).parent.parent / "viva_data" / "rag_index"

with open(_DIR / "chunks.json", encoding="utf-8") as f:
    _CHUNKS = json.load(f)

with open(_DIR / "bm25.pkl", "rb") as f:
    _BM25 = pickle.load(f)

SYSTEM_PROMPT = """Eres el asistente de Recursos Humanos de Axioma, empresa de arquitectura y construcción en México.
Responde preguntas de empleados sobre prestaciones, políticas, seguros, capacitación y noticias internas.
Responde siempre en español, de forma clara y amigable.
Basa tus respuestas ÚNICAMENTE en el contexto proporcionado.
Si la información no está disponible, dilo honestamente — no inventes datos."""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ]+", text.lower())


def _retrieve(query: str, top_k: int = 5) -> list[dict]:
    scores = _BM25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [_CHUNKS[i] for i in top if scores[i] > 0]


def _build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[Fuente {i} — {h['category']} | {h['group']} | {h['date']}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def _llm_answer(query: str, context: str) -> str:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if anthropic_key.startswith("sk-ant-"):
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"}],
        )
        return resp.content[0].text

    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"},
            ],
        )
        return resp.choices[0].message.content

    # No LLM — return formatted excerpts
    lines = ["Aquí está la información más relevante que encontré:\n"]
    for h in json.loads(context.split("\n\n---\n\n")[0].split("\n", 1)[1])[:3] if False else []:
        pass
    return "⚠️ No hay LLM configurado. Configura ANTHROPIC_API_KEY o OPENAI_API_KEY en Vercel."


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        query = (body.get("query") or "").strip()

        if not query:
            self._respond(400, {"error": "query requerido"})
            return

        hits = _retrieve(query)
        if not hits:
            self._respond(200, {
                "answer": "No encontré información relevante sobre ese tema en nuestra base de conocimiento.",
                "sources": [],
            })
            return

        context = _build_context(hits)
        try:
            answer = _llm_answer(query, context)
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

        self._respond(200, {"answer": answer, "sources": sources})

    def _respond(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence access logs
