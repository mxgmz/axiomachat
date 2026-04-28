# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**axiomachat** — an HR chatbot for Axioma (Mexican architecture firm) deployed on Vercel. It lets employees ask questions about benefits, HR policies, training, and the employee directory in Spanish. It also has an AI image generator for branded social media cards.

The backend is a Flask Python app (`index.py`) served via Vercel's Python runtime. There is no Node.js layer — no `package.json`.

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask dev server (loads .env automatically)
python index.py
# → http://localhost:5000
```

The server requires `viva_data/rag_index/` (FAISS index + chunks.json) to exist at startup — it reads these into memory when the module loads. Missing index = crash on import.

## Environment variables

All secrets live in `.env` (gitignored). Required keys:

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | Embeddings (text-embedding-3-small), chat (gpt-4o-mini), image gen (gpt-image-1/2) |
| `RUNPOD_API_KEY` | Nano Banana 2 image model via RunPod async job API |
| `YAMMER_TOKEN` | Live Viva Engage API calls (recent posts tool) |
| `ANTHROPIC_API_KEY` | `chatbot.py` CLI only (falls back to OpenAI) |

## Data pipeline (local only — never deployed)

Run these scripts in order to rebuild the RAG index from scratch:

```bash
# 1. Download all messages + files from Viva Engage
python extract_viva.py          # → viva_data/group_*.json, viva_data/_all_messages.json

# 2. Extract employee directory
python extract_people.py        # → viva_data/people.json

# 3. Download file attachments referenced in messages (optional, needed for full extraction)
python download_attachments.py  # → viva_data/attachments/

# 4. Process messages → structured JSONL (expensive: calls GPT-4o-mini for every thread)
python process_content.py       # → viva_data/processed/knowledge_base.jsonl
                                 # Checkpoints to .processed_ids.json — safe to interrupt

# 5. Build FAISS vector index
python build_rag.py             # → viva_data/rag_index/index.faiss + chunks.json
```

`requirements-pipeline.txt` installs the extra deps (pymupdf, python-docx, openpyxl, whisper, rank_bm25) needed only for the pipeline.

## Architecture

### Request flow (`index.py`)

```
Browser → GET /           → serves index.html (chat UI)
Browser → GET /imagenes   → serves imagenes.html (image generator UI)
Browser → POST /api/chat  → _llm_answer() → OpenAI gpt-4o-mini with tool use loop
Browser → POST /api/generate-image → _generate_image_api() → OpenAI or RunPod
```

**Chat tool-use loop** (`_llm_answer`): up to 5 rounds. Three tools:
- `search_knowledge_base` — FAISS cosine similarity over pre-built index
- `search_people` — in-memory keyword scoring over `viva_data/people.json`
- `get_recent_viva_posts` — live Yammer REST API call

**Image generation** supports three model backends:
- `gpt-image-1` (default) — OpenAI Images API
- `gpt-image-2` — OpenAI Responses API (multimodal, sends brand reference images first)
- `nano-banana-2` — RunPod async job (submits, polls every 2s up to 120s)

Brand identity constants (`_BRAND_SPEC`, `_TYPE_CONTEXTS`) in `index.py` control the image prompt. Brand reference images are loaded from `Axioma images/`.

### Key data files (committed or generated)

- `viva_data/rag_index/index.faiss` — FAISS IndexFlatIP (inner product, L2-normalized = cosine sim)
- `viva_data/rag_index/chunks.json` — metadata + text for each vector
- `viva_data/people.json` — employee directory (name, email, title, dept, phone)
- `viva_data/processed/knowledge_base.jsonl` — intermediate: one record per Viva thread

### MCP server (`viva_engage_mcp.py`)

A standalone MCP server (not used by the Flask app) that exposes Viva Engage tools to Claude Desktop. Configured via `claude_desktop_config.json`. Run with `python viva_engage_mcp.py`.

## Deployment

Deployed to Vercel (project: `axiomachat`). The Flask app is served as a Python serverless function. Static HTML files (`index.html`, `imagenes.html`) are served inline via Flask routes, not as Vercel static assets.

Vercel project ID: `prj_700Ax6kU4sb6e22qBuK4XLqtPYMG`
