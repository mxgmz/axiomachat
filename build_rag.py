"""
Build semantic search index using OpenAI text-embedding-3-small + FAISS.
Outputs: viva_data/rag_index/chunks.json  — chunk text + metadata
         viva_data/rag_index/index.faiss  — FAISS vector index
"""
import json
import os
import time
import re
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
KNOWLEDGE_BASE = Path("viva_data/processed/knowledge_base.jsonl")
OUT_DIR = Path("viva_data/rag_index")
CHUNK_SIZE    = 1800
CHUNK_OVERLAP = 200
BATCH_SIZE    = 50


def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts: list[str], retries: int = 6) -> list[list[float]]:
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [r.embedding for r in resp.data]
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 20 * (attempt + 1)
                print(f"  Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after retries")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    with open(KNOWLEDGE_BASE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("content", "").strip():
                records.append(r)

    print(f"Records: {len(records)}")

    chunks_text, chunks_meta = [], []
    for rec in records:
        content = rec.get("content", "").strip()
        summary = rec.get("summary") or ""
        prefix  = f"[{rec.get('category','').upper()} | {rec.get('group','')} | {rec.get('date','')[:10]}]\n"
        if summary:
            prefix += f"Resumen: {summary}\n\n"

        for j, chunk in enumerate(chunk_text(prefix + content)):
            chunks_text.append(chunk)
            chunks_meta.append({
                "record_id":     rec["id"],
                "category":      rec.get("category", ""),
                "group":         rec.get("group", ""),
                "date":          rec.get("date", "")[:10],
                "year":          rec.get("year", 0),
                "freshness_tier":rec.get("freshness_tier", ""),
                "is_time_sensitive": rec.get("is_time_sensitive", False),
                "author":        rec.get("author", ""),
                "summary":       summary[:300],
                "message_url":   rec.get("message_url", ""),
                "chunk_index":   j,
                "text":          chunk,
            })

    print(f"Total chunks: {len(chunks_text)}")

    all_embeddings = []
    for i in range(0, len(chunks_text), BATCH_SIZE):
        batch = chunks_text[i:i + BATCH_SIZE]
        print(f"  Embedding {min(i+BATCH_SIZE, len(chunks_text))}/{len(chunks_text)}...")
        all_embeddings.extend(embed(batch))
        time.sleep(0.5)

    matrix = np.array(all_embeddings, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(matrix)

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    with open(OUT_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False)

    print(f"\nDone — {len(chunks_text)} vectors saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
