"""
Build BM25 search index from the filtered knowledge base.
Outputs: viva_data/rag_index/chunks.json  (chunks + metadata)
         viva_data/rag_index/bm25.json    (serialised BM25 term stats)
Run once; chatbot.py loads these files at startup.
"""
import json
import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

KNOWLEDGE_BASE = Path("viva_data/processed/knowledge_base.jsonl")
OUT_DIR = Path("viva_data/rag_index")
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-záéíóúüñ]+", text)


def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    with open(KNOWLEDGE_BASE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("content", "").strip():  # include everything with content
                records.append(r)

    print(f"Useful records: {len(records)}")

    chunks_meta = []
    for rec in records:
        content = rec.get("content", "").strip()
        if not content:
            continue
        summary = rec.get("summary") or ""
        prefix = f"[{rec.get('category','').upper()} | {rec.get('group','')} | {rec.get('date','')[:10]}]\n"
        if summary:
            prefix += f"Resumen: {summary}\n\n"

        parts = chunk_text(prefix + content)
        for j, chunk in enumerate(parts):
            chunks_meta.append({
                "record_id": rec["id"],
                "category": rec.get("category", ""),
                "group": rec.get("group", ""),
                "date": rec.get("date", "")[:10],
                "year": rec.get("year", 0),
                "freshness_tier": rec.get("freshness_tier", ""),
                "is_time_sensitive": rec.get("is_time_sensitive", False),
                "author": rec.get("author", ""),
                "summary": summary[:300],
                "message_url": rec.get("message_url", ""),
                "chunk_index": j,
                "chunk_total": len(parts),
                "text": chunk,
            })

    print(f"Total chunks: {len(chunks_meta)}")

    # Build BM25 index
    corpus = [tokenize(c["text"]) for c in chunks_meta]
    bm25 = BM25Okapi(corpus)

    # Save chunks metadata
    with open(OUT_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)

    # Save BM25 model (pickle is fine for local use)
    with open(OUT_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print(f"\nDone. Index saved to {OUT_DIR}/")
    print(f"  chunks.json — {len(chunks_meta)} chunks with text + metadata")
    print(f"  bm25.pkl    — BM25 index ready for search")


if __name__ == "__main__":
    main()
