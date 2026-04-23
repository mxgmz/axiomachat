"""
Processes all Viva Engage messages into structured JSONL for RAG.

One record per post. Each record combines:
  - Post text body
  - Descriptions + OCR from attached images
  - Transcripts from attached videos
  - Extracted text from attached documents

Output schema:
  id, source_type, content, content_parts, date, year, freshness_tier,
  is_time_sensitive, group, group_id, message_id, author,
  category, is_useful, summary, attachment_count, message_url
"""

import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
MESSAGES_FILE  = Path("viva_data/_all_messages.json")
ATTACHMENTS_DIR = Path("viva_data/attachments")
OUT_DIR        = Path("viva_data/processed")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE       = OUT_DIR / "knowledge_base.jsonl"
CHECKPOINT_FILE = OUT_DIR / ".processed_ids.json"

MODEL_TEXT   = "gpt-4o-mini"
MODEL_VISION = "gpt-4o-mini"
MAX_CONTENT_FOR_CLASSIFICATION = 6000  # chars sent to classifier
MAX_ATTACHMENT_CONTENT = 1500          # chars per attachment in combined content

CATEGORIES = [
    "benefits",       # health insurance, vacation, perks, wellness
    "hr_policy",      # rules, procedures, regulations, conduct
    "company_news",   # announcements, achievements, milestones
    "training",       # courses, workshops, certifications, skills
    "project_update", # construction/architectural project progress
    "social",         # birthday wishes, celebrations, personal photos
    "irrelevant",     # test posts, spam, non-informational
]

FRESHNESS_TIERS = {
    "current": (0, 1),
    "recent":  (1, 3),
    "dated":   (3, 5),
    "stale":   (5, 999),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".tif", ".tiff", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov", ".3gp", ".avi", ".mkv"}
DOC_EXTS   = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppsx"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    for fmt in ("%Y/%m/%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def freshness_tier(date: datetime | None) -> str:
    if not date:
        return "unknown"
    age_years = (datetime.now(timezone.utc) - date).days / 365
    for tier, (lo, hi) in FRESHNESS_TIERS.items():
        if lo <= age_years < hi:
            return tier
    return "stale"


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()))
    return set()


def save_checkpoint(ids: set):
    CHECKPOINT_FILE.write_text(json.dumps(list(ids)))


def append_record(record: dict):
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_attachment_file(att_id: int, name: str) -> Path | None:
    safe = "".join(c for c in name if c not in r'\/:*?"<>|\n\r')[:80].strip()
    candidate = ATTACHMENTS_DIR / f"{att_id}_{safe}"
    if candidate.exists():
        return candidate
    for f in ATTACHMENTS_DIR.glob(f"{att_id}_*"):
        return f
    return None


# ── Content extractors ───────────────────────────────────────────────────────

def extract_image_content(file_path: Path) -> str:
    try:
        data = base64.standard_b64encode(file_path.read_bytes()).decode()
        ext = file_path.suffix.lower().lstrip(".")
        media_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "jfif": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "webp": "image/webp", "heic": "image/jpeg",
            "tif": "image/jpeg", "tiff": "image/jpeg",
        }
        media_type = media_map.get(ext, "image/jpeg")

        resp = client.chat.completions.create(
            model=MODEL_VISION,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
                    {"type": "text", "text": (
                        "Describe this image in 1-2 sentences. "
                        "Then extract ALL visible text verbatim (OCR). "
                        "Format: DESCRIPTION: ... | OCR: ..."
                        "If no text is visible, write OCR: none."
                    )}
                ]
            }]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Image extraction error: {e}]"


def extract_pdf_text(file_path: Path) -> str:
    try:
        import fitz
        doc = fitz.open(str(file_path))
        return "\n".join(page.get_text() for page in doc)[:MAX_ATTACHMENT_CONTENT * 3]
    except Exception as e:
        return f"[PDF error: {e}]"


def extract_docx_text(file_path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX error: {e}]"


def extract_xlsx_text(file_path: Path) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(file_path), data_only=True)
        lines = []
        for sheet in wb.worksheets:
            lines.append(f"[Sheet: {sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join(str(c) for c in row if c is not None)
                if row_text.strip():
                    lines.append(row_text)
        return "\n".join(lines)
    except Exception as e:
        return f"[XLSX error: {e}]"


def transcribe_video(file_path: Path) -> str:
    # OpenAI Whisper API limit is 25MB
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > 25:
        return f"[Video too large for API ({size_mb:.1f}MB) — skipped]"
    try:
        with open(file_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return resp.strip() if isinstance(resp, str) else resp
    except Exception as e:
        return f"[Transcription error: {e}]"


def extract_attachment_content(att: dict) -> tuple[str, str]:
    """Returns (label, content) for an attachment."""
    att_id = att.get("id")
    name = att.get("name") or att.get("original_name") or f"file_{att_id}"
    ext = Path(name).suffix.lower()
    file_path = find_attachment_file(att_id, name)

    if not file_path:
        return ("file", f"[{name} — not downloaded]")

    if ext in IMAGE_EXTS:
        return ("image", extract_image_content(file_path))
    elif ext in VIDEO_EXTS:
        return ("video", transcribe_video(file_path))
    elif ext == ".pdf":
        return ("pdf", extract_pdf_text(file_path))
    elif ext in (".docx", ".doc"):
        return ("doc", extract_docx_text(file_path))
    elif ext in (".xlsx", ".xls"):
        return ("spreadsheet", extract_xlsx_text(file_path))
    else:
        return ("file", f"[{name} — unsupported type {ext}]")


# ── Classifier ───────────────────────────────────────────────────────────────

def classify_post(combined_content: str, group: str) -> dict:
    """Classify the full combined post content."""
    truncated = combined_content[:MAX_CONTENT_FOR_CLASSIFICATION]
    prompt = f"""You are classifying an employee social network post for an HR knowledge base RAG system at Axioma, a Mexican architecture/construction company.

Community: {group}

Post content (may include text + image descriptions + OCR + file extracts):
\"\"\"
{truncated}
\"\"\"

Respond ONLY with valid JSON:
{{
  "category": "one of {CATEGORIES}",
  "is_useful": true if this contains reusable knowledge for employees (policies, benefits, procedures, training info, company updates) — consider ALL content including images and attachments,
  "is_time_sensitive": true if the content mentions specific dates, deadlines, annual enrollment periods, or time-limited offers,
  "summary": "one-sentence summary of the useful information, or null if not useful"
}}"""

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL_TEXT,
                max_tokens=300,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if "429" in str(e):
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s, 75s
                print(f"    rate limit — waiting {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
                continue
            print(f"    classify error: {e}")
            return {"category": "irrelevant", "is_useful": False, "is_time_sensitive": False, "summary": None}


# ── Main ─────────────────────────────────────────────────────────────────────

def extract_thread_content(msgs: list[dict]) -> list[dict]:
    """Extract and combine content from all messages in a thread."""
    content_parts = []
    for idx, msg in enumerate(msgs):
        role = "Post" if idx == 0 else f"Reply {idx}"
        author = msg.get("sender", {}) or {}
        author_name = author.get("full_name") or author.get("name", "Unknown")
        text = msg.get("body", {}).get("plain", "") or ""
        attachments = [a for a in msg.get("attachments", []) if isinstance(a, dict)]

        if text.strip():
            content_parts.append({
                "type": "post_text",
                "role": role,
                "author": author_name,
                "date": msg.get("created_at", ""),
                "content": text,
            })

        for att in attachments:
            att_name = att.get("name") or f"file_{att.get('id')}"
            label, content = extract_attachment_content(att)
            if content and not content.startswith("["):
                content_parts.append({
                    "type": label,
                    "role": role,
                    "name": att_name,
                    "content": content[:MAX_ATTACHMENT_CONTENT],
                    "date": att.get("created_at", ""),
                })
            time.sleep(0.5)

    return content_parts


def build_combined(content_parts: list[dict]) -> str:
    combined = ""
    for part in content_parts:
        role = part.get("role", "")
        if part["type"] == "post_text":
            author = part.get("author", "")
            combined += f"[{role} — {author}]\n{part['content']}\n\n"
        else:
            combined += f"[{part['type'].upper()}: {part.get('name', '')} ({role})]\n{part['content']}\n\n"
    return combined.strip()


def main():
    print("Loading messages...")
    with open(MESSAGES_FILE, encoding="utf-8") as f:
        messages = json.load(f)
    print(f"Loaded {len(messages)} messages")

    # Build group_id → name lookup
    groups_file = Path("viva_data/groups.json")
    group_names = {}
    if groups_file.exists():
        for g in json.loads(groups_file.read_text(encoding="utf-8")):
            group_names[g["id"]] = g.get("name", "")
    print(f"Loaded {len(group_names)} group names")

    # Group messages into threads using thread_id
    from collections import defaultdict
    threads: dict[int, list] = defaultdict(list)
    for msg in messages:
        tid = msg.get("thread_id") or msg.get("id")
        threads[tid].append(msg)

    # Sort each thread by date (root first, replies after)
    for tid in threads:
        threads[tid].sort(key=lambda m: m.get("created_at", ""))

    thread_list = list(threads.items())
    print(f"Grouped into {len(thread_list)} threads\n")

    processed_ids = load_checkpoint()
    print(f"Already processed: {len(processed_ids)} records\n")

    total = useful = 0

    for i, (thread_id, thread_msgs) in enumerate(thread_list, 1):
        record_id = f"thread_{thread_id}"
        if record_id in processed_ids:
            continue

        root = thread_msgs[0]
        replies = thread_msgs[1:]
        group_name = group_names.get(root.get("group_id", 0), "")
        root_date = parse_date(root.get("created_at"))
        root_author = (root.get("sender", {}) or {})
        root_author_name = root_author.get("full_name") or root_author.get("name", "Unknown")
        all_attachments = sum(len([a for a in m.get("attachments", []) if isinstance(a, dict)]) for m in thread_msgs)

        print(f"[{i}/{len(thread_list)}] thread {thread_id} | {group_name} | {len(replies)} replies | {all_attachments} attachments")

        # ── Quick text pre-filter ─────────────────────────────────────────
        root_text = root.get("body", {}).get("plain", "") or ""
        all_text = " ".join(
            (m.get("body", {}).get("plain", "") or "") for m in thread_msgs
        ).strip()

        text_clf = None
        skip_images = False
        if all_text:
            text_clf = classify_post(all_text[:MAX_CONTENT_FOR_CLASSIFICATION], group_name)
            time.sleep(0.5)
            skip_images = not text_clf.get("is_useful", False) and \
                          text_clf.get("category", "") in ("irrelevant", "social")

        # ── Extract full thread content ───────────────────────────────────
        content_parts = []
        for idx, msg in enumerate(thread_msgs):
            role = "Post" if idx == 0 else f"Reply {idx}"
            author = (msg.get("sender", {}) or {})
            author_name = author.get("full_name") or author.get("name", "Unknown")
            text = msg.get("body", {}).get("plain", "") or ""
            attachments = [a for a in msg.get("attachments", []) if isinstance(a, dict)]

            if text.strip():
                content_parts.append({
                    "type": "post_text",
                    "role": role,
                    "author": author_name,
                    "date": msg.get("created_at", ""),
                    "content": text,
                })

            image_atts = [a for a in attachments if Path(a.get("name") or "").suffix.lower() in IMAGE_EXTS]
            other_atts = [a for a in attachments if Path(a.get("name") or "").suffix.lower() not in IMAGE_EXTS]

            for att in other_atts:
                att_name = att.get("name") or f"file_{att.get('id')}"
                print(f"    → {att_name[:50]}")
                label, content = extract_attachment_content(att)
                if content and not content.startswith("["):
                    content_parts.append({
                        "type": label, "role": role,
                        "name": att_name,
                        "content": content[:MAX_ATTACHMENT_CONTENT],
                        "date": att.get("created_at", ""),
                    })
                time.sleep(0.5)

            if not skip_images:
                for att in image_atts:
                    att_name = att.get("name") or f"file_{att.get('id')}"
                    print(f"    → [img] {att_name[:50]}")
                    label, content = extract_attachment_content(att)
                    if content and not content.startswith("["):
                        content_parts.append({
                            "type": label, "role": role,
                            "name": att_name,
                            "content": content[:MAX_ATTACHMENT_CONTENT],
                            "date": att.get("created_at", ""),
                        })
                    time.sleep(0.5)
            elif image_atts:
                print(f"    → skipping {len(image_atts)} image(s) (thread already irrelevant)")

        if not content_parts:
            processed_ids.add(record_id)
            save_checkpoint(processed_ids)
            continue

        # ── Final classification on full thread ───────────────────────────
        combined = build_combined(content_parts)
        if skip_images and not any(p["type"] not in ("post_text",) for p in content_parts):
            clf = text_clf or classify_post(combined, group_name)
        else:
            clf = classify_post(combined[:MAX_CONTENT_FOR_CLASSIFICATION], group_name)
            time.sleep(0.5)

        # ── Write record ──────────────────────────────────────────────────
        record = {
            "id": record_id,
            "source_type": "thread",
            "content": combined,
            "content_parts": content_parts,
            "date": root_date.isoformat() if root_date else None,
            "year": root_date.year if root_date else None,
            "freshness_tier": freshness_tier(root_date),
            "is_time_sensitive": clf.get("is_time_sensitive", False),
            "group": group_name,
            "group_id": root.get("group_id"),
            "thread_id": thread_id,
            "root_message_id": root.get("id"),
            "reply_count": len(replies),
            "author": root_author_name,
            "category": clf.get("category", "irrelevant"),
            "is_useful": clf.get("is_useful", False),
            "summary": clf.get("summary"),
            "attachment_count": all_attachments,
            "message_url": root.get("web_url", ""),
        }

        append_record(record)
        processed_ids.add(record_id)
        save_checkpoint(processed_ids)
        total += 1
        if record["is_useful"]:
            useful += 1

    print(f"\nDone. Total: {total}  Useful: {useful}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)
    main()
