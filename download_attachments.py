import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
import requests

load_dotenv()

COOKIES = {
    "FedAuth": os.getenv("SP_FEDAUTH"),
    "rtFa":    os.getenv("SP_RTFA"),
    "SIMI":    os.getenv("SP_SIMI"),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Accept": "*/*",
}

OUT = Path("viva_data/attachments")
OUT.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.cookies.update(COOKIES)
session.headers.update(HEADERS)


def safe_name(name: str, max_len: int = 80) -> str:
    cleaned = "".join(c for c in name if c not in r'\/:*?"<>|\n\r').strip()
    if "." in cleaned:
        stem, ext = cleaned.rsplit(".", 1)
        return stem[:max_len - len(ext) - 1] + "." + ext
    return cleaned[:max_len]


def site_from_url(sp_url: str) -> str:
    """Extract SharePoint site base URL from a sharepoint_web_url."""
    parsed = urlparse(sp_url)
    parts = parsed.path.split("/")
    if "sites" in parts:
        idx = parts.index("sites")
        site_path = "/".join(parts[:idx + 2])
    else:
        site_path = ""
    return f"{parsed.scheme}://{parsed.netloc}{site_path}"


def server_relative_path(sp_url: str) -> str:
    return unquote(urlparse(sp_url).path)


def download(att: dict, dest: Path) -> bool:
    sp_url = att.get("sharepoint_web_url")
    if not sp_url:
        return False

    site = site_from_url(sp_url)
    rel_path = server_relative_path(sp_url)

    # SharePoint REST API direct file download
    api_url = f"{site}/_api/web/GetFileByServerRelativePath(decodedurl='{rel_path}')/$value"

    try:
        r = session.get(api_url, stream=True, timeout=30, allow_redirects=True)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        else:
            print(f"    HTTP {r.status_code}: {att.get('name', '')}")
            return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


# ── Load all attachments, deduplicate by ID ──────────────────────────────────
print("Loading messages...")
with open("viva_data/_all_messages.json", encoding="utf-8") as f:
    messages = json.load(f)

seen_ids = set()
attachments = []
for msg in messages:
    for att in msg.get("attachments", []):
        if not isinstance(att, dict):
            continue
        att_id = att.get("id")
        if att_id and att_id not in seen_ids:
            seen_ids.add(att_id)
            attachments.append(att)

print(f"Unique attachments: {len(attachments)}")

# ── Download ─────────────────────────────────────────────────────────────────
downloaded = skipped = failed = 0

for i, att in enumerate(attachments, 1):
    att_id = att.get("id")
    name = att.get("name") or att.get("original_name") or f"file_{att_id}"
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    dest = OUT / f"{att_id}_{safe_name(name)}"

    if dest.exists():
        skipped += 1
        continue

    if not att.get("sharepoint_web_url"):
        failed += 1
        continue

    print(f"  [{i}/{len(attachments)}] {name[:60]}")
    if download(att, dest):
        downloaded += 1
    else:
        failed += 1

    time.sleep(0.3)

print(f"\nDone. Downloaded: {downloaded}  Skipped: {skipped}  Failed: {failed}")
print(f"Files saved to: {OUT}/")
