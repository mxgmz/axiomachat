import requests
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("YAMMER_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://www.yammer.com/api/v1"
OUT = Path("viva_data")
OUT.mkdir(exist_ok=True)
(OUT / "attachments").mkdir(exist_ok=True)


def get(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    time.sleep(0.4)
    return r.json()


def download_file(url, dest_path):
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
        return False


# ── 1. Groups ────────────────────────────────────────────────────────────────
print("Fetching groups...")
all_groups = []
page = 1
while True:
    data = get(f"{BASE}/groups.json", {"page": page})
    if not data:
        break
    all_groups.extend(data)
    print(f"  {len(all_groups)} groups so far...")
    if len(data) < 20:
        break
    page += 1

with open(OUT / "groups.json", "w", encoding="utf-8") as f:
    json.dump(all_groups, f, indent=2, ensure_ascii=False)
print(f"Saved {len(all_groups)} groups\n")


# ── 2. Messages per group ────────────────────────────────────────────────────
print("Fetching messages per group...")
all_messages = []

for group in all_groups:
    gid = group["id"]
    gname = group["name"].replace("/", "_").replace(" ", "_")
    messages = []
    older_than = None

    filename = f"group_{gid}_{gname}_messages.json"
    if (OUT / filename).exists():
        print(f"  Skipping (already saved): {group['name']}")
        with open(OUT / filename, encoding="utf-8") as f:
            all_messages.extend(json.load(f))
        continue

    print(f"  Group: {group['name']}")
    while True:
        params = {"older_than": older_than} if older_than else {}
        data = get(f"{BASE}/messages/in_group/{gid}.json", params)
        batch = data.get("messages", [])
        if not batch:
            break
        messages.extend(batch)
        all_messages.extend(batch)
        older_than = batch[-1]["id"]
        print(f"    {len(messages)} messages...")
        if len(batch) < 20:
            break

    filename = f"group_{gid}_{gname}_messages.json"
    with open(OUT / filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


with open(OUT / "_all_messages.json", "w", encoding="utf-8") as f:
    json.dump(all_messages, f, indent=2, ensure_ascii=False)
print(f"Total messages saved: {len(all_messages)}\n")


# ── 3. Files metadata ────────────────────────────────────────────────────────
print("Fetching file metadata...")
all_files = []
page = 1
while True:
    data = get(f"{BASE}/files.json", {"page": page})
    if not data:
        break
    all_files.extend(data)
    if len(data) < 20:
        break
    page += 1

with open(OUT / "files.json", "w", encoding="utf-8") as f:
    json.dump(all_files, f, indent=2, ensure_ascii=False)
print(f"Found {len(all_files)} files\n")


# ── 4. Download attachments ──────────────────────────────────────────────────
print("Downloading file attachments...")
downloaded = 0
skipped = 0

for file in all_files:
    if not isinstance(file, dict):
        continue
    fid = file.get("id")
    name = file.get("name", f"file_{fid}")
    safe_name = "".join(c for c in name if c not in r'\/:*?"<>|')
    dest = OUT / "attachments" / f"{fid}_{safe_name}"

    if dest.exists():
        skipped += 1
        continue

    url = file.get("download_url") or file.get("content_url")
    if not url:
        print(f"  No URL for: {name}")
        continue

    print(f"  Downloading: {name}")
    if download_file(url, dest):
        downloaded += 1
    time.sleep(0.3)

print(f"Downloaded {downloaded} files, skipped {skipped} already present\n")


# ── 5. Also grab attachments embedded in messages ───────────────────────────
print("Downloading attachments from messages...")
msg_downloaded = 0

for msg in all_messages:
    for att in msg.get("attachments", []):
        att_id = att.get("id")
        att_name = att.get("name", f"att_{att_id}") or f"att_{att_id}"
        safe_name = "".join(c for c in att_name if c not in r'\/:*?"<>|\n\r')[:80]
        dest = OUT / "attachments" / f"{att_id}_{safe_name}"

        if dest.exists():
            continue

        url = att.get("download_url") or att.get("content_url")
        if not url:
            continue

        if download_file(url, dest):
            msg_downloaded += 1
        time.sleep(0.3)

print(f"Downloaded {msg_downloaded} message attachments\n")
print(f"All done. Data saved to ./{OUT}/")
