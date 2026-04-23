"""
Extract all Axioma employee profiles from Yammer API.
Uses paginated /api/v1/users.json — covers everyone, not just those who posted.
Output: viva_data/people.json
"""
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

TOKEN = os.getenv("YAMMER_TOKEN")
BASE  = "https://www.yammer.com/api/v1"
OUT   = Path("viva_data/people.json")

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
})


def fetch_page(page: int) -> list[dict]:
    r = session.get(f"{BASE}/users.json", params={"page": page}, timeout=20)
    if r.status_code == 200:
        return r.json()
    print(f"  HTTP {r.status_code} on page {page}: {r.text[:200]}")
    return []


def parse_user(u: dict) -> dict:
    contact = u.get("contact", {})
    phones  = contact.get("phone_numbers", [])
    emails  = contact.get("email_addresses", [])

    mobile = next((p["number"] for p in phones if p.get("type") == "mobile"), "")
    work   = next((p["number"] for p in phones if p.get("type") == "work"), "")
    email  = next((e["address"] for e in emails if e.get("type") == "primary"),
                  u.get("email", ""))

    return {
        "id":           u.get("id"),
        "name":         u.get("full_name", ""),
        "email":        email,
        "job_title":    u.get("job_title", ""),
        "department":   u.get("department", ""),
        "location":     u.get("location", ""),
        "mobile":       mobile,
        "work_phone":   work,
        "expertise":    u.get("expertise", ""),
        "summary":      u.get("summary", ""),
        "profile_url":  u.get("web_url", ""),
        "active":       u.get("state", "") == "active",
    }


people = []
page   = 1

print("Fetching users...")
while True:
    print(f"  Page {page}...")
    batch = fetch_page(page)
    if not batch:
        break
    people.extend(parse_user(u) for u in batch)
    print(f"    {len(batch)} users — total so far: {len(people)}")
    if len(batch) < 50:   # last page
        break
    page += 1
    time.sleep(0.5)

# Deduplicate by id
seen = set()
unique = []
for p in people:
    if p["id"] not in seen:
        seen.add(p["id"])
        unique.append(p)

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(unique, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(unique)} employees saved to {OUT}")
active = sum(1 for p in unique if p["active"])
print(f"  Active: {active}  |  Inactive/guests: {len(unique) - active}")
