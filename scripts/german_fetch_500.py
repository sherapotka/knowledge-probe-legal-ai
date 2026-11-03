"""
Fetch 500 German Court Decisions per domain from Open Legal Data API
====================================================================
- 500 Steuerrecht cases from BFH
- 500 Asylrecht cases from BVerwG (Senat 1, asylum-related)
"""

import json
import os
import re
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "german_500_cases.json")

API_BASE = "https://de.openlegaldata.io/api/cases/"

ASYLUM_KEYWORDS = [
    "AsylG", "Asylgesetz", "Flüchtlingseigenschaft", "subsidiär",
    "BAMF", "Abschiebungsverbot", "Flüchtling",
    "Aufenthaltsgesetz", "AufenthG", "Asylverfahren",
    "Schutzstatus", "Abschiebung", "Duldung",
]


def strip_html(html):
    text = re.sub(r'<br\s*/?>', '\n', html)
    text = re.sub(r'<[^>]+>', ' ', text)
    for entity, char in [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                          ('&quot;', '"'), ('&auml;', 'ä'), ('&ouml;', 'ö'),
                          ('&uuml;', 'ü'), ('&szlig;', 'ß'), ('&Auml;', 'Ä'),
                          ('&Ouml;', 'Ö'), ('&Uuml;', 'Ü')]:
        text = text.replace(entity, char)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch_paginated(court_slug, max_cases, filter_fn=None):
    """Fetch cases with pagination."""
    cases = []
    page = 1
    page_size = 100

    while len(cases) < max_cases:
        print(f"    Page {page} (have {len(cases)} cases)...")
        try:
            resp = requests.get(API_BASE, params={
                "court__slug": court_slug,
                "page_size": page_size,
                "page": page,
            }, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    Error on page {page}: {e}")
            time.sleep(2)
            page += 1
            continue

        results = data.get("results", [])
        if not results:
            print(f"    No more results at page {page}")
            break

        for case in results:
            if len(cases) >= max_cases:
                break

            content = case.get("content", "")

            # Apply domain filter if provided
            if filter_fn and not filter_fn(case, content):
                continue

            text = strip_html(content)
            if len(text) < 500:
                continue

            cases.append({
                "id": case["id"],
                "slug": case.get("slug", ""),
                "file_number": case.get("file_number", ""),
                "date": case.get("date", ""),
                "court_name": case.get("court", {}).get("name", ""),
                "ecli": case.get("ecli", ""),
                "text": text,
                "chars": len(text),
            })

        total = data.get("count", 0)
        if page * page_size >= total:
            print(f"    Reached end of results ({total} total)")
            break

        page += 1
        time.sleep(0.5)  # be polite

    return cases


def is_asylum_case(case, content):
    """Filter for asylum-related BVerwG cases."""
    fn = case.get("file_number", "")
    if not fn.startswith("1 "):
        return False
    return any(kw in content for kw in ASYLUM_KEYWORDS)


def main():
    target = 500
    all_cases = {}

    # Tax cases from BFH
    print(f"{'='*60}")
    print(f"Fetching {target} Steuerrecht cases (BFH)")
    print(f"{'='*60}")
    tax_cases = fetch_paginated("bfh", target)
    for case in tax_cases:
        key = f"tax/{case['slug']}"
        all_cases[key] = {
            "domain": "tax",
            "case_id": str(case["id"]),
            "title": f"BFH {case['file_number']} ({case['date']})",
            "court": "Bundesfinanzhof",
            "date": case["date"],
            "reference": case["file_number"],
            "ecli": case["ecli"],
            "chars": case["chars"],
            "text": case["text"],
        }
    print(f"  Got {len(tax_cases)} tax cases\n")

    # Asylum cases from BVerwG
    print(f"{'='*60}")
    print(f"Fetching {target} Asylrecht cases (BVerwG Senat 1)")
    print(f"{'='*60}")
    asylum_cases = fetch_paginated("bverwg", target, filter_fn=is_asylum_case)
    for case in asylum_cases:
        key = f"asylum/{case['slug']}"
        all_cases[key] = {
            "domain": "asylum",
            "case_id": str(case["id"]),
            "title": f"BVerwG {case['file_number']} ({case['date']})",
            "court": "Bundesverwaltungsgericht",
            "date": case["date"],
            "reference": case["file_number"],
            "ecli": case["ecli"],
            "chars": case["chars"],
            "text": case["text"],
        }
    print(f"  Got {len(asylum_cases)} asylum cases\n")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_cases)} total cases to {OUTPUT_FILE}")
    print(f"  Tax: {len(tax_cases)}, Asylum: {len(asylum_cases)}")


if __name__ == "__main__":
    main()
