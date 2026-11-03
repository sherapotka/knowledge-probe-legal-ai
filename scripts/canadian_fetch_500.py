"""
Fetch 500 Canadian Court Decisions per domain from A2AJ API
============================================================
- 500 Tax Court of Canada (TCC) decisions
- 500 Refugee Protection Division (RPD) decisions
"""

import json
import os
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "canadian_500_cases.json")

API_BASE = "https://api.a2aj.ca"


def fetch_cases(dataset_code, domain_label, max_cases=500):
    """Fetch cases from A2AJ API using search endpoint."""
    cases = []
    page = 0
    page_size = 50  # API max is 50

    while len(cases) < max_cases:
        print(f"    Fetching page {page+1} (have {len(cases)} cases)...")
        try:
            resp = requests.get(f"{API_BASE}/search", params={
                "query": "*",
                "doc_type": "cases",
                "dataset": dataset_code,
                "size": page_size,
                "search_language": "en",
                "sort_results": "newest_first",
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

            citation = case.get("citation_en", "")
            if not citation:
                continue

            # Text is already in the search results
            text = case.get("unofficial_text_en", "") or ""

            # If not in search results, fetch separately
            if len(text) < 500:
                try:
                    text_resp = requests.get(f"{API_BASE}/fetch", params={
                        "citation": citation,
                        "doc_type": "cases",
                        "output_language": "en",
                    }, timeout=30)
                    text_resp.raise_for_status()
                    fetch_data = text_resp.json()
                    fetch_results = fetch_data.get("results", [])
                    if fetch_results:
                        text = fetch_results[0].get("unofficial_text_en", "") or ""
                except Exception as e:
                    print(f"    Error fetching text for {citation}: {e}")
                    continue

            if len(text) < 500:
                continue

            cases.append({
                "citation": citation,
                "name": case.get("name_en", ""),
                "date": case.get("document_date_en", ""),
                "dataset": dataset_code,
                "text": text,
                "chars": len(text),
            })

            if len(cases) % 50 == 0:
                print(f"    ... {len(cases)} cases collected")

        page += 1
        time.sleep(0.3)

    return cases


def main():
    all_cases = {}

    # Tax Court cases
    print(f"{'='*60}")
    print("Fetching 500 Tax Court of Canada (TCC) cases")
    print(f"{'='*60}")
    tax_cases = fetch_cases("TCC", "tax", 500)
    for i, case in enumerate(tax_cases):
        key = f"tax/{i:04d}_{case['citation'].replace(' ', '_')}"
        all_cases[key] = {
            "domain": "tax",
            "title": f"{case['name']} ({case['citation']})",
            "citation": case["citation"],
            "date": case["date"],
            "court": "Tax Court of Canada",
            "chars": case["chars"],
            "text": case["text"],
        }
    print(f"  Got {len(tax_cases)} tax cases\n")

    # Refugee cases
    print(f"{'='*60}")
    print("Fetching 500 Refugee Protection Division (RPD) cases")
    print(f"{'='*60}")
    refugee_cases = fetch_cases("RPD", "refugee", 500)
    for i, case in enumerate(refugee_cases):
        key = f"refugee/{i:04d}_{case['citation'].replace(' ', '_')}"
        all_cases[key] = {
            "domain": "refugee",
            "title": f"{case['name']} ({case['citation']})",
            "citation": case["citation"],
            "date": case["date"],
            "court": "Refugee Protection Division",
            "chars": case["chars"],
            "text": case["text"],
        }
    print(f"  Got {len(refugee_cases)} refugee cases\n")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_cases)} total cases to {OUTPUT_FILE}")
    print(f"  Tax: {len(tax_cases)}, Refugee: {len(refugee_cases)}")


if __name__ == "__main__":
    main()
