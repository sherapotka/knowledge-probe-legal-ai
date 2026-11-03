"""Re-fetch the court decision texts that are not redistributed with this repository.

The case files under data/ carry every identifier of each decision but not its text. This
script downloads the texts from the original sources and writes them back into the `text`
field, in place, so that the pipeline scripts find the files in the shape they expect.

Sources
  Germany   Open Legal Data, https://de.openlegaldata.io/api/cases/<id>/  (field `case_id`)
  France    Conseil d'Etat open data, monthly zip archives of XML decisions at
            https://opendata.justice-administrative.fr/DCE/<year>/<month>/CE_<year><month>.zip
            (fields `source_file` and `date`)
  Canada    A2AJ, https://api.a2aj.ca/fetch?citation=<citation>  (field `citation`)

Usage (from the repository root):
  python scripts/refetch_decisions.py                       # every case file
  python scripts/refetch_decisions.py data/german/bsg_cases_200.json data/french/french_200_cases.json

An Open Legal Data API token can be passed through the OLD_API_TOKEN environment variable if
the anonymous rate limit is too low. Monthly French archives are cached in .cache/ next to
this file. Re-running skips records that already have a text.
"""
import io
import json
import os
import re
import sys
import time
import zipfile

import requests
from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(CACHE, exist_ok=True)

CASE_FILES = [
    "data/german/german_500_cases.json",
    "data/german/bsg_cases.json",
    "data/german/bsg_cases_200.json",
    "data/german/german_rag_corpus.json",
    "data/french/french_200_cases.json",
    "data/french/french_50_cases.json",
    "data/french/french_pilot_cases.json",
    "data/canadian/canadian_500_cases.json",
    "data/canadian/sst_cases.json",
]

OLD_API = "https://de.openlegaldata.io/api/cases/"
A2AJ_API = "https://api.a2aj.ca"
CE_ZIP = "https://opendata.justice-administrative.fr/DCE/{y}/{m:02d}/CE_{y}{m:02d}.zip"


# --- text cleaning, identical to the fetch scripts that built the original files ---------

def strip_html_german(html):
    text = re.sub(r"<br\s*/?>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", text)
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                         ("&auml;", "ä"), ("&ouml;", "ö"), ("&uuml;", "ü"), ("&szlig;", "ß"),
                         ("&Auml;", "Ä"), ("&Ouml;", "Ö"), ("&Uuml;", "Ü")]:
        text = text.replace(entity, char)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_html_french(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --- per-source fetchers -----------------------------------------------------------------

def fetch_german(case_id):
    headers = {}
    token = os.environ.get("OLD_API_TOKEN")
    if token:
        headers["Authorization"] = f"Token {token}"
    r = requests.get(f"{OLD_API}{case_id}/", headers=headers, timeout=60)
    r.raise_for_status()
    return strip_html_german(r.json().get("content", ""))


def fetch_canadian(citation):
    r = requests.get(f"{A2AJ_API}/fetch", params={
        "citation": citation, "doc_type": "cases", "output_language": "en"}, timeout=60)
    r.raise_for_status()
    results = r.json().get("results", [])
    return (results[0].get("unofficial_text_en") or "") if results else ""


_zip_cache = {}


def french_archive(year, month):
    key = (year, month)
    if key not in _zip_cache:
        local = os.path.join(CACHE, f"CE_{year}{month:02d}.zip")
        if not os.path.exists(local):
            url = CE_ZIP.format(y=year, m=month)
            print(f"  downloading {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(local, "wb") as fh:
                fh.write(r.content)
        _zip_cache[key] = zipfile.ZipFile(local)
    return _zip_cache[key]


def fetch_french(source_file, date):
    year, month = int(date[:4]), int(date[5:7])
    z = french_archive(year, month)
    names = [n for n in z.namelist() if n.endswith(source_file)]
    if not names:
        return ""
    root = etree.fromstring(z.read(names[0]), etree.XMLParser(recover=True))
    ti = root.find(".//Texte_Integral")
    return clean_html_french(etree.tostring(ti, encoding="unicode")) if ti is not None else ""


# --- driver ------------------------------------------------------------------------------

def records(data):
    if isinstance(data, list):
        return data
    return list(data.values())


def refetch(path):
    full = os.path.join(ROOT, path)
    with open(full, encoding="utf-8") as fh:
        data = json.load(fh)
    recs = records(data)
    todo = [r for r in recs if not r.get("text")]
    print(f"{path}: {len(recs)} records, {len(todo)} to fetch")
    done = failed = 0
    for i, r in enumerate(todo, 1):
        try:
            if "source_file" in r:
                text = fetch_french(r["source_file"], r["date"])
            elif "citation" in r:
                text = fetch_canadian(r["citation"])
            else:
                text = fetch_german(r["case_id"])
        except Exception as e:
            print(f"  [{i}/{len(todo)}] failed: {e}")
            failed += 1
            time.sleep(2)
            continue
        if not text:
            print(f"  [{i}/{len(todo)}] empty text for {r.get('citation') or r.get('source_file') or r.get('case_id')}")
            failed += 1
            continue
        r["text"] = text
        done += 1
        if r.get("chars") and abs(len(text) - r["chars"]) > 0.05 * r["chars"]:
            print(f"  [{i}/{len(todo)}] length differs from the original "
                  f"({len(text)} vs {r['chars']} characters); the source may have changed")
        if i % 25 == 0:
            with open(full, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
        time.sleep(0.5)
    with open(full, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    print(f"  {done} fetched, {failed} failed")


if __name__ == "__main__":
    for p in (sys.argv[1:] or CASE_FILES):
        refetch(p)
