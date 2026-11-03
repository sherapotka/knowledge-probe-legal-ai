"""Draw the 200-case German tax subsample reported in the paper from the 500-case tax run.

The paper's German comparison uses 200 tax cases against 200 BSG social-welfare cases. The
tax side was run on 500 BFH cases (german_500_progress.json); the 200 reported are a seeded
random sample of them, kept with their original probes, answers and judge scores. This
script reproduces that draw (verified: it returns the same 200 cases in the same order as
data/german/german_tax_200_seed42.json).

Usage (from the repository root): python scripts/subsample_german_tax_200.py [--write]
"""
import json
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
SRC = "data/german/german_500_progress.json"
DST = "data/german/german_tax_200_seed42.json"

with open(SRC, encoding="utf-8") as fh:
    full = json.load(fh)
tax = [r for r in full["results"] if r["domain"] == "tax"]
random.seed(42)
sample = random.sample(tax, 200)
out = {"processed_keys": [r["case_key"] for r in sample], "results": sample}

if "--write" in sys.argv:
    with open(DST, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", DST)
else:
    with open(DST, encoding="utf-8") as fh:
        existing = json.load(fh)
    same = [r["case_key"] for r in existing["results"]] == out["processed_keys"]
    print(f"{len(sample)} cases drawn; matches {DST}: {same}")
