"""Compare admitted_uncertainty rates under original judge vs strict and liberal rubrics across all experiments."""
import json, math, sys, os
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

# Each entry: (label, tax_file_or_filter, soc_file_or_filter)
# For files containing both domains in one progress file, we filter by domain.
EXPERIMENTS = {
    "Germany Llama (headline)": {
        "tax": ("data/german/german_tax_200_seed42{suffix}.json", None),
        "soc": ("data/german/bsg_probe_200_progress{suffix}.json", None),
    },
    "Germany Qwen3 32B": {
        "tax": ("data/german/german_qwen_progress{suffix}.json", "tax"),
        "soc": ("data/german/german_qwen_progress{suffix}.json", "sozialrecht"),
    },
    "Germany gpt-oss 120B": {
        "tax": ("data/german/german_gptoss_progress{suffix}.json", "tax"),
        "soc": ("data/german/german_gptoss_progress{suffix}.json", "sozialrecht"),
    },
    "Germany RAG (BM25)": {
        "tax": ("data/german/german_rag_progress{suffix}.json", "tax"),
        "soc": ("data/german/german_rag_progress{suffix}.json", "sozialrecht"),
    },
    "France Llama": {
        "tax": ("data/french/french_200_progress{suffix}.json", "tax"),
        "soc": ("data/french/french_200_progress{suffix}.json", "social"),
    },
    "Canada Llama": {
        "tax": ("data/canadian/canadian_200_progress{suffix}.json", "tax"),
        "soc": ("data/canadian/canadian_200_progress{suffix}.json", "sst"),
    },
}

SUFFIXES = {"original": "", "strict": "_strict", "liberal": "_liberal"}


def hedging_rate(path, domain_filter):
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cases = d["results"] if isinstance(d, dict) else d
    n_yes, n_total = 0, 0
    for c in cases:
        if domain_filter is not None and c.get("domain") != domain_filter:
            continue
        for p in c.get("probes", []):
            if p.get("type") == "citation_recall":
                continue
            sc = p.get("scores") or {}
            au = sc.get("admitted_uncertainty")
            if au not in ("yes", "no"):
                continue
            n_total += 1
            if au == "yes":
                n_yes += 1
    return n_yes, n_total


def ztest(a, na, b, nb):
    if not (na and nb):
        return None
    p1, p2 = a / na, b / nb
    pp = (a + b) / (na + nb)
    se = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    z = (p1 - p2) / se if se else 0.0
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p1, p2, z, pv


def fmt_p(p):
    if p is None:
        return "n/a"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


for exp_name, files in EXPERIMENTS.items():
    print(f"\n=== {exp_name} ===")
    print(f"{'rubric':<10} {'tax DHR':<18} {'soc DHR':<18} {'gap (pp)':<10} {'z':<8} {'p':<10}")
    print("-" * 80)
    for rubric in ("original", "strict", "liberal"):
        tax_path = files["tax"][0].format(suffix=SUFFIXES[rubric])
        soc_path = files["soc"][0].format(suffix=SUFFIXES[rubric])
        ty, tn = hedging_rate(tax_path, files["tax"][1])
        sy, sn = hedging_rate(soc_path, files["soc"][1])
        if ty is None or sy is None or tn == 0 or sn == 0:
            print(f"{rubric:<10} (file missing)")
            continue
        res = ztest(ty, tn, sy, sn)
        if res is None:
            print(f"{rubric:<10} (insufficient data)")
            continue
        pT, pS, z, pv = res
        gap = (pT - pS) * 100
        print(
            f"{rubric:<10} {ty}/{tn}={pT:6.1%}    {sy}/{sn}={pS:6.1%}    "
            f"{gap:+7.1f}   {z:6.2f}  {fmt_p(pv)}"
        )
