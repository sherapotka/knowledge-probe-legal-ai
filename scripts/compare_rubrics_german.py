"""Compare admitted_uncertainty rates under original judge vs strict and liberal rubrics, German Llama only."""
import json, math, sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

TAX_FILES = {
    "original": "data/german/german_tax_200_seed42.json",
    "strict":   "data/german/german_tax_200_seed42_strict.json",
    "liberal":  "data/german/german_tax_200_seed42_liberal.json",
}
SOC_FILES = {
    "original": "data/german/bsg_probe_200_progress.json",
    "strict":   "data/german/bsg_probe_200_progress_strict.json",
    "liberal":  "data/german/bsg_probe_200_progress_liberal.json",
}


def hedging_rate(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cases = d["results"] if isinstance(d, dict) else d
    n_yes, n_total = 0, 0
    for c in cases:
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
    p1, p2 = a / na, b / nb
    pp = (a + b) / (na + nb)
    se = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    z = (p1 - p2) / se if se else 0.0
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p1, p2, z, pv


print(f"{'rubric':<10} {'tax DHR':<14} {'soc DHR':<14} {'gap (pp)':<10} {'z':<8} {'p':<10}")
print("-" * 70)
for rubric in ("original", "strict", "liberal"):
    tax_yes, tax_n = hedging_rate(TAX_FILES[rubric])
    soc_yes, soc_n = hedging_rate(SOC_FILES[rubric])
    pT, pS, z, pv = ztest(tax_yes, tax_n, soc_yes, soc_n)
    gap = (pT - pS) * 100
    print(
        f"{rubric:<10} {tax_yes}/{tax_n}={pT:6.1%}  {soc_yes}/{soc_n}={pS:6.1%}  "
        f"{gap:+7.1f}  {z:6.2f}  {pv:.6f}"
    )

# Agreement matrix between original and the two rubrics, per probe
print("\nAgreement vs original (% of probes with same label):")
for rubric in ("strict", "liberal"):
    for label_dom, dom_files in [("tax", TAX_FILES), ("social", SOC_FILES)]:
        with open(dom_files["original"], encoding="utf-8") as f:
            orig = json.load(f)
        with open(dom_files[rubric], encoding="utf-8") as f:
            new = json.load(f)
        oc = orig["results"] if isinstance(orig, dict) else orig
        nc = new["results"] if isinstance(new, dict) else new
        same, total, conf = 0, 0, Counter()
        for co, cn in zip(oc, nc):
            for po, pn in zip(co.get("probes", []), cn.get("probes", [])):
                if po.get("type") == "citation_recall":
                    continue
                lo = (po.get("scores") or {}).get("admitted_uncertainty")
                ln = (pn.get("scores") or {}).get("admitted_uncertainty")
                if lo not in ("yes", "no") or ln not in ("yes", "no"):
                    continue
                total += 1
                if lo == ln:
                    same += 1
                conf[(lo, ln)] += 1
        print(
            f"  {rubric} vs original  ({label_dom}):  agreement={same}/{total}={same/total:.1%}    "
            f"orig=YES&new=NO: {conf[('yes','no')]}    orig=NO&new=YES: {conf[('no','yes')]}"
        )
