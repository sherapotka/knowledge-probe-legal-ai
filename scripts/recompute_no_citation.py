"""Recompute all paper numbers excluding citation-recall probes."""
import json, sys, math, random
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

def compute_stats(results):
    scores = defaultdict(list)
    unc = [0, 0]
    for r in results:
        for p in r.get('probes', []):
            if p.get('type') == 'citation_recall':
                continue
            s = p.get('scores', {})
            if not isinstance(s, dict):
                continue
            for m in ['correct_test', 'correct_citations', 'correct_statute', 'specificity']:
                v = s.get(m)
                if isinstance(v, dict):
                    v = v.get('score')
                if v:
                    scores[m].append(v)
            au = s.get('admitted_uncertainty')
            unc[1] += 1
            if au in (True, 'yes'):
                unc[0] += 1
    return unc, scores

def ztest(unc1, n1, unc2, n2):
    p1, p2 = unc1/n1, unc2/n2
    pp = (unc1+unc2)/(n1+n2)
    z = (p1-p2) / math.sqrt(pp*(1-pp)*(1/n1+1/n2))
    pv = 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return p1, p2, z, pv

def avg(lst):
    return sum(lst)/len(lst) if lst else 0

# GERMANY
random.seed(42)
with open('data/german/german_500_progress.json', encoding='utf-8', errors='replace') as f:
    de = json.load(f)
tax_de = random.sample([r for r in de['results'] if r.get('domain')=='tax'], 200)
with open('data/german/bsg_probe_200_progress.json', encoding='utf-8', errors='replace') as f:
    bsg = json.load(f)
soc_de = bsg['results'][:200]

t_unc, t_sc = compute_stats(tax_de)
s_unc, s_sc = compute_stats(soc_de)
p1,p2,z,pv = ztest(t_unc[0],t_unc[1],s_unc[0],s_unc[1])
print(f"GERMANY: Tax {t_unc[0]}/{t_unc[1]}={p1:.1%} vs Soc {s_unc[0]}/{s_unc[1]}={p2:.1%}, z={z:.2f}, p={pv:.6f}")
print(f"  Tax: test={avg(t_sc['correct_test']):.2f} cit={avg(t_sc['correct_citations']):.2f} stat={avg(t_sc['correct_statute']):.2f} spec={avg(t_sc['specificity']):.2f}")
print(f"  Soc: test={avg(s_sc['correct_test']):.2f} cit={avg(s_sc['correct_citations']):.2f} stat={avg(s_sc['correct_statute']):.2f} spec={avg(s_sc['specificity']):.2f}")
print(f"  Gap: {(p1-p2)*100:.1f} pp")

# FRANCE
with open('data/french/french_200_progress.json', encoding='utf-8') as f:
    fr = json.load(f)
tax_fr = [r for r in fr['results'] if r['domain']=='tax']
soc_fr = [r for r in fr['results'] if r['domain']=='social']
t_unc, t_sc = compute_stats(tax_fr)
s_unc, s_sc = compute_stats(soc_fr)
p1,p2,z,pv = ztest(t_unc[0],t_unc[1],s_unc[0],s_unc[1])
print(f"\nFRANCE: Tax {t_unc[0]}/{t_unc[1]}={p1:.1%} vs Soc {s_unc[0]}/{s_unc[1]}={p2:.1%}, z={z:.2f}, p={pv:.4f}")
print(f"  Tax: test={avg(t_sc['correct_test']):.2f} cit={avg(t_sc['correct_citations']):.2f} stat={avg(t_sc['correct_statute']):.2f} spec={avg(t_sc['specificity']):.2f}")
print(f"  Soc: test={avg(s_sc['correct_test']):.2f} cit={avg(s_sc['correct_citations']):.2f} stat={avg(s_sc['correct_statute']):.2f} spec={avg(s_sc['specificity']):.2f}")

# CANADA
with open('data/canadian/canadian_200_progress.json', encoding='utf-8') as f:
    ca = json.load(f)
tax_ca = [r for r in ca['results'] if r['domain']=='tax']
soc_ca = [r for r in ca['results'] if r['domain']=='sst']
t_unc, t_sc = compute_stats(tax_ca)
s_unc, s_sc = compute_stats(soc_ca)
p1,p2,z,pv = ztest(t_unc[0],t_unc[1],s_unc[0],s_unc[1])
print(f"\nCANADA: Tax {t_unc[0]}/{t_unc[1]}={p1:.1%} vs Soc {s_unc[0]}/{s_unc[1]}={p2:.1%}, z={z:.2f}, p={pv:.4f}")
print(f"  Tax: test={avg(t_sc['correct_test']):.2f} cit={avg(t_sc['correct_citations']):.2f} stat={avg(t_sc['correct_statute']):.2f} spec={avg(t_sc['specificity']):.2f}")
print(f"  Soc: test={avg(s_sc['correct_test']):.2f} cit={avg(s_sc['correct_citations']):.2f} stat={avg(s_sc['correct_statute']):.2f} spec={avg(s_sc['specificity']):.2f}")

# RAG
with open('data/german/german_rag_progress.json', encoding='utf-8') as f:
    rag = json.load(f)
rag_tax = [r for r in rag['results'] if r['domain']=='tax']
rag_soc = [r for r in rag['results'] if r['domain']=='sozialrecht']

# No-RAG baseline uses same tax_de / soc_de
norag_t, _ = compute_stats(tax_de)
norag_s, _ = compute_stats(soc_de)
rag_t, _ = compute_stats(rag_tax)
rag_s, _ = compute_stats(rag_soc)

p1,p2,z1,pv1 = ztest(norag_t[0],norag_t[1],norag_s[0],norag_s[1])
p3,p4,z2,pv2 = ztest(rag_t[0],rag_t[1],rag_s[0],rag_s[1])
gap_before = (p1-p2)*100
gap_after = (p3-p4)*100
closure = (1 - gap_after/gap_before)*100 if gap_before != 0 else 0
print(f"\nRAG:")
print(f"  No RAG: Tax {norag_t[0]}/{norag_t[1]}={p1:.1%} vs Soc {norag_s[0]}/{norag_s[1]}={p2:.1%}, z={z1:.2f}, p={pv1:.6f}")
print(f"  With RAG: Tax {rag_t[0]}/{rag_t[1]}={p3:.1%} vs Soc {rag_s[0]}/{rag_s[1]}={p4:.1%}, z={z2:.2f}, p={pv2:.4f}")
print(f"  Gap: {gap_before:.1f} -> {gap_after:.1f} pp, closure={closure:.0f}%")

# Cross-model
for model_name, fname in [("Llama", None), ("Qwen3", "german_qwen_progress.json"), ("gpt-oss", "german_gptoss_progress.json")]:
    if fname:
        with open(f"data/german/{fname}", encoding='utf-8') as f:
            md = json.load(f)
        mt = [r for r in md['results'] if r['domain']=='tax']
        ms = [r for r in md['results'] if r['domain']=='sozialrecht']
    else:
        mt, ms = tax_de, soc_de
    mt_unc, _ = compute_stats(mt)
    ms_unc, _ = compute_stats(ms)
    p1,p2,z,pv = ztest(mt_unc[0],mt_unc[1],ms_unc[0],ms_unc[1])

    # Pooled accuracy
    all_sc = defaultdict(list)
    for r in mt + ms:
        for p in r.get('probes', []):
            if p.get('type') == 'citation_recall': continue
            s = p.get('scores', {})
            if not isinstance(s, dict): continue
            for m in ['correct_test', 'correct_citations', 'correct_statute', 'specificity']:
                v = s.get(m)
                if isinstance(v, dict): v = v.get('score')
                if v: all_sc[m].append(v)
    print(f"\n{model_name}: Tax {mt_unc[0]}/{mt_unc[1]}={p1:.1%} vs Soc {ms_unc[0]}/{ms_unc[1]}={p2:.1%}, z={z:.2f}, p={pv:.4f}")
    print(f"  Pooled: test={avg(all_sc['correct_test']):.2f} cit={avg(all_sc['correct_citations']):.2f} stat={avg(all_sc['correct_statute']):.2f} spec={avg(all_sc['specificity']):.2f}")
