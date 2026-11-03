"""
Multi-model experiment: Run German 200+200 probes on Qwen3-32B.
Reuses existing ground truths from Llama runs. Only redoes blind answer + judge.
"""
import json, os, re, time, sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "qwen/qwen3-32b"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
PROGRESS_FILE = os.path.join(DATA_DIR, "other", "german_qwen_progress.json")

BSG_PROGRESS = os.path.join(DATA_DIR, "other", "bsg_probe_200_progress.json")
TAX_PROGRESS = os.path.join(DATA_DIR, "german", "german_500_progress.json")


def strip_think(text):
    """Remove <think>...</think> reasoning tags from Qwen output."""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def ask(prompt, max_tokens=2000, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, max_tokens=max_tokens,
            )
            return strip_think(r.choices[0].message.content or "")
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 15 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"
    return "ERROR: max retries exceeded"


PROBE_PROMPT = """Du bist ein deutscher Rechtsexperte. Antworte präzise. Zitiere konkrete Fälle, Gesetze und Rechtsprüfungsmaßstäbe. Falls unsicher, sag es.

FRAGE: {question}"""

JUDGE_PROMPT = """Du bist ein strenger Prüfer für deutsche Rechtsgenauigkeit. Vergleiche die BLINDE ANTWORT mit der MUSTERLÖSUNG.

FRAGE: {question}

MUSTERLÖSUNG:
{ground_truth}

BLINDE ANTWORT:
{blind_answer}

Bewerte 1-5 für jedes Kriterium mit einer kurzen Begründung:
1. CORRECT_TEST: Richtiger Rechtsprüfungsmaßstab?
2. CORRECT_CITATIONS: Echte, relevante Fälle zitiert? (1=halluziniert, 5=korrekt)
3. CORRECT_STATUTE: Korrekte Gesetzesvorschrift?
4. SPECIFICITY: Spezifisch und detailliert?
5. ADMITTED_UNCERTAINTY: yes/no

Gib NUR gültiges JSON zurück:
{{"correct_test": {{"score": N, "reason": "..."}}, "correct_citations": {{"score": N, "reason": "..."}}, "correct_statute": {{"score": N, "reason": "..."}}, "specificity": {{"score": N, "reason": "..."}}, "admitted_uncertainty": "yes or no"}}"""


def load_existing_probes():
    with open(BSG_PROGRESS, encoding='utf-8', errors='replace') as f:
        bsg = json.load(f)
    bsg_results = bsg['results'][:200]

    with open(TAX_PROGRESS, encoding='utf-8', errors='replace') as f:
        tax_data = json.load(f)
    tax_results = [r for r in tax_data['results'] if r.get('domain') == 'tax'][:200]

    print(f"Loaded {len(tax_results)} tax, {len(bsg_results)} BSG (Llama probes)")
    return tax_results, bsg_results


def main():
    tax_probes, bsg_probes = load_existing_probes()

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding='utf-8') as f:
            progress = json.load(f)
    else:
        progress = {"model": MODEL, "done_keys": [], "results": []}

    done_keys = set(progress["done_keys"])
    results = progress["results"]

    all_probes = [(r, 'tax') for r in tax_probes] + [(r, 'sozialrecht') for r in bsg_probes]
    print(f"Total: {len(all_probes)} cases, {len(done_keys)} already done")
    print(f"Model: {MODEL}\n")

    for i, (orig_result, domain) in enumerate(all_probes):
        case_key = orig_result['case_key']
        if case_key in done_keys:
            continue

        gt = orig_result['ground_truth']
        orig_probe_list = orig_result.get('probes', [])
        if not orig_probe_list:
            continue

        print(f"  [{len(results)+1}/{len(all_probes)}] {domain} | {case_key}")

        new_probes = []
        for orig_probe in orig_probe_list:
            question = orig_probe['question']

            blind = ask(PROBE_PROMPT.format(question=question))

            gt_info = {
                "legal_test": gt.get("legal_test_name", ""),
                "source": gt.get("legal_test_source", ""),
                "statutes": gt.get("key_statutes", []),
                "cases": gt.get("key_cases_cited", []),
            }
            score_raw = ask(JUDGE_PROMPT.format(
                question=question,
                ground_truth=json.dumps(gt_info),
                blind_answer=blind,
            ))
            try:
                score = json.loads(score_raw[score_raw.find("{"):score_raw.rfind("}") + 1])
            except Exception:
                score = {}

            new_probes.append({
                "type": orig_probe.get("type", ""),
                "question": question,
                "blind_answer": blind,
                "scores": score,
            })

        results.append({
            "case_key": case_key,
            "domain": domain,
            "ground_truth": gt,
            "probes": new_probes,
        })
        done_keys.add(case_key)
        progress["done_keys"] = list(done_keys)
        progress["results"] = results
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False)

        if len(results) % 20 == 0:
            print(f"    checkpoint: {len(results)} done")

    # Stats
    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} cases ({MODEL})")
    print(f"{'='*60}")

    from collections import defaultdict
    for domain in ['tax', 'sozialrecht']:
        dom_results = [r for r in results if r['domain'] == domain]
        scores = defaultdict(list)
        unc = [0, 0]
        for r in dom_results:
            for probe in r.get('probes', []):
                s = probe.get('scores', {})
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
        print(f"\n--- {domain} (n={len(dom_results)}) ---")
        for m in ['correct_test', 'correct_citations', 'correct_statute', 'specificity']:
            vals = scores[m]
            if vals:
                print(f"  {m}: {sum(vals)/len(vals):.2f} (n={len(vals)})")
        if unc[1]:
            print(f"  admitted_uncertainty: {unc[0]}/{unc[1]} = {unc[0]/unc[1]:.1%}")


if __name__ == "__main__":
    main()
