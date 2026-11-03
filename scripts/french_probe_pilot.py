"""
French pilot probe: 10 tax + 10 social security cases from Conseil d'État.
Pipeline: extract ground truth (70B) → generate questions → blind answer (70B) → judge (70B).
"""
import json, os, time, sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, os.pardir, "data", "french")
CASES_FILE = os.path.join(DATA_DIR, "french_200_cases.json")
PROGRESS_FILE = os.path.join(DATA_DIR, "french_200_progress.json")


def ask(prompt, max_tokens=1500, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"
    return "ERROR: max retries exceeded"


# --- Step 1: Extract ground truth ---

GT_PROMPT = """You are a French legal expert. Analyze this Conseil d'État decision and extract structured ground truth.

DECISION TEXT:
{text}

Return ONLY valid JSON with these fields:
{{
  "core_legal_issue": "the main legal question decided",
  "legal_test_name": "name of the legal test or standard applied, if any",
  "legal_test_source": "source case or article for the test",
  "key_statutes": ["list of statutes/code articles cited, e.g. 'article 199 du code général des impôts'"],
  "key_cases_cited": ["list of case citations referenced"],
  "court": "Conseil d'État",
  "outcome": "outcome of the decision",
  "legal_domain": "fiscal / aide sociale / sécurité sociale / etc."
}}"""


def extract_ground_truth(case):
    text = case['text']
    if len(text) > 12000:
        text = text[:12000]
    raw = ask(GT_PROMPT.format(text=text), max_tokens=2000)
    try:
        return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return {"raw": raw, "error": "parse_failed"}


# --- Step 2: Generate questions ---

def generate_questions(gt):
    questions = []
    if gt.get("legal_test_name") and gt.get("core_legal_issue"):
        questions.append({
            "type": "legal_test",
            "question": f"En droit administratif français, quel est le test juridique applicable pour {gt['core_legal_issue'].rstrip('.')} ? Nommez le test, ses éléments, et la jurisprudence de référence.",
        })
    if gt.get("key_statutes") and isinstance(gt["key_statutes"], list) and gt["key_statutes"]:
        stat = gt["key_statutes"][0]
        questions.append({
            "type": "statute",
            "question": f"Que prévoit {stat} en droit français ? Expliquez son objet et la manière dont le Conseil d'État l'applique.",
        })
    if gt.get("key_cases_cited") and isinstance(gt["key_cases_cited"], list) and len(gt["key_cases_cited"]) >= 1:
        questions.append({
            "type": "citation_recall",
            "question": f"Qu'a décidé le Conseil d'État dans l'affaire {gt['key_cases_cited'][0]} ? Expliquez la solution retenue et sa portée.",
        })
    if gt.get("court"):
        questions.append({
            "type": "procedural",
            "question": f"Quel est le contrôle exercé par le Conseil d'État en cassation sur les décisions des cours administratives d'appel en matière de {gt.get('legal_domain', 'contentieux administratif')} ? Citez la jurisprudence applicable.",
        })
    return questions


# --- Step 3: Blind answer + judge ---

PROBE_PROMPT = """You are a French administrative law expert. Answer precisely in French. Cite specific cases, statutes, and legal tests by name. If unsure, say so.

QUESTION: {question}"""

JUDGE_PROMPT = """You are a strict legal accuracy judge for French administrative law. Compare the BLIND ANSWER against GROUND TRUTH.

QUESTION: {question}

GROUND TRUTH:
{ground_truth}

BLIND ANSWER:
{blind_answer}

Score 1-5 on each criterion with a one-line reason:
1. CORRECT_TEST: Correct legal test identified?
2. CORRECT_CITATIONS: Real, relevant cases cited? (1=hallucinated, 5=accurate)
3. CORRECT_STATUTE: Correct statute/section?
4. SPECIFICITY: Specific and detailed?
5. ADMITTED_UNCERTAINTY: yes/no

Return ONLY valid JSON:
{{"correct_test": {{"score": N, "reason": "..."}}, "correct_citations": {{"score": N, "reason": "..."}}, "correct_statute": {{"score": N, "reason": "..."}}, "specificity": {{"score": N, "reason": "..."}}, "admitted_uncertainty": "yes or no"}}"""


def main():
    with open(CASES_FILE, encoding='utf-8') as f:
        cases = json.load(f)

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding='utf-8') as f:
            progress = json.load(f)
    else:
        progress = {"processed_keys": [], "results": []}

    done_keys = set(progress["processed_keys"])
    results = progress["results"]
    print(f"Loaded {len(cases)} cases, {len(done_keys)} already done\n")

    for i, case in enumerate(cases):
        case_key = case['case_number']
        domain = case['domain']

        if case_key in done_keys:
            print(f"  [{i+1}/{len(cases)}] skip {case_key}")
            continue

        print(f"  [{i+1}/{len(cases)}] {domain} | case {case_key} ({case['text_length']} chars)")

        # Step 1: extract ground truth
        gt = extract_ground_truth(case)
        if gt.get("error"):
            print(f"    GT extraction failed, skipping")
            continue

        # Step 2-3: generate questions, blind answer, judge
        questions = generate_questions(gt)
        probes = []

        for q in questions:
            blind = ask(PROBE_PROMPT.format(question=q["question"]))
            gt_info = {
                "legal_test": gt.get("legal_test_name", ""),
                "source": gt.get("legal_test_source", ""),
                "statutes": gt.get("key_statutes", []),
                "cases": gt.get("key_cases_cited", []),
            }
            score_raw = ask(JUDGE_PROMPT.format(
                question=q["question"],
                ground_truth=json.dumps(gt_info, ensure_ascii=False),
                blind_answer=blind,
            ))
            try:
                score = json.loads(score_raw[score_raw.find("{"):score_raw.rfind("}") + 1])
            except Exception:
                score = {}
            probes.append({
                "type": q["type"],
                "question": q["question"],
                "blind_answer": blind,
                "scores": score,
            })

        results.append({
            "case_key": case_key,
            "domain": domain,
            "ground_truth": gt,
            "probes": probes,
        })
        done_keys.add(case_key)
        progress["processed_keys"] = list(done_keys)
        progress["results"] = results

        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False)
        print(f"    saved ({len(results)} total)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} cases")
    print(f"{'='*60}")

    from collections import defaultdict
    for dom in ["tax", "social"]:
        dom_results = [r for r in results if r["domain"] == dom]
        scores = defaultdict(list)
        unc = [0, 0]
        for r in dom_results:
            for probe in r.get("probes", []):
                s = probe.get("scores", {})
                if not isinstance(s, dict):
                    continue
                for m in ["correct_test", "correct_citations", "correct_statute", "specificity"]:
                    v = s.get(m)
                    if isinstance(v, dict):
                        v = v.get("score")
                    if v:
                        scores[m].append(v)
                au = s.get("admitted_uncertainty")
                unc[1] += 1
                if au in (True, "yes"):
                    unc[0] += 1
        print(f"\n--- {dom} (n={len(dom_results)}) ---")
        for m in ["correct_test", "correct_citations", "correct_statute", "specificity"]:
            vals = scores[m]
            if vals:
                print(f"  {m}: {sum(vals)/len(vals):.2f} (n={len(vals)})")
        if unc[1]:
            print(f"  admitted_uncertainty: {unc[0]}/{unc[1]} = {unc[0]/unc[1]:.1%}")


if __name__ == "__main__":
    main()
