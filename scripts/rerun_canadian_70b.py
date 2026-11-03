"""
Re-run Canadian experiments with 70B throughout (no split-model) to match German pipeline.
- Answerer: Llama 3.3 70B (blind, no case text)
- Judge: Llama 3.3 70B
- Reuses existing ground truth from canadian_split_progress.json (tax)
  and sst_probe_progress.json (SST).
- 100 tax + 100 SST = 200 cases.
- Output: canadian_70b_progress.json
"""
import json, os, random, time
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL          = "llama-3.3-70b-versatile"
BASE           = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE  = os.path.join(BASE, "canadian_70b_progress.json")


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


def generate_questions(gt):
    questions = []
    if gt.get("legal_test_name") and gt.get("core_legal_issue"):
        questions.append({
            "type": "legal_test",
            "question": f"In Canadian law, what is the legal test for {gt['core_legal_issue'].rstrip('.')}? Name the test, its elements, and the leading authority.",
        })
    if gt.get("key_statutes"):
        stat = gt["key_statutes"][0] if isinstance(gt["key_statutes"], list) else gt["key_statutes"]
        questions.append({
            "type": "statute",
            "question": f"What does {stat} provide in Canadian law? Explain its purpose and how courts apply it.",
        })
    if gt.get("key_cases_cited") and isinstance(gt["key_cases_cited"], list) and len(gt["key_cases_cited"]) >= 2:
        questions.append({
            "type": "citation_recall",
            "question": f"What did the court decide in {gt['key_cases_cited'][0]}? Explain the holding and significance.",
        })
    if gt.get("court"):
        questions.append({
            "type": "procedural",
            "question": f"What is the standard of review when a {gt['court']} decision is appealed? Cite the relevant authority.",
        })
    return questions


PROBE_PROMPT = """You are a Canadian legal expert. Answer precisely. Cite specific cases, statutes, and legal tests by name. If unsure, say so.

QUESTION: {question}"""

JUDGE_PROMPT = """You are a strict legal accuracy judge for Canadian law. Compare the BLIND ANSWER against GROUND TRUTH.

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


def load_existing_ground_truths():
    # Tax: 100 from canadian_split_progress
    with open(os.path.join(BASE, "canadian_split_progress.json"), encoding="utf-8", errors="replace") as f:
        csplit = json.load(f)
    tax_results = [r for r in csplit["results"] if r.get("domain") == "tax"][:100]
    print(f"Loaded {len(tax_results)} tax cases")

    # SST: 100 from sst_probe_progress
    with open(os.path.join(BASE, "sst_probe_progress.json"), encoding="utf-8", errors="replace") as f:
        sst = json.load(f)
    sst_results = sst["results"][:100]
    print(f"Loaded {len(sst_results)} SST cases")

    entries = []
    for r in tax_results:
        entries.append({"case_key": r["case_key"], "domain": "tax", "ground_truth": r["ground_truth"]})
    for r in sst_results:
        entries.append({"case_key": r["case_key"], "domain": "sst", "ground_truth": r["ground_truth"]})
    return entries


def main():
    entries = load_existing_ground_truths()

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8", errors="replace") as f:
            progress = json.load(f)
    else:
        progress = {"processed_keys": [], "results": []}

    done_keys = set(progress["processed_keys"])
    results = progress["results"]
    print(f"Already done: {len(done_keys)}/{len(entries)}\n")

    for i, entry in enumerate(entries):
        case_key = entry["case_key"]
        if case_key in done_keys:
            print(f"  [{i+1}/{len(entries)}] skip {case_key}")
            continue

        gt = entry["ground_truth"]
        domain = entry["domain"]
        print(f"  [{i+1}/{len(entries)}] {domain} | {case_key}")

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
                ground_truth=json.dumps(gt_info),
                blind_answer=blind,
            ))
            try:
                score = json.loads(score_raw[score_raw.find("{"):score_raw.rfind("}") + 1])
            except Exception:
                score = {}
            probes.append({"type": q["type"], "question": q["question"], "blind_answer": blind, "scores": score})

        results.append({"case_key": case_key, "domain": domain, "ground_truth": gt, "probes": probes})
        done_keys.add(case_key)
        progress["processed_keys"] = list(done_keys)
        progress["results"] = results
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)
        print(f"    saved ({len(results)} total)")

    print(f"\n=== DONE: {len(results)} cases ===")
    from collections import defaultdict
    for dom in ["tax", "sst"]:
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
