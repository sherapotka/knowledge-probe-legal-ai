"""
Extend canadian_split_progress.json from 40 to 100 cases per domain.
Uses the same split-model design (8B answerer, 70B judge).
Skips already-processed cases automatically.
"""
import json, os, time
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

EXTRACTOR_MODEL = "llama-3.3-70b-versatile"
ANSWERER_MODEL  = "llama-3.1-8b-instant"
JUDGE_MODEL     = "llama-3.3-70b-versatile"
MAX_CASE_CHARS  = 25000
TARGET          = 100
BASE            = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE   = os.path.join(BASE, "canadian_split_progress.json")


def ask(prompt, model, max_tokens=1500, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 10 * (attempt + 1)
                print(f"    Rate limited ({model}), waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"
    return "ERROR: max retries exceeded"


EXTRACTION_PROMPT = """You are a Canadian legal analyst. Read the court decision below and extract the following in valid JSON format. Be precise and cite exactly what the case says.

{{
  "case_name": "short case name, e.g. Smith v Canada",
  "citation": "the neutral citation, e.g. 2020 TCC 45",
  "court": "which court or tribunal decided this",
  "date": "decision date",
  "legal_domain": "tax or refugee or other",
  "legal_test_name": "name of the primary legal test or standard applied",
  "legal_test_source": "the statute section or leading case the test comes from",
  "legal_test_elements": ["list each element or prong of the test"],
  "outcome": "allowed/dismissed/granted/denied",
  "key_statutes": ["list the main statutes cited, with section numbers"],
  "key_cases_cited": ["list up to 5 important case citations"],
  "core_legal_issue": "one sentence describing the central legal question"
}}

Return ONLY valid JSON.

CASE TEXT:
{case_text}"""


def generate_questions(gt):
    questions = []
    if gt.get("legal_test_name") and gt.get("core_legal_issue"):
        questions.append({"type": "legal_test",
            "question": f"In Canadian law, what is the legal test for {gt['core_legal_issue'].rstrip('.')}? Name the test, its elements, and the leading authority."})
    if gt.get("key_statutes"):
        questions.append({"type": "statute",
            "question": f"What does {gt['key_statutes'][0]} provide in Canadian law? Explain its purpose and how courts apply it."})
    if gt.get("key_cases_cited") and len(gt["key_cases_cited"]) >= 2:
        questions.append({"type": "citation_recall",
            "question": f"What did the court decide in {gt['key_cases_cited'][0]}? Explain the holding and significance."})
    if gt.get("court"):
        questions.append({"type": "procedural",
            "question": f"What is the standard of review when a {gt['court']} decision is appealed? Cite the relevant authority."})
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


def process_case(case_key, case_data):
    text = case_data.get("text", "")[:MAX_CASE_CHARS]
    gt_raw = ask(EXTRACTION_PROMPT.format(case_text=text), EXTRACTOR_MODEL, max_tokens=1000)
    try:
        gt = json.loads(gt_raw[gt_raw.find("{"):gt_raw.rfind("}")+1])
    except Exception:
        return None
    questions = generate_questions(gt)
    probes = []
    for q in questions:
        blind = ask(PROBE_PROMPT.format(question=q["question"]), ANSWERER_MODEL)
        gt_info = {"legal_test": gt.get("legal_test_name",""), "source": gt.get("legal_test_source",""),
                   "statutes": gt.get("key_statutes",[]), "cases": gt.get("key_cases_cited",[])}
        score_raw = ask(JUDGE_PROMPT.format(question=q["question"],
            ground_truth=json.dumps(gt_info), blind_answer=blind), JUDGE_MODEL)
        try:
            score = json.loads(score_raw[score_raw.find("{"):score_raw.rfind("}")+1])
        except Exception:
            score = {}
        probes.append({"type": q["type"], "question": q["question"], "blind_answer": blind, "scores": score})
    domain = case_data.get("domain", case_key.split("/")[0])
    return {"case_key": case_key, "domain": domain, "ground_truth": gt, "probes": probes}


def print_stats(results):
    from collections import defaultdict
    scores = defaultdict(lambda: defaultdict(list))
    unc = defaultdict(lambda: [0, 0])
    for r in results:
        d = r.get("domain", "?")
        for probe in r.get("probes", []):
            s = probe.get("scores", {})
            if not isinstance(s, dict): continue
            for m in ["correct_test","correct_citations","correct_statute","specificity"]:
                v = s.get(m)
                if isinstance(v, dict): v = v.get("score")
                if v: scores[d][m].append(v)
            au = s.get("admitted_uncertainty")
            unc[d][1] += 1
            if au in (True, "yes"): unc[d][0] += 1
    for d in sorted(scores):
        n = sum(1 for r in results if r.get("domain") == d)
        print(f"\n--- {d} (n={n} cases) ---")
        for m in ["correct_test","correct_citations","correct_statute","specificity"]:
            vals = scores[d][m]
            if vals: print(f"  {m}: {sum(vals)/len(vals):.2f}")
        if unc[d][1]:
            print(f"  admitted_uncertainty: {unc[d][0]}/{unc[d][1]} = {unc[d][0]/unc[d][1]:.1%}")


def main():
    with open(os.path.join(BASE, "canadian_500_cases.json"), encoding="utf-8", errors="replace") as f:
        all_cases = json.load(f)

    tax_keys     = [k for k in all_cases if k.startswith("tax/")][:TARGET]
    refugee_keys = [k for k in all_cases if k.startswith("refugee/")][:TARGET]
    all_keys     = tax_keys + refugee_keys
    print(f"Target: {len(tax_keys)} tax + {len(refugee_keys)} refugee = {len(all_keys)} cases")

    with open(PROGRESS_FILE, encoding="utf-8", errors="replace") as f:
        progress = json.load(f)

    done_keys = set(progress["processed_keys"])
    results   = progress["results"]
    print(f"Already done: {len(done_keys)} cases — will process {len(all_keys)-len(done_keys)} new\n")

    for i, case_key in enumerate(all_keys):
        if case_key in done_keys:
            print(f"  [{i+1}/{len(all_keys)}] skip {case_key}")
            continue
        case_data = all_cases[case_key]
        text_len = len(case_data.get("text","")[:MAX_CASE_CHARS])
        print(f"  [{i+1}/{len(all_keys)}] {case_key} ({text_len} chars)")
        result = process_case(case_key, case_data)
        if result is None:
            print("    extraction failed, skip")
            continue
        results.append(result)
        done_keys.add(case_key)
        progress["processed_keys"] = list(done_keys)
        progress["results"] = results
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)
        print(f"    saved ({len(results)} total)")

    print(f"\n=== DONE: {len(results)} cases processed ===")
    print_stats(results)


if __name__ == "__main__":
    main()
