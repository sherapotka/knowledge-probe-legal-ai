"""
Fetch 100 Canadian SST (Social Security Tribunal) cases from A2AJ and run probe.
Split-model: Llama 3.1 8B answerer + Llama 3.3 70B extractor/judge.
Output: sst_cases.json, sst_probe_progress.json
"""
import json, os, time, requests
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
CASES_FILE      = os.path.join(BASE, "sst_cases.json")
PROGRESS_FILE   = os.path.join(BASE, "sst_probe_progress.json")
A2AJ_BASE       = "https://api.a2aj.ca"

SST_QUERIES = [
    "Social Security Tribunal CPP disability",
    "Social Security Tribunal employment insurance",
    "SST disability pension",
    "SST old age security",
]


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


def fetch_sst_cases(target):
    cases = {}
    seen_citations = set()

    for query in SST_QUERIES:
        if len(cases) >= target:
            break
        print(f"  Query: '{query}'")
        for page in range(20):
            if len(cases) >= target:
                break
            try:
                resp = requests.get(f"{A2AJ_BASE}/search", params={
                    "query": query, "doc_type": "cases", "size": 50,
                    "search_language": "en", "sort_results": "newest_first", "page": page,
                }, timeout=15)
                resp.raise_for_status()
                results = resp.json().get("results", [])
            except Exception as e:
                print(f"    Error page {page}: {e}")
                break
            if not results:
                break

            for case in results:
                if len(cases) >= target:
                    break
                citation = case.get("citation_en", "")
                if not citation or citation in seen_citations:
                    continue
                # Only SST cases
                if "SST" not in citation:
                    continue
                # Fetch full text
                try:
                    fr = requests.get(f"{A2AJ_BASE}/fetch", params={
                        "citation": citation, "doc_type": "cases", "output_language": "en",
                    }, timeout=15)
                    fd = fr.json().get("results", [])
                    text = fd[0].get("unofficial_text_en", "") if fd else ""
                except Exception:
                    continue
                if not text or len(text) < 500:
                    continue
                seen_citations.add(citation)
                i = len(cases)
                key = f"sst/{i:04d}_{citation.replace(' ', '_')}"
                cases[key] = {
                    "domain": "sst", "title": f"{case.get('name_en', '')} ({citation})",
                    "citation": citation, "date": case.get("document_date_en", ""),
                    "court": case.get("tribunal_name_en", ""), "chars": len(text), "text": text,
                }
                print(f"    [{len(cases)}/{target}] {citation} ({len(text)} chars)")
                time.sleep(1)
            time.sleep(0.5)

    return cases


EXTRACTION_PROMPT = """You are a Canadian legal analyst. Read the court decision below and extract the following in valid JSON format. Be precise and cite exactly what the case says.

{{
  "case_name": "short case name",
  "citation": "the neutral citation, e.g. 2024 SST 100",
  "court": "Social Security Tribunal of Canada",
  "date": "decision date",
  "legal_domain": "social security",
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


def main():
    # Fetch cases
    if os.path.exists(CASES_FILE):
        with open(CASES_FILE, encoding="utf-8", errors="replace") as f:
            all_cases = json.load(f)
        print(f"Loaded {len(all_cases)} cases")
    else:
        print(f"Fetching {TARGET} SST cases from A2AJ...")
        all_cases = fetch_sst_cases(TARGET)
        with open(CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_cases)} cases")

    if not all_cases:
        print("No cases. Exiting.")
        return

    # Load progress
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8", errors="replace") as f:
            progress = json.load(f)
    else:
        progress = {"processed_keys": [], "results": []}

    done_keys = set(progress["processed_keys"])
    results = progress["results"]
    print(f"Already done: {len(done_keys)}/{len(all_cases)} — will process {len(all_cases) - len(done_keys)} new\n")

    keys = list(all_cases.keys())
    for i, case_key in enumerate(keys):
        if case_key in done_keys:
            print(f"  [{i+1}/{len(keys)}] skip {case_key}")
            continue
        case_data = all_cases[case_key]
        text = case_data.get("text", "")[:MAX_CASE_CHARS]
        print(f"  [{i+1}/{len(keys)}] {case_key} ({len(text)} chars)")

        gt_raw = ask(EXTRACTION_PROMPT.format(case_text=text), EXTRACTOR_MODEL, max_tokens=1000)
        try:
            gt = json.loads(gt_raw[gt_raw.find("{"):gt_raw.rfind("}") + 1])
        except Exception:
            print("    extraction failed, skip")
            continue

        probes = []
        for q in generate_questions(gt):
            blind = ask(PROBE_PROMPT.format(question=q["question"]), ANSWERER_MODEL)
            gt_info = {"legal_test": gt.get("legal_test_name", ""), "source": gt.get("legal_test_source", ""),
                       "statutes": gt.get("key_statutes", []), "cases": gt.get("key_cases_cited", [])}
            score_raw = ask(JUDGE_PROMPT.format(question=q["question"],
                ground_truth=json.dumps(gt_info), blind_answer=blind), JUDGE_MODEL)
            try:
                score = json.loads(score_raw[score_raw.find("{"):score_raw.rfind("}") + 1])
            except Exception:
                score = {}
            probes.append({"type": q["type"], "question": q["question"], "blind_answer": blind, "scores": score})

        results.append({"case_key": case_key, "domain": "sst", "ground_truth": gt, "probes": probes})
        done_keys.add(case_key)
        progress["processed_keys"] = list(done_keys)
        progress["results"] = results
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)
        print(f"    saved ({len(results)} total)")

    print(f"\n=== DONE: {len(results)} cases ===")
    from collections import defaultdict
    scores = defaultdict(list); unc = [0, 0]
    for r in results:
        for probe in r.get("probes", []):
            s = probe.get("scores", {})
            if not isinstance(s, dict): continue
            for m in ["correct_test", "correct_citations", "correct_statute", "specificity"]:
                v = s.get(m)
                if isinstance(v, dict): v = v.get("score")
                if v: scores[m].append(v)
            au = s.get("admitted_uncertainty"); unc[1] += 1
            if au in (True, "yes"): unc[0] += 1
    for m in ["correct_test", "correct_citations", "correct_statute", "specificity"]:
        vals = scores[m]
        if vals: print(f"  {m}: {sum(vals)/len(vals):.2f} (n={len(vals)})")
    if unc[1]: print(f"  admitted_uncertainty: {unc[0]}/{unc[1]} = {unc[0]/unc[1]:.1%}")


if __name__ == "__main__":
    main()
