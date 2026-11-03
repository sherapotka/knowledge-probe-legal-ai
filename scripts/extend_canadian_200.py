"""
Extend Canadian experiment to 200 tax + 200 SST with 70B throughout.
We already have 100+100 probed in canadian_70b_progress.json.
Fetch 100 more of each from A2AJ/CanLII, run full pipeline (extract GT + probe + judge).
"""
import json, os, time, requests, sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"
MAX_CASE_CHARS = 25000
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "canadian")
PROGRESS_FILE = os.path.join(DATA_DIR, "canadian_200_progress.json")
OLD_PROGRESS = os.path.join(DATA_DIR, "canadian_70b_progress.json")
A2AJ_BASE = "https://api.a2aj.ca"

TAX_QUERIES = [
    "Income Tax Act assessment",
    "Canada Revenue Agency tax court appeal",
    "tax deduction capital gains",
    "GST HST excise tax",
    "taxpayer reassessment notice",
    "income tax penalty gross negligence",
]

SST_QUERIES = [
    "Social Security Tribunal CPP disability",
    "Social Security Tribunal employment insurance",
    "SST disability pension",
    "SST old age security",
    "CPP disability appeal tribunal",
    "employment insurance benefits appeal",
]


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


def fetch_cases(queries, citation_filter, domain, target, existing_citations):
    cases = {}
    for query in queries:
        if len(cases) >= target:
            break
        print(f"  Query: '{query}'")
        for page in range(30):
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
                if not citation or citation in existing_citations:
                    continue
                if not citation_filter(citation):
                    continue
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
                existing_citations.add(citation)
                key = f"{domain}/{len(cases):04d}_{citation.replace(' ', '_')}"
                cases[key] = {
                    "domain": domain,
                    "title": f"{case.get('name_en', '')} ({citation})",
                    "citation": citation,
                    "date": case.get("document_date_en", ""),
                    "court": case.get("tribunal_name_en", ""),
                    "chars": len(text),
                    "text": text,
                }
                print(f"    [{len(cases)}/{target}] {citation} ({len(text)} chars)")
                time.sleep(1)
            time.sleep(0.5)
    return cases


EXTRACTION_PROMPT = """You are a Canadian legal analyst. Read the court decision below and extract the following in valid JSON format. Be precise and cite exactly what the case says.

{{
  "case_name": "short case name",
  "citation": "the neutral citation",
  "court": "which court decided",
  "date": "decision date",
  "legal_domain": "{domain}",
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


def main():
    # Load existing 70B results (100 tax + 100 sst)
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8", errors="replace") as f:
            progress = json.load(f)
    elif os.path.exists(OLD_PROGRESS):
        with open(OLD_PROGRESS, encoding="utf-8", errors="replace") as f:
            progress = json.load(f)
    else:
        progress = {"processed_keys": [], "results": []}

    done_keys = set(progress["processed_keys"])
    results = progress["results"]
    existing_tax = [r for r in results if r["domain"] == "tax"]
    existing_sst = [r for r in results if r["domain"] == "sst"]
    print(f"Existing: {len(existing_tax)} tax, {len(existing_sst)} SST")

    # Collect existing citations to avoid duplicates
    existing_citations = set()
    for r in results:
        gt = r.get("ground_truth", {})
        if gt.get("citation"):
            existing_citations.add(gt["citation"])

    # Fetch new tax cases
    need_tax = 200 - len(existing_tax)
    need_sst = 200 - len(existing_sst)

    new_tax_cases = {}
    new_sst_cases = {}

    if need_tax > 0:
        print(f"\nFetching {need_tax} more tax cases...")
        new_tax_cases = fetch_cases(
            TAX_QUERIES,
            lambda c: "TCC" in c or "FCA" in c or "Tax" in c,
            "tax", need_tax, existing_citations
        )
        print(f"Fetched {len(new_tax_cases)} new tax cases")

    if need_sst > 0:
        print(f"\nFetching {need_sst} more SST cases...")
        new_sst_cases = fetch_cases(
            SST_QUERIES,
            lambda c: "SST" in c,
            "sst", need_sst, existing_citations
        )
        print(f"Fetched {len(new_sst_cases)} new SST cases")

    # Probe all new cases
    all_new = list(new_tax_cases.items()) + list(new_sst_cases.items())
    print(f"\nProbing {len(all_new)} new cases...")

    for i, (case_key, case_data) in enumerate(all_new):
        if case_key in done_keys:
            continue

        domain = case_data["domain"]
        text = case_data.get("text", "")[:MAX_CASE_CHARS]
        current_tax = len([r for r in results if r["domain"] == "tax"])
        current_sst = len([r for r in results if r["domain"] == "sst"])
        print(f"  [{i+1}/{len(all_new)}] {domain} | {case_key} (tax={current_tax}, sst={current_sst})")

        gt_raw = ask(EXTRACTION_PROMPT.format(case_text=text, domain=domain), max_tokens=1000)
        try:
            gt = json.loads(gt_raw[gt_raw.find("{"):gt_raw.rfind("}") + 1])
        except Exception:
            print("    extraction failed, skip")
            continue

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

    # Stats
    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} total cases")
    print(f"{'='*60}")

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
