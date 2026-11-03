"""
German Auto-Probe at Scale: 500 cases per domain
=================================================
Same pipeline as german_auto_probe.py but:
- Reads from german_500_cases.json
- Handles rate limits with retry logic
- Saves progress incrementally
- Outputs to german_500_probe_results.json
"""

import json
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "llama-3.3-70b-versatile"
MAX_CASE_CHARS = 25000  # slightly lower to stay within token limits

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_FILE = os.path.join(BASE_DIR, "german_500_cases.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "german_500_probe_results.json")
PROGRESS_FILE = os.path.join(BASE_DIR, "german_500_progress.json")


def ask_llama(client, prompt, max_tokens=1500, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"
    return "ERROR: max retries exceeded"


EXTRACTION_PROMPT = """You are a German legal analyst. Read the court decision below and extract the following in valid JSON format. Be precise. Answer in English.

{{
  "case_name": "short identifier",
  "reference": "official Aktenzeichen",
  "court": "court name",
  "date": "decision date",
  "legal_domain": "tax or asylum or other",
  "legal_test_name": "primary legal test (German + English)",
  "legal_test_source": "statute or leading case",
  "legal_test_elements": ["elements of the test"],
  "outcome": "allowed/dismissed/granted/denied/remanded",
  "key_statutes": ["main statutes with § numbers"],
  "key_cases_cited": ["up to 5 case citations"],
  "core_legal_issue": "one sentence (English)"
}}

Return ONLY valid JSON.

CASE TEXT:
{case_text}"""


def generate_questions(gt):
    questions = []
    if gt.get("legal_test_name") and gt.get("core_legal_issue"):
        questions.append({
            "type": "legal_test",
            "question": f"In German law, what is the legal test for {gt['core_legal_issue'].rstrip('.')}? Name the test, its elements, and the leading authority.",
            "ground_truth_test": gt["legal_test_name"],
            "ground_truth_source": gt.get("legal_test_source", ""),
            "ground_truth_elements": gt.get("legal_test_elements", []),
        })
    if gt.get("key_statutes"):
        questions.append({
            "type": "statute",
            "question": f"What does {gt['key_statutes'][0]} provide in German law? Explain its purpose and how courts apply it.",
            "ground_truth_statute": gt["key_statutes"][0],
        })
    if gt.get("key_cases_cited") and len(gt["key_cases_cited"]) >= 2:
        questions.append({
            "type": "citation_recall",
            "question": f"What did the court decide in {gt['key_cases_cited'][0]}? Explain the holding and significance.",
            "ground_truth_citation": gt["key_cases_cited"][0],
        })
    if gt.get("court"):
        questions.append({
            "type": "procedural",
            "question": f"What is the standard of review when a {gt['court']} decision is appealed? Cite the relevant provision.",
            "ground_truth_court": gt["court"],
        })
    return questions


PROBE_PROMPT = """You are an expert in German law. Answer precisely. Cite cases (with Aktenzeichen), statutes (with §), and legal tests. If unsure, say so.

QUESTION: {question}"""


JUDGE_PROMPT = """You are a strict legal accuracy judge for German law. Compare the BLIND ANSWER against GROUND TRUTH.

QUESTION: {question}

GROUND TRUTH:
{ground_truth}

BLIND ANSWER:
{blind_answer}

Score 1-5 on each criterion with a one-line reason:
1. CORRECT_TEST: Correct legal test identified?
2. CORRECT_CITATIONS: Real, relevant cases cited? (1=hallucinated, 5=accurate)
3. CORRECT_STATUTE: Correct statute/paragraph?
4. SPECIFICITY: Specific and detailed?
5. ADMITTED_UNCERTAINTY: yes/no

Return ONLY valid JSON:
{{"correct_test": {{"score": N, "reason": "..."}}, "correct_citations": {{"score": N, "reason": "..."}}, "correct_statute": {{"score": N, "reason": "..."}}, "specificity": {{"score": N, "reason": "..."}}, "admitted_uncertainty": "yes or no"}}"""


def load_progress():
    """Load already-processed case keys."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"processed_keys": [], "results": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False)


def main():
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    client = Groq(api_key=GROQ_API_KEY)
    progress = load_progress()
    processed = set(progress["processed_keys"])
    all_results = progress["results"]

    total = len(cases)
    done = len(processed)
    print(f"Total cases: {total}, already processed: {done}")

    for i, (case_key, case_data) in enumerate(cases.items()):
        if case_key in processed:
            continue

        print(f"\n[{done+1}/{total}] {case_key} ({case_data['domain']}, {case_data.get('chars', '?')} chars)")

        # Step 1: Extract ground truth
        extraction = ask_llama(
            client,
            EXTRACTION_PROMPT.replace("{case_text}", case_data["text"][:MAX_CASE_CHARS]),
            max_tokens=1500,
        )

        try:
            cleaned = extraction.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            gt = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            print(f"  SKIP (bad JSON)")
            processed.add(case_key)
            progress["processed_keys"] = list(processed)
            save_progress(progress)
            continue

        # Step 2: Generate questions
        questions = generate_questions(gt)

        # Step 3: Ask blind + Step 4: Judge
        case_result = {
            "case_key": case_key,
            "domain": case_data["domain"],
            "ground_truth": gt,
            "probes": [],
        }

        for q in questions:
            answer = ask_llama(client, PROBE_PROMPT.replace("{question}", q["question"]))

            # Judge immediately
            gt_info = {k: v for k, v in q.items() if k.startswith("ground_truth")}
            gt_info["case_name"] = gt.get("case_name", "")
            gt_info["legal_test_from_case"] = gt.get("legal_test_name", "")
            gt_info["key_statutes_from_case"] = gt.get("key_statutes", [])
            gt_info["key_cases_from_case"] = gt.get("key_cases_cited", [])

            scores_raw = ask_llama(
                client,
                JUDGE_PROMPT.format(
                    question=q["question"],
                    ground_truth=json.dumps(gt_info, ensure_ascii=False),
                    blind_answer=answer,
                ),
                max_tokens=500,
            )

            try:
                sc = scores_raw.strip()
                if sc.startswith("```"):
                    sc = sc.split("\n", 1)[1]
                    sc = sc.rsplit("```", 1)[0]
                scores = json.loads(sc)
            except (json.JSONDecodeError, IndexError):
                scores = {"error": "parse_failed"}

            q["blind_answer"] = answer
            q["scores"] = scores
            case_result["probes"].append(q)

        all_results.append(case_result)
        processed.add(case_key)
        done += 1

        # Save progress every 5 cases
        if done % 5 == 0:
            progress["processed_keys"] = list(processed)
            progress["results"] = all_results
            save_progress(progress)
            print(f"  [checkpoint saved: {done}/{total}]")

    # Final save
    progress["processed_keys"] = list(processed)
    progress["results"] = all_results
    save_progress(progress)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Generate summary
    domain_scores = {}
    for r in all_results:
        d = r["domain"]
        if d not in domain_scores:
            domain_scores[d] = {"correct_test": [], "correct_citations": [],
                                "correct_statute": [], "specificity": [],
                                "uncertainty_count": 0, "total": 0}
        for probe in r["probes"]:
            s = probe.get("scores", {})
            if "error" in s:
                continue
            for metric in ["correct_test", "correct_citations", "correct_statute", "specificity"]:
                if metric in s and isinstance(s[metric], dict) and "score" in s[metric]:
                    domain_scores[d][metric].append(s[metric]["score"])
            if s.get("admitted_uncertainty") == "yes":
                domain_scores[d]["uncertainty_count"] += 1
            domain_scores[d]["total"] += 1

    print(f"\n\n{'='*60}")
    print("FINAL RESULTS — 500-CASE SCALE")
    print(f"{'='*60}")
    print(f"Total cases processed: {len(all_results)}")

    metrics = ["correct_test", "correct_citations", "correct_statute", "specificity"]
    header = f"{'Metric':<22}"
    for d in domain_scores:
        header += f" | {d:>8}"
    header += " |    Gap"
    print(header)
    print("-" * len(header))

    domain_keys = list(domain_scores.keys())
    for metric in metrics:
        row = f"{metric:<22}"
        avgs = {}
        for d in domain_keys:
            vals = domain_scores[d][metric]
            avg = sum(vals) / len(vals) if vals else 0
            avgs[d] = avg
            row += f" | {avg:8.2f}"
        if len(domain_keys) >= 2:
            gap = avgs.get(domain_keys[0], 0) - avgs.get(domain_keys[-1], 0)
            row += f" | {gap:+6.2f}"
        print(row)

    row = f"{'admitted_uncertainty':<22}"
    for d in domain_keys:
        ds = domain_scores[d]
        pct = ds["uncertainty_count"] / ds["total"] * 100 if ds["total"] else 0
        row += f" | {pct:7.0f}%"
    print(row)

    # Save report
    report_file = os.path.join(BASE_DIR, "german_500_report.md")
    lines = [
        "# German 500-Case Auto-Probe Report",
        f"## {len(all_results)} cases processed",
        "",
        "| Metric | " + " | ".join(domain_keys) + " | Gap |",
        "|--------|" + "|".join(["------"] * len(domain_keys)) + "|-----|",
    ]
    for metric in metrics:
        row = f"| {metric} |"
        avgs = {}
        for d in domain_keys:
            vals = domain_scores[d][metric]
            avg = sum(vals) / len(vals) if vals else 0
            avgs[d] = avg
            row += f" {avg:.2f} |"
        gap = avgs.get(domain_keys[0], 0) - avgs.get(domain_keys[-1], 0) if len(domain_keys) >= 2 else 0
        row += f" {gap:+.2f} |"
        lines.append(row)

    row = "| admitted_uncertainty |"
    for d in domain_keys:
        ds = domain_scores[d]
        pct = ds["uncertainty_count"] / ds["total"] * 100 if ds["total"] else 0
        row += f" {pct:.0f}% |"
    row += " |"
    lines.append(row)

    with open(report_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
