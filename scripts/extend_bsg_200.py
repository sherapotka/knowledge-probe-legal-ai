"""
Extend BSG to 200 cases. We already have 100 probed in bsg_probe_progress.json.
Fetch 100 more from openlegaldata.io (skipping existing), probe them, append to progress.
"""
import json, os, re, time, requests, sys
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"
MAX_CASE_CHARS = 25000
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "other")
CASES_FILE = os.path.join(DATA_DIR, "bsg_cases_200.json")
PROGRESS_FILE = os.path.join(DATA_DIR, "bsg_probe_200_progress.json")
OLD_CASES_FILE = os.path.join(DATA_DIR, "bsg_cases.json")
OLD_PROGRESS_FILE = os.path.join(DATA_DIR, "bsg_probe_progress.json")
API_BASE = "https://de.openlegaldata.io/api/cases/"
TARGET = 200


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


def strip_html(html):
    text = re.sub(r'<br\s*/?>', '\n', html)
    text = re.sub(r'<[^>]+>', ' ', text)
    for entity, char in [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                          ('&quot;', '"'), ('&auml;', 'ä'), ('&ouml;', 'ö'),
                          ('&uuml;', 'ü'), ('&szlig;', 'ß'), ('&Auml;', 'Ä'),
                          ('&Ouml;', 'Ö'), ('&Uuml;', 'Ü')]:
        text = text.replace(entity, char)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch_bsg_cases(target, existing_ids):
    cases = {}
    page = 1
    page_size = 100

    while len(cases) < target:
        print(f"  Page {page} (have {len(cases)} new cases)...")
        try:
            resp = requests.get(API_BASE, params={
                "court__slug": "bsg",
                "page_size": page_size,
                "page": page,
            }, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error page {page}: {e}")
            time.sleep(2)
            page += 1
            continue

        results = data.get("results", [])
        if not results:
            break

        for case in results:
            if len(cases) >= target:
                break
            case_id = str(case["id"])
            slug = case.get("slug", case_id)
            key = f"sozialrecht/{slug}"
            if key in existing_ids:
                continue
            content = case.get("content", "")
            text = strip_html(content)
            if len(text) < 500:
                continue
            cases[key] = {
                "domain": "sozialrecht",
                "case_id": case_id,
                "title": f"BSG {case.get('file_number', '')} ({case.get('date', '')})",
                "court": "Bundessozialgericht",
                "date": case.get("date", ""),
                "reference": case.get("file_number", ""),
                "ecli": case.get("ecli", ""),
                "chars": len(text),
                "text": text,
            }

        total = data.get("count", 0)
        if page * page_size >= total:
            break
        page += 1
        time.sleep(0.5)

    return cases


EXTRACTION_PROMPT = """Du bist ein deutscher Rechtsanalyst. Lies die folgende Gerichtsentscheidung und extrahiere die folgenden Informationen im JSON-Format. Sei präzise und zitiere genau, was im Fall steht.

{{
  "case_name": "kurzer Fallname, z.B. BSG B 4 AS 1/20 R",
  "citation": "ECLI oder Aktenzeichen",
  "court": "welches Gericht hat entschieden",
  "date": "Entscheidungsdatum",
  "legal_domain": "Sozialrecht",
  "legal_test_name": "Name des primären Rechtsprüfungsmaßstabs",
  "legal_test_source": "die Gesetzesvorschrift oder der Leitfall",
  "legal_test_elements": ["Liste jedes Elements des Tests"],
  "outcome": "zugelassen/abgewiesen/aufgehoben/bestätigt",
  "key_statutes": ["Liste der Hauptgesetze mit Paragraphen, z.B. § 7 SGB II"],
  "key_cases_cited": ["Liste bis zu 5 wichtige Fallzitate"],
  "core_legal_issue": "ein Satz über die zentrale Rechtsfrage"
}}

Gib NUR gültiges JSON zurück.

FALLTEXT:
{case_text}"""


def generate_questions(gt):
    questions = []
    if gt.get("legal_test_name") and gt.get("core_legal_issue"):
        questions.append({
            "type": "legal_test",
            "question": f"Was ist im deutschen Recht der Prüfungsmaßstab für: {gt['core_legal_issue'].rstrip('.')}? Nennen Sie den Test, seine Elemente und die maßgebliche Rechtsgrundlage.",
        })
    if gt.get("key_statutes"):
        stat = gt["key_statutes"][0] if isinstance(gt["key_statutes"], list) else gt["key_statutes"]
        questions.append({
            "type": "statute",
            "question": f"Was regelt {stat} im deutschen Sozialrecht? Erläutern Sie Zweck und Anwendung durch die Gerichte.",
        })
    if gt.get("key_cases_cited") and isinstance(gt["key_cases_cited"], list) and len(gt["key_cases_cited"]) >= 2:
        questions.append({
            "type": "citation_recall",
            "question": f"Was hat das Gericht in {gt['key_cases_cited'][0]} entschieden? Erläutern Sie das Ergebnis und die Bedeutung.",
        })
    if gt.get("court"):
        questions.append({
            "type": "procedural",
            "question": f"Welcher Prüfungsmaßstab gilt, wenn eine Entscheidung des {gt['court']} angefochten wird? Nennen Sie die maßgebliche Rechtsgrundlage.",
        })
    return questions


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


def main():
    # Load existing 100 cases and progress
    if os.path.exists(OLD_CASES_FILE):
        with open(OLD_CASES_FILE, encoding="utf-8", errors="replace") as f:
            old_cases = json.load(f)
        print(f"Existing BSG cases: {len(old_cases)}")
    else:
        old_cases = {}

    if os.path.exists(OLD_PROGRESS_FILE):
        with open(OLD_PROGRESS_FILE, encoding="utf-8", errors="replace") as f:
            old_progress = json.load(f)
        print(f"Existing probed: {len(old_progress.get('results', []))}")
    else:
        old_progress = {"processed_keys": [], "results": []}

    # Initialize new progress from old
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8", errors="replace") as f:
            progress = json.load(f)
    else:
        progress = {
            "processed_keys": list(old_progress.get("processed_keys", [])),
            "results": list(old_progress.get("results", [])),
        }

    done_keys = set(progress["processed_keys"])
    results = progress["results"]
    print(f"Total probed so far: {len(results)}")

    # Need 100 more cases
    need = TARGET - len(old_cases)
    if need > 0:
        existing_ids = set(old_cases.keys()) if isinstance(old_cases, dict) else set()
        print(f"\nFetching {need} more BSG cases...")
        new_cases = fetch_bsg_cases(need, existing_ids)
        print(f"Fetched {len(new_cases)} new cases")

        # Merge
        all_cases = dict(old_cases) if isinstance(old_cases, dict) else {}
        all_cases.update(new_cases)
        with open(CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_cases)} total cases")
    else:
        all_cases = old_cases

    # Probe new cases
    keys = list(all_cases.keys())
    for i, case_key in enumerate(keys):
        if case_key in done_keys:
            continue

        case_data = all_cases[case_key]
        text = case_data.get("text", "")[:MAX_CASE_CHARS]
        print(f"  [{len(results)+1}/{TARGET}] {case_key} ({len(text)} chars)")

        gt_raw = ask(EXTRACTION_PROMPT.format(case_text=text), max_tokens=1000)
        try:
            gt = json.loads(gt_raw[gt_raw.find("{"):gt_raw.rfind("}")+1])
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
                score = json.loads(score_raw[score_raw.find("{"):score_raw.rfind("}")+1])
            except Exception:
                score = {}
            probes.append({"type": q["type"], "question": q["question"], "blind_answer": blind, "scores": score})

        results.append({"case_key": case_key, "domain": "sozialrecht", "ground_truth": gt, "probes": probes})
        done_keys.add(case_key)
        progress["processed_keys"] = list(done_keys)
        progress["results"] = results
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)
        print(f"    saved ({len(results)} total)")

    # Stats
    print(f"\n=== DONE: {len(results)} BSG cases ===")
    from collections import defaultdict
    scores = defaultdict(list)
    unc = [0, 0]
    for r in results:
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
    for m in ["correct_test", "correct_citations", "correct_statute", "specificity"]:
        vals = scores[m]
        if vals:
            print(f"  {m}: {sum(vals)/len(vals):.2f} (n={len(vals)})")
    if unc[1]:
        print(f"  admitted_uncertainty: {unc[0]}/{unc[1]} = {unc[0]/unc[1]:.1%}")


if __name__ == "__main__":
    main()
