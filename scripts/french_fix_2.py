"""Fetch and probe 2 more French social cases to reach 200."""
import requests, zipfile, io, json, sys, re, os, time
from lxml import etree
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "french")

with open(os.path.join(DATA_DIR, "french_200_cases.json"), encoding='utf-8') as f:
    existing = json.load(f)
existing_nums = set(c['case_number'] for c in existing)

SOCIAL_KW = ['sécurité sociale', 'aide sociale', 'RSA', 'revenu de solidarité active',
    'AAH', 'MDPH', 'APL', 'protection sociale', 'prestation sociale',
    "code de l'action sociale", 'allocation de solidarité', 'handicap']
TAX_KW = ['impôt sur le revenu', 'impôt sur les sociétés', 'TVA',
    'code général des impôts', 'contributions directes', 'taxe foncière',
    'fiscal', 'fiscalité']


def clean_html(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


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
                time.sleep(10 * (attempt + 1))
            else:
                return f"ERROR: {e}"
    return "ERROR: max retries"


# Find 2 new social cases from months we haven't used
found = []
for year, month in [(2023, 12), (2023, 11), (2023, 10)]:
    if len(found) >= 2:
        break
    url = f'https://opendata.justice-administrative.fr/DCE/{year}/{month:02d}/CE_{year}{month:02d}.zip'
    print(f'Downloading {url}...')
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        continue
    z = zipfile.ZipFile(io.BytesIO(r.content))
    parser = etree.XMLParser(recover=True)
    for name in z.namelist():
        if len(found) >= 2:
            break
        xml_bytes = z.read(name)
        if not xml_bytes.strip():
            continue
        try:
            root = etree.fromstring(xml_bytes, parser)
        except Exception:
            continue
        case_num = root.findtext('.//Numero_Dossier', '')
        if case_num in existing_nums:
            continue
        ti = root.find('.//Texte_Integral')
        if ti is None:
            continue
        texte = etree.tostring(ti, method='text', encoding='unicode')
        text_lower = texte.lower()
        social_score = sum(1 for kw in SOCIAL_KW if kw.lower() in text_lower)
        tax_score = sum(1 for kw in TAX_KW if kw.lower() in text_lower)
        if social_score >= 2 and tax_score == 0 and len(texte) >= 3000:
            found.append({
                'case_number': case_num,
                'domain': 'social',
                'date': root.findtext('.//Date_Lecture', ''),
                'text': clean_html(etree.tostring(ti, encoding='unicode')),
                'text_length': len(texte),
            })
            print(f'  Found social case {case_num} ({len(texte)} chars)')

print(f'Found {len(found)} new social cases')

GT_PROMPT = """You are a French legal expert. Analyze this Conseil d'État decision and extract structured ground truth.

DECISION TEXT:
{text}

Return ONLY valid JSON with these fields:
{{"core_legal_issue": "...", "legal_test_name": "...", "legal_test_source": "...", "key_statutes": ["..."], "key_cases_cited": ["..."], "court": "Conseil d'État", "outcome": "...", "legal_domain": "..."}}"""

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

progress_path = os.path.join(DATA_DIR, "french_200_progress.json")
with open(progress_path, encoding='utf-8') as f:
    progress = json.load(f)

for case in found:
    text = case['text'][:12000]
    print(f"\nProbing {case['case_number']}...")
    gt_raw = ask(GT_PROMPT.format(text=text), max_tokens=2000)
    try:
        gt = json.loads(gt_raw[gt_raw.find("{"):gt_raw.rfind("}") + 1])
    except Exception:
        print("  GT failed, skip")
        continue

    questions = []
    if gt.get("legal_test_name") and gt.get("core_legal_issue"):
        questions.append({"type": "legal_test",
            "question": f"En droit administratif français, quel est le test juridique applicable pour {gt['core_legal_issue'].rstrip('.')} ? Nommez le test, ses éléments, et la jurisprudence de référence."})
    if gt.get("key_statutes") and isinstance(gt["key_statutes"], list) and gt["key_statutes"]:
        questions.append({"type": "statute",
            "question": f"Que prévoit {gt['key_statutes'][0]} en droit français ? Expliquez son objet et la manière dont le Conseil d'État l'applique."})
    if gt.get("key_cases_cited") and isinstance(gt["key_cases_cited"], list) and len(gt["key_cases_cited"]) >= 1:
        questions.append({"type": "citation_recall",
            "question": f"Qu'a décidé le Conseil d'État dans l'affaire {gt['key_cases_cited'][0]} ? Expliquez la solution retenue et sa portée."})
    questions.append({"type": "procedural",
        "question": f"Quel est le contrôle exercé par le Conseil d'État en cassation sur les décisions des cours administratives d'appel en matière de {gt.get('legal_domain', 'contentieux administratif')} ? Citez la jurisprudence applicable."})

    probes = []
    for q in questions:
        blind = ask(PROBE_PROMPT.format(question=q["question"]))
        gt_info = {"legal_test": gt.get("legal_test_name", ""), "source": gt.get("legal_test_source", ""),
                   "statutes": gt.get("key_statutes", []), "cases": gt.get("key_cases_cited", [])}
        score_raw = ask(JUDGE_PROMPT.format(
            question=q["question"],
            ground_truth=json.dumps(gt_info, ensure_ascii=False),
            blind_answer=blind))
        try:
            score = json.loads(score_raw[score_raw.find("{"):score_raw.rfind("}") + 1])
        except Exception:
            score = {}
        probes.append({"type": q["type"], "question": q["question"], "blind_answer": blind, "scores": score})

    progress["results"].append({"case_key": case['case_number'], "domain": "social", "ground_truth": gt, "probes": probes})
    progress["processed_keys"].append(case['case_number'])
    print(f"  Done, probes: {len(probes)}")

with open(progress_path, 'w', encoding='utf-8') as f:
    json.dump(progress, f, ensure_ascii=False)

social_count = len([r for r in progress['results'] if r['domain'] == 'social'])
tax_count = len([r for r in progress['results'] if r['domain'] == 'tax'])
print(f"\nFinal: {tax_count} tax, {social_count} social")
