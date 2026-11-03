"""
RAG experiment for Germany: Does retrieval-augmented generation close the confidence gap?

Design:
- Probe corpus: existing 200 tax + 200 BSG cases (questions already generated)
- RAG corpus: 100 NEW tax + 100 NEW BSG cases (used ONLY for retrieval)
- No overlap between probe and RAG corpora.

Steps:
1. Fetch 100 new tax + 100 new BSG cases from openlegaldata.io
2. Chunk them into ~500-word passages, build BM25 index
3. For each existing probe question, retrieve top-3 passages
4. Re-run blind answer with retrieved context prepended
5. Re-judge against existing ground truth
6. Compare admitted_uncertainty rates: no-RAG vs RAG
"""
import json, os, re, time, sys, requests
from collections import defaultdict
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
RAG_CASES_FILE = os.path.join(DATA_DIR, "other", "german_rag_corpus.json")
RAG_PROGRESS_FILE = os.path.join(DATA_DIR, "other", "german_rag_progress.json")

# Existing probed results (200 BSG from extend_bsg_200)
BSG_PROGRESS = os.path.join(DATA_DIR, "other", "bsg_probe_200_progress.json")
# German 500 tax progress (we'll use first 200)
TAX_PROGRESS = os.path.join(DATA_DIR, "german", "german_500_progress.json")
# Existing BSG cases (to exclude from RAG fetch)
OLD_BSG_CASES = os.path.join(DATA_DIR, "other", "bsg_cases_200.json")

API_BASE = "https://de.openlegaldata.io/api/cases/"


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


# ── Step 1: Fetch RAG corpus ─────────────────────────────────────────────────

def fetch_rag_corpus():
    """Fetch 100 new tax + 100 new BSG cases, disjoint from probe corpus."""
    if os.path.exists(RAG_CASES_FILE):
        with open(RAG_CASES_FILE, encoding='utf-8') as f:
            corpus = json.load(f)
        tax_count = len([c for c in corpus if c['domain'] == 'tax'])
        bsg_count = len([c for c in corpus if c['domain'] == 'sozialrecht'])
        print(f"RAG corpus loaded: {tax_count} tax, {bsg_count} social")
        if tax_count >= 100 and bsg_count >= 100:
            return corpus

    # Load existing case IDs to exclude
    existing_ids = set()
    if os.path.exists(OLD_BSG_CASES):
        with open(OLD_BSG_CASES, encoding='utf-8', errors='replace') as f:
            old = json.load(f)
        for key in old:
            existing_ids.add(old[key].get('case_id', ''))

    corpus = []

    # Fetch BSG cases
    print("Fetching 100 new BSG cases for RAG corpus...")
    page = 3  # Start from page 3 to skip pages 1-2 used in original fetch
    while len([c for c in corpus if c['domain'] == 'sozialrecht']) < 100:
        try:
            resp = requests.get(API_BASE, params={
                "court__slug": "bsg", "page_size": 100, "page": page,
            }, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error page {page}: {e}")
            page += 1
            continue

        results = data.get("results", [])
        if not results:
            break

        for case in results:
            if len([c for c in corpus if c['domain'] == 'sozialrecht']) >= 100:
                break
            case_id = str(case["id"])
            if case_id in existing_ids:
                continue
            content = case.get("content", "")
            text = strip_html(content)
            if len(text) < 500:
                continue
            existing_ids.add(case_id)
            corpus.append({
                "domain": "sozialrecht",
                "case_id": case_id,
                "text": text,
            })

        total = data.get("count", 0)
        if page * 100 >= total:
            break
        page += 1
        time.sleep(0.5)

    bsg_count = len([c for c in corpus if c['domain'] == 'sozialrecht'])
    print(f"  Got {bsg_count} BSG cases")

    # Fetch tax cases (use Finanzgerichte)
    print("Fetching 100 new tax cases for RAG corpus...")
    tax_courts = ["bfh", "fg-berlin-brandenburg", "fg-muenchen", "fg-koeln",
                  "fg-hamburg", "fg-duesseldorf", "fg-nuernberg"]
    for court_slug in tax_courts:
        if len([c for c in corpus if c['domain'] == 'tax']) >= 100:
            break
        page = 1
        while len([c for c in corpus if c['domain'] == 'tax']) < 100:
            try:
                resp = requests.get(API_BASE, params={
                    "court__slug": court_slug, "page_size": 100, "page": page,
                }, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                break

            results = data.get("results", [])
            if not results:
                break

            for case in results:
                if len([c for c in corpus if c['domain'] == 'tax']) >= 100:
                    break
                case_id = str(case["id"])
                if case_id in existing_ids:
                    continue
                content = case.get("content", "")
                text = strip_html(content)
                if len(text) < 500:
                    continue
                existing_ids.add(case_id)
                corpus.append({
                    "domain": "tax",
                    "case_id": case_id,
                    "text": text,
                })

            total = data.get("count", 0)
            if page * 100 >= total:
                break
            page += 1
            time.sleep(0.5)

    tax_count = len([c for c in corpus if c['domain'] == 'tax'])
    print(f"  Got {tax_count} tax cases")

    with open(RAG_CASES_FILE, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f"Saved RAG corpus: {len(corpus)} cases total")
    return corpus


# ── Step 2: Build BM25 index ─────────────────────────────────────────────────

def chunk_text(text, chunk_size=500):
    """Split text into chunks of ~chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk) > 100:  # skip tiny trailing chunks
            chunks.append(chunk)
    return chunks


def build_index(corpus):
    """Build BM25 index over chunked RAG corpus. Returns (index, chunks, domains)."""
    all_chunks = []
    chunk_domains = []
    for case in corpus:
        for chunk in chunk_text(case['text']):
            all_chunks.append(chunk)
            chunk_domains.append(case['domain'])

    tokenized = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index: {len(all_chunks)} chunks ({len([d for d in chunk_domains if d == 'tax'])} tax, {len([d for d in chunk_domains if d == 'sozialrecht'])} social)")
    return bm25, all_chunks, chunk_domains


def retrieve(bm25, chunks, query, top_k=3):
    """Retrieve top-k chunks for a query."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]


# ── Step 3-5: Re-probe with RAG ──────────────────────────────────────────────

RAG_PROBE_PROMPT = """Du bist ein deutscher Rechtsexperte. Nutze die folgenden Rechtstexte als Kontext und antworte präzise auf die Frage. Zitiere konkrete Fälle, Gesetze und Rechtsprüfungsmaßstäbe. Falls unsicher, sag es.

KONTEXT:
{context}

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
    """Load existing probe results (no-RAG) for both domains."""
    # BSG (200 cases)
    with open(BSG_PROGRESS, encoding='utf-8', errors='replace') as f:
        bsg = json.load(f)
    bsg_results = bsg['results'][:200]

    # Tax (first 200 from german_500)
    with open(TAX_PROGRESS, encoding='utf-8', errors='replace') as f:
        tax_data = json.load(f)
    tax_results = [r for r in tax_data['results'] if r.get('domain') == 'tax'][:200]

    print(f"Loaded {len(tax_results)} tax probes, {len(bsg_results)} BSG probes")
    return tax_results, bsg_results


def run_rag_experiment(bm25, chunks, tax_probes, bsg_probes):
    """Re-run probes with RAG context. Only redo blind answer + judge."""

    if os.path.exists(RAG_PROGRESS_FILE):
        with open(RAG_PROGRESS_FILE, encoding='utf-8') as f:
            progress = json.load(f)
    else:
        progress = {"done_keys": [], "results": []}

    done_keys = set(progress["done_keys"])
    results = progress["results"]

    all_probes = [(r, 'tax') for r in tax_probes] + [(r, 'sozialrecht') for r in bsg_probes]
    print(f"\nRunning RAG probes: {len(all_probes)} cases, {len(done_keys)} already done")

    for i, (orig_result, domain) in enumerate(all_probes):
        case_key = orig_result['case_key']
        if case_key in done_keys:
            continue

        gt = orig_result['ground_truth']
        orig_probes = orig_result.get('probes', [])
        if not orig_probes:
            continue

        print(f"  [{len(results)+1}/{len(all_probes)}] {domain} | {case_key}")

        rag_probes = []
        for orig_probe in orig_probes:
            question = orig_probe['question']

            # Retrieve context
            retrieved = retrieve(bm25, chunks, question, top_k=3)
            context = "\n\n---\n\n".join(retrieved)

            # RAG-augmented blind answer
            rag_answer = ask(RAG_PROBE_PROMPT.format(context=context[:4000], question=question))

            # Judge
            gt_info = {
                "legal_test": gt.get("legal_test_name", ""),
                "source": gt.get("legal_test_source", ""),
                "statutes": gt.get("key_statutes", []),
                "cases": gt.get("key_cases_cited", []),
            }
            score_raw = ask(JUDGE_PROMPT.format(
                question=question,
                ground_truth=json.dumps(gt_info),
                blind_answer=rag_answer,
            ))
            try:
                score = json.loads(score_raw[score_raw.find("{"):score_raw.rfind("}") + 1])
            except Exception:
                score = {}

            rag_probes.append({
                "type": orig_probe.get("type", ""),
                "question": question,
                "rag_answer": rag_answer,
                "scores": score,
                "orig_scores": orig_probe.get("scores", {}),
            })

        results.append({
            "case_key": case_key,
            "domain": domain,
            "probes": rag_probes,
        })
        done_keys.add(case_key)
        progress["done_keys"] = list(done_keys)
        progress["results"] = results
        with open(RAG_PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False)

        if (len(results)) % 10 == 0:
            print(f"    checkpoint: {len(results)} done")

    return results


def print_comparison(results, tax_probes, bsg_probes):
    """Compare no-RAG vs RAG uncertainty rates."""
    print(f"\n{'='*60}")
    print("RESULTS: No-RAG vs RAG")
    print(f"{'='*60}")

    for domain, orig_list, label in [
        ('tax', tax_probes, 'Tax'),
        ('sozialrecht', [r for r in results if r['domain'] == 'sozialrecht'], 'Social'),
    ]:
        # No-RAG stats (from original probes)
        if domain == 'tax':
            orig = tax_probes
        else:
            orig = bsg_probes

        no_rag_unc = [0, 0]
        for r in orig:
            for p in r.get('probes', []):
                s = p.get('scores', {})
                if not isinstance(s, dict):
                    continue
                no_rag_unc[1] += 1
                au = s.get('admitted_uncertainty')
                if au in (True, 'yes'):
                    no_rag_unc[0] += 1

        # RAG stats
        rag_results = [r for r in results if r['domain'] == domain]
        rag_unc = [0, 0]
        rag_scores = defaultdict(list)
        for r in rag_results:
            for p in r.get('probes', []):
                s = p.get('scores', {})
                if not isinstance(s, dict):
                    continue
                rag_unc[1] += 1
                au = s.get('admitted_uncertainty')
                if au in (True, 'yes'):
                    rag_unc[0] += 1
                for m in ['correct_test', 'correct_citations', 'correct_statute', 'specificity']:
                    v = s.get(m)
                    if isinstance(v, dict):
                        v = v.get('score')
                    if v:
                        rag_scores[m].append(v)

        print(f"\n--- {label} ---")
        if no_rag_unc[1]:
            print(f"  No-RAG uncertainty: {no_rag_unc[0]}/{no_rag_unc[1]} = {no_rag_unc[0]/no_rag_unc[1]:.1%}")
        if rag_unc[1]:
            print(f"  RAG uncertainty:    {rag_unc[0]}/{rag_unc[1]} = {rag_unc[0]/rag_unc[1]:.1%}")
        for m in ['correct_test', 'correct_citations', 'correct_statute', 'specificity']:
            vals = rag_scores[m]
            if vals:
                print(f"  RAG {m}: {sum(vals)/len(vals):.2f} (n={len(vals)})")


def main():
    # Step 1: Fetch RAG corpus
    corpus = fetch_rag_corpus()

    # Step 2: Build BM25 index
    bm25, chunks, chunk_domains = build_index(corpus)

    # Load existing probes
    tax_probes, bsg_probes = load_existing_probes()

    # Step 3-5: Run RAG experiment
    results = run_rag_experiment(bm25, chunks, tax_probes, bsg_probes)

    # Step 6: Compare
    print_comparison(results, tax_probes, bsg_probes)


if __name__ == "__main__":
    main()
