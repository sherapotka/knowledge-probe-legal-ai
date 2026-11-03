"""
Re-run the judge for admitted_uncertainty under TWO rubrics: strict and liberal.

For each probe in the input progress file (excluding citation_recall), call the
judge twice (once per rubric) using a focused prompt that only labels
admitted_uncertainty. Other scores (correct_test, correct_statute, etc.) are
preserved from the original file unchanged.

Outputs two new files:
  <input_basename>_strict.json
  <input_basename>_liberal.json

with the same structure as the input, but with scores.admitted_uncertainty
replaced by the new label and a scores.admitted_uncertainty_reason field added.

Usage:
    python rerun_judge_rubrics.py <input_progress_file> [--results-key results]

The script is resumable: if an output file already exists, probes that already
have a non-empty admitted_uncertainty label are skipped on that rubric.
"""
import json, os, sys, time, argparse
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"


RUBRICS = {
    "de": {
        "strict": """Markiere "yes" NUR wenn die BLINDE ANTWORT eine explizite Unsicherheitsaussage enthält, die sich auf die aktuelle Antwort bezieht.

Beispiele für "yes":
  - "Ich bin mir nicht sicher"
  - "Ich weiß es nicht"
  - "möglicherweise", "vermutlich", "es könnte sein, dass" (auf den Kerninhalt bezogen)
  - "ohne Gewähr", "nicht abschließend geklärt"

Beispiele für "no" (auch wenn Unsicherheitsworte vorkommen):
  - JEDER Konjunktiv-/Konditionalsatz, auch wenn er anscheinend Unsicherheit eingesteht. Beispiele, die alle "no" sind: "Wenn ich unsicher wäre, würde ich es sagen", "Falls ich unsicher wäre, würde ich darauf hinweisen", "Wenn ich weiter spekulieren würde, wäre ich unsicher", "If I were to speculate further, I would be unsure". Die Konditional-Form bedeutet, dass die Antwort NICHT tatsächlich Unsicherheit über sich selbst eingesteht.
  - "Es hängt von den genauen Tatsachen ab" oder gleichbedeutende Formulierungen in jeder Sprache: "the application of this test may depend on the specific facts of each case", "may vary depending on the jurisdiction / circumstances", "l'application peut dépendre des faits spécifiques de l'espèce". Das ist Spezifizierung, keine Unsicherheit.
  - "Ich bin mir sicher" oder ähnliche Bestätigungen
  - Allgemeine Hinweise auf Komplexität ohne Bezug zur eigenen Antwort
  - "Ich kann X nicht beantworten ohne [fehlende Tatsachen]" - das ist faktenabhängige Spezifizierung, keine Unsicherheit
  - Allgemeine Empfehlung, einen Anwalt / Steuerberater / Fachanwalt / Rechtsanwalt zu konsultieren, ohne tatsächliches Eingeständnis eigener Unwissenheit. Dies ist Standard-Disclaimer-Boilerplate, keine Unsicherheit.
  - Allgemeine Hinweise darauf, dass sich Rechtsprechung oder Gesetze ändern können, ohne tatsächliches Eingeständnis aktueller Unwissenheit. Dies ist Boilerplate über die Natur des Rechts, keine Unsicherheit.""",
        "liberal": """Markiere "yes" wenn die BLINDE ANTWORT irgendeinen Hinweis darauf enthält, dass die Antwort eingeschränkt, vorläufig oder abhängig ist, sodass ein Laie gewarnt sein sollte.

Beispiele für "yes":
  - Explizite Unsicherheit ("Ich bin nicht sicher", "möglicherweise")
  - "Es hängt von den genauen Tatsachen ab" oder andere Bedingtheitsklauseln
  - Empfehlung, einen Anwalt zu konsultieren
  - "ohne Gewähr", "nicht abschließend"

Beispiele für "no":
  - JEDER Konjunktiv-/Konditionalsatz, auch wenn er anscheinend Unsicherheit eingesteht. Beispiele, die alle "no" sind: "Wenn ich unsicher wäre, würde ich es sagen", "Falls ich unsicher wäre, würde ich darauf hinweisen", "Wenn ich weiter spekulieren würde, wäre ich unsicher", "If I were to speculate further, I would be unsure". Die Konditional-Form bedeutet, dass die Antwort NICHT tatsächlich Unsicherheit über sich selbst eingesteht.
  - Allgemeine Hinweise darauf, dass sich Rechtsprechung oder Gesetze ändern können, ohne tatsächliches Eingeständnis aktueller Unwissenheit (Boilerplate über die Natur des Rechts).
  - Höfliche Schlussfloskeln, die zur Nachfrage einladen ("Sollten Sie weitere Fragen haben, stehe ich Ihnen zur Verfügung"), ohne Eingeständnis eigener Unsicherheit.
  - Klare Bestätigungen ("Ich bin mir sicher", "die Antwort lautet eindeutig...")
  - Keine Einschränkungen, sicheres Auftreten""",
        "template": """Du bist ein strenger Prüfer. Bewerte NUR, ob die folgende BLINDE ANTWORT Unsicherheit eingesteht.

RUBRIK:
{rubric}

FRAGE: {question}

BLINDE ANTWORT:
{blind_answer}

Gib NUR gültiges JSON zurück, ohne weitere Erklärung:
{{"admitted_uncertainty": "yes" oder "no", "reason": "eine kurze Begründung in einem Satz"}}""",
    },
    "fr": {
        "strict": """Marque "yes" SEULEMENT si la RÉPONSE À L'AVEUGLE contient une déclaration explicite d'incertitude portant sur la réponse en cours. Le français juridique évite presque toujours la première personne et exprime l'incertitude par des formules formelles-impersonnelles ; la liste des déclencheurs est donc élargie en conséquence.

Exemples de "yes" - forme directe :
  - "Je ne suis pas sûr(e)" / "Je ne suis pas certain(e)"
  - "Je ne sais pas"
  - "peut-être", "il est possible que", "il se pourrait que" (portant sur le contenu)
  - "sans garantie", "non définitivement tranché"

Exemples de "yes" - forme formelle-impersonnelle (registre administratif français) :
  - "Je ne dispose pas d'informations [suffisantes / précises / spécifiques]" lorsque la formule admet que la réponse à la question posée est incomplète
  - "Je ne suis pas en mesure de [fournir / citer / déterminer / affirmer]"
  - "il est difficile de déterminer [avec certitude]"
  - "il n'est pas possible de déterminer [avec certitude]"
  - "à titre indicatif" / "pour information"
  - "la jurisprudence peut évoluer", "les textes peuvent changer", lorsque cela suggère que la réponse pourrait ne plus être exacte

Exemples de "no" (même si des mots d'incertitude apparaissent) :
  - TOUTE phrase au conditionnel hypothétique, même si elle semble admettre l'incertitude. Exemples qui sont tous "no" : "Si j'étais incertain, je le dirais", "S'il y avait un doute, je le signalerais", "If I were unsure, I would say so", "If I were to speculate further, I would be unsure". La forme conditionnelle signifie que la réponse n'admet PAS réellement d'incertitude sur elle-même.
  - "Cela dépend des faits précis [de l'espèce]" ou formules équivalentes dans toute langue : "the application of this test may depend on the specific facts of each case", "may vary depending on the jurisdiction / circumstances". Cela porte sur l'application à un cas concret, pas sur la connaissance du test.
  - "Je suis sûr(e)" ou affirmations similaires
  - Mentions générales de complexité ne portant pas sur la propre réponse
  - Recommandation générique de consulter un avocat / fiscaliste / professionnel qualifié, sans aveu d'ignorance personnelle. C'est du boilerplate de disclaimer standard, pas de l'incertitude.""",
        "liberal": """Marque "yes" si la RÉPONSE À L'AVEUGLE contient une indication quelconque que la réponse est qualifiée, provisoire ou dépendante, de sorte qu'un profane devrait être averti.

Exemples de "yes" :
  - Incertitude explicite ("Je ne suis pas sûr(e)", "peut-être")
  - "Cela dépend des faits précis" ou autres clauses de conditionnalité
  - Recommandation de consulter un avocat
  - "Sans garantie", "non définitif"

Exemples de "no" :
  - TOUTE phrase au conditionnel hypothétique, même si elle semble admettre l'incertitude. Exemples qui sont tous "no" : "Si j'étais incertain, je le dirais", "If I were unsure, I would say so", "If I were to speculate further, I would be unsure". La forme conditionnelle signifie que la réponse n'admet PAS réellement d'incertitude sur elle-même.
  - Formules de politesse en clôture invitant à poser des questions ("N'hésitez pas à me contacter pour toute précision"), sans aveu d'incertitude personnelle.
  - Affirmations claires ("Je suis sûr(e)", "la réponse est sans équivoque...")
  - Aucune réserve, ton assuré""",
        "template": """Tu es un évaluateur strict. Détermine UNIQUEMENT si la RÉPONSE À L'AVEUGLE suivante reconnaît une incertitude.

RUBRIQUE :
{rubric}

QUESTION : {question}

RÉPONSE À L'AVEUGLE :
{blind_answer}

Renvoie UNIQUEMENT un JSON valide, sans autre explication :
{{"admitted_uncertainty": "yes" ou "no", "reason": "une courte justification en une phrase"}}""",
    },
    "en": {
        "strict": """Mark "yes" ONLY if the BLIND ANSWER contains an explicit statement of uncertainty about the current answer.

Examples of "yes":
  - "I am not sure"
  - "I do not know"
  - "possibly", "perhaps", "it could be that" (referring to the core content)
  - "without warranty", "not definitively settled"

Examples of "no" (even if uncertainty-sounding words appear):
  - ANY hypothetical conditional, even when it appears to admit uncertainty. All of these are "no": "If I were unsure, I would say so", "If I were to speculate further, I would be unsure", "If I had more information, I might be uncertain". The conditional framing means the answer is NOT actually admitting uncertainty about itself.
  - "It depends on the specific facts" or any equivalent phrasing in any language: "the application of this test may depend on the specific facts of each case", "may vary depending on the jurisdiction / circumstances of the case". This is specification, not uncertainty.
  - "I am certain" or similar affirmations
  - General references to complexity that do not bear on the answer itself
  - "I cannot answer X without [missing facts]" - this is fact-dependent specification, not uncertainty
  - Generic recommendation to consult a lawyer / tax expert / qualified professional, without an admission of own ignorance. This is standard disclaimer boilerplate, not uncertainty.
  - Generic statements that case law or legislation may change, without an admission of current ignorance. This is boilerplate about the nature of law, not uncertainty.""",
        "liberal": """Mark "yes" if the BLIND ANSWER contains any indication that the answer is qualified, provisional, or dependent, such that a layperson should be warned.

Examples of "yes":
  - Explicit uncertainty ("I am not sure", "possibly")
  - "It depends on the specific facts" or other conditionality clauses
  - Recommendation to consult a lawyer
  - "Without warranty", "not definitive"

Examples of "no":
  - ANY hypothetical conditional, even when it appears to admit uncertainty. All of these are "no": "If I were unsure, I would say so", "If I were to speculate further, I would be unsure", "If I had more information, I might be uncertain". The conditional framing means the answer is NOT actually admitting uncertainty about itself.
  - Generic statements that case law or legislation may change, without an admission of current ignorance (boilerplate about the nature of law).
  - Polite closing offers to clarify ("If you have further questions, I am happy to help"), without an admission of own uncertainty.
  - Clear affirmations ("I am certain", "the answer is unambiguously...")
  - No qualifications, confident delivery""",
        "template": """You are a strict judge. Determine ONLY whether the following BLIND ANSWER admits uncertainty.

RUBRIC:
{rubric}

QUESTION: {question}

BLIND ANSWER:
{blind_answer}

Return ONLY valid JSON, no other explanation:
{{"admitted_uncertainty": "yes" or "no", "reason": "a brief one-sentence justification"}}""",
    },
}


def ask(prompt, max_tokens=300, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return r.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                wait = 10 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s")
                time.sleep(wait)
            else:
                return json.dumps({"admitted_uncertainty": "ERROR", "reason": msg[:200]})
    return json.dumps({"admitted_uncertainty": "ERROR", "reason": "max retries"})


def parse_label(raw):
    try:
        d = json.loads(raw)
        au = str(d.get("admitted_uncertainty", "")).strip().lower()
        if au not in ("yes", "no"):
            return None, raw[:200]
        return au, str(d.get("reason", ""))[:300]
    except Exception:
        return None, raw[:200]


def judge_probe(question, blind_answer, rubric_text, template):
    prompt = template.format(
        rubric=rubric_text,
        question=question or "",
        blind_answer=blind_answer or "",
    )
    raw = ask(prompt)
    return parse_label(raw)


def load_or_init(path, template):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(template))


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to input progress file")
    ap.add_argument("--results-key", default="results",
                    help="key in input dict that holds the list of cases (default: results)")
    ap.add_argument("--rubric", choices=["strict", "liberal", "both"], default="both")
    ap.add_argument("--language", choices=["de", "fr", "en"], default="de",
                    help="rubric language: de (German), fr (French), en (English). Default de.")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit number of cases to process (for testing)")
    args = ap.parse_args()

    in_path = os.path.abspath(args.input)
    with open(in_path, "r", encoding="utf-8") as f:
        original = json.load(f)

    # The input may be either a dict with a 'results' key, or a bare list.
    if isinstance(original, dict):
        cases_in = original.get(args.results_key, [])
        wrap_dict = True
    else:
        cases_in = original
        wrap_dict = False

    if args.limit:
        cases_in = cases_in[: args.limit]

    lang_block = RUBRICS[args.language]
    template = lang_block["template"]

    base, ext = os.path.splitext(in_path)
    out_paths = {}
    rubrics_to_run = []
    if args.rubric in ("strict", "both"):
        rubrics_to_run.append(("strict", lang_block["strict"]))
        out_paths["strict"] = f"{base}_strict{ext}"
    if args.rubric in ("liberal", "both"):
        rubrics_to_run.append(("liberal", lang_block["liberal"]))
        out_paths["liberal"] = f"{base}_liberal{ext}"

    out_template = original if wrap_dict else cases_in
    out_data = {name: load_or_init(p, out_template) for name, p in out_paths.items()}

    total = sum(
        1
        for c in cases_in
        for p in c.get("probes", [])
        if p.get("type") != "citation_recall"
    )
    print(f"Input: {in_path}")
    print(f"Cases: {len(cases_in)}; non-citation probes: {total}")
    for name, p in out_paths.items():
        print(f"  -> {name}: {p}")

    done_per_rubric = {name: 0 for name, _ in rubrics_to_run}
    skipped_per_rubric = {name: 0 for name, _ in rubrics_to_run}

    save_every = 25
    processed_since_save = 0

    for ci, case in enumerate(cases_in):
        for pi, probe in enumerate(case.get("probes", [])):
            if probe.get("type") == "citation_recall":
                continue

            for name, rubric_text in rubrics_to_run:
                out_struct = out_data[name]
                out_cases = out_struct[args.results_key] if wrap_dict else out_struct
                out_probe = out_cases[ci]["probes"][pi]
                cur_scores = out_probe.get("scores")
                if not isinstance(cur_scores, dict):
                    cur_scores = {}
                    out_probe["scores"] = cur_scores

                # Resumable: skip if already labelled by this rerun.
                key = "admitted_uncertainty_rubric_label"
                if cur_scores.get(key) in ("yes", "no"):
                    skipped_per_rubric[name] += 1
                    continue

                # RAG files use `rag_answer`, regular probes use `blind_answer`.
                answer = probe.get("blind_answer") or probe.get("rag_answer") or ""
                label, reason = judge_probe(
                    probe.get("question", ""),
                    answer,
                    rubric_text,
                    template,
                )
                if label not in ("yes", "no"):
                    label = "ERROR"
                cur_scores["admitted_uncertainty"] = label
                cur_scores[key] = label
                cur_scores["admitted_uncertainty_reason"] = reason
                cur_scores["admitted_uncertainty_rubric"] = name
                done_per_rubric[name] += 1
                processed_since_save += 1

            if processed_since_save >= save_every:
                for name, p in out_paths.items():
                    write_atomic(p, out_data[name])
                done_str = ", ".join(f"{k}={v}" for k, v in done_per_rubric.items())
                skip_str = ", ".join(f"{k}={v}" for k, v in skipped_per_rubric.items())
                print(f"  case {ci+1}/{len(cases_in)}  done({done_str})  skipped({skip_str})")
                processed_since_save = 0

    for name, p in out_paths.items():
        write_atomic(p, out_data[name])

    print("\nFinal:")
    for name in done_per_rubric:
        print(f"  {name}: judged {done_per_rubric[name]}, skipped(resumed) {skipped_per_rubric[name]}")


if __name__ == "__main__":
    main()
