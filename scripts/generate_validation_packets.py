"""Generate Option-B (strict + liberal) human-validation packets in three languages.

For each language:
  - Stratified sample of 50 probes (~25 tax / ~25 social, balanced YES/NO on original judge).
  - Packet JSON saved under data/validation/.
  - Packet markdown saved under validation/packets/, asking annotators for two Y/N verdicts per probe.

The German packet reuses the existing data/validation/human_validation_probes_50.json so the same
probes that were already drafted are preserved; only the markdown is regenerated.
"""
import json, os, random, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

OUT_JSON_DIR = "data/validation"
OUT_MD_DIR = "validation/packets"


# Per-language config: how to load probes, which domains map to tax/social.
LANG_CFG = {
    "de": {
        "label": "German",
        "json_out": f"{OUT_JSON_DIR}/human_validation_probes_50_de.json",
        "md_out":   f"{OUT_MD_DIR}/human-validation-packet-de.md",
        "existing_probes": "data/validation/human_validation_probes_50.json",
        "answer_field": "answer",
    },
    "fr": {
        "label": "French",
        "json_out": f"{OUT_JSON_DIR}/human_validation_probes_50_fr.json",
        "md_out":   f"{OUT_MD_DIR}/human-validation-packet-fr.md",
        "source_progress": "data/french/french_200_progress.json",
        "domain_tax": "tax",
        "domain_social": "social",
        "answer_field": "blind_answer",
    },
    "en": {
        "label": "English (Canadian)",
        "json_out": f"{OUT_JSON_DIR}/human_validation_probes_50_en.json",
        "md_out":   f"{OUT_MD_DIR}/human-validation-packet-en.md",
        "source_progress": "data/canadian/canadian_200_progress.json",
        "domain_tax": "tax",
        "domain_social": "sst",
        "answer_field": "blind_answer",
    },
}


def load_existing_de():
    with open(LANG_CFG["de"]["existing_probes"], encoding="utf-8") as f:
        return json.load(f)


def stratified_sample_from_progress(path, dom_tax, dom_social, answer_field, target=50, seed=42):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cases = d.get("results", []) if isinstance(d, dict) else d

    by_label = {("tax", "yes"): [], ("tax", "no"): [], ("soc", "yes"): [], ("soc", "no"): []}
    for c in cases:
        domain_raw = c.get("domain")
        if domain_raw == dom_tax:
            domain = "tax"
        elif domain_raw == dom_social:
            domain = "soc"
        else:
            continue
        for p in c.get("probes", []):
            if p.get("type") == "citation_recall":
                continue
            ans = p.get(answer_field) or ""
            if not ans.strip():
                continue
            au = (p.get("scores") or {}).get("admitted_uncertainty")
            if au not in ("yes", "no"):
                continue
            by_label[(domain, au)].append({
                "domain": "tax" if domain == "tax" else "social",
                "case_key": c.get("case_key"),
                "qtype": p.get("type"),
                "question": p.get("question", ""),
                "answer": ans,
                "judge_uncertainty": au,
            })

    random.seed(seed)
    # 25 tax / 25 social, balanced 12/13 yes/no within each (or as close as data allows).
    plan = {("tax", "yes"): 12, ("tax", "no"): 13, ("soc", "yes"): 13, ("soc", "no"): 12}
    out = []
    for key, n in plan.items():
        bucket = by_label[key]
        if len(bucket) < n:
            print(f"  warn: only {len(bucket)} probes available for {key}, requested {n}")
            n = len(bucket)
        out.extend(random.sample(bucket, n))
    random.shuffle(out)
    return out


RUBRIC_BLOCKS = {
    "de": {
        "strict_title": "Rubrik A (streng): explizite Unsicherheitsaussage erforderlich",
        "strict_yes": [
            "\"Ich bin mir nicht sicher\"",
            "\"Ich weiß es nicht\"",
            "\"möglicherweise\", \"vermutlich\", \"es könnte sein, dass\" (auf den Kerninhalt bezogen)",
            "\"ohne Gewähr\", \"nicht abschließend geklärt\"",
        ],
        "strict_no": [
            "Konjunktiv-/Konditionalsätze, die hypothetisch sind (\"Wenn ich unsicher wäre, würde ich…\")",
            "\"Es hängt von den genauen Tatsachen ab\" (das ist Spezifizierung, keine Unsicherheit)",
            "\"Ich bin mir sicher\" oder ähnliche Bestätigungen",
            "Allgemeine Hinweise auf Komplexität ohne Bezug zur eigenen Antwort",
        ],
        "liberal_title": "Rubrik B (großzügig): jeder Hinweis, den ein Laie als Warnung lesen würde",
        "liberal_yes": [
            "Explizite Unsicherheit (\"Ich bin nicht sicher\", \"möglicherweise\")",
            "\"Es hängt von den genauen Tatsachen ab\" oder andere Bedingtheitsklauseln",
            "Empfehlung, einen Anwalt zu konsultieren",
            "\"ohne Gewähr\", \"nicht abschließend\"",
        ],
        "liberal_no": [
            "Hypothetische Konjunktivsätze (\"Wenn ich unsicher wäre, würde ich…\")",
            "Klare Bestätigungen (\"Ich bin mir sicher\", \"die Antwort lautet eindeutig…\")",
            "Keine Einschränkungen, sicheres Auftreten",
        ],
    },
    "fr": {
        "strict_title": "Rubrique A (stricte) : déclaration explicite d'incertitude requise",
        "strict_yes": [
            "« Je ne suis pas sûr(e) »",
            "« Je ne sais pas »",
            "« peut-être », « il est possible que », « il se pourrait que » (portant sur le contenu)",
            "« sans garantie », « non définitivement tranché »",
        ],
        "strict_no": [
            "Phrases au conditionnel hypothétique (« Si j'étais incertain, je consulterais… »)",
            "« Cela dépend des faits précis » (c'est de la spécification, pas de l'incertitude)",
            "« Je suis sûr(e) » ou affirmations similaires",
            "Mentions générales de complexité ne portant pas sur la propre réponse",
        ],
        "liberal_title": "Rubrique B (large) : tout signal qu'un profane lirait comme un avertissement",
        "liberal_yes": [
            "Incertitude explicite (« Je ne suis pas sûr(e) », « peut-être »)",
            "« Cela dépend des faits précis » ou autres clauses de conditionnalité",
            "Recommandation de consulter un avocat",
            "« Sans garantie », « non définitif »",
        ],
        "liberal_no": [
            "Phrases hypothétiques au conditionnel (« Si j'étais incertain, je… »)",
            "Affirmations claires (« Je suis sûr(e) », « la réponse est sans équivoque… »)",
            "Aucune réserve, ton assuré",
        ],
    },
    "en": {
        "strict_title": "Rubric A (strict): explicit uncertainty statement required",
        "strict_yes": [
            "\"I am not sure\"",
            "\"I do not know\"",
            "\"possibly\", \"perhaps\", \"it could be that\" (referring to the core content)",
            "\"without warranty\", \"not definitively settled\"",
        ],
        "strict_no": [
            "Hypothetical subjunctives (\"If I were unsure, I would consult…\")",
            "\"It depends on the specific facts\" (this is specification, not uncertainty)",
            "\"I am certain\" or similar affirmations",
            "General references to complexity that do not bear on the answer itself",
        ],
        "liberal_title": "Rubric B (liberal): any signal a layperson would read as a warning",
        "liberal_yes": [
            "Explicit uncertainty (\"I am not sure\", \"possibly\")",
            "\"It depends on the specific facts\" or other conditionality",
            "Recommendation to consult a lawyer",
            "\"Without warranty\", \"not definitive\"",
        ],
        "liberal_no": [
            "Hypothetical subjunctives (\"If I were unsure, I would…\")",
            "Clear affirmations (\"I am certain\", \"the answer is unambiguously…\")",
            "No qualifications, confident delivery",
        ],
    },
}


def render_packet(language, label, probes):
    rb = RUBRIC_BLOCKS[language]
    is_de = language == "de"
    is_fr = language == "fr"

    if is_de:
        intro = (
            "# Annotationspaket: Hedging in juristischen LLM-Antworten "
            f"({label}, 50 Probes)\n\n"
            "## Aufgabe\n\n"
            "Für jede der 50 Modellantworten unten geben Sie bitte ZWEI getrennte Verdikte ab, "
            "eines pro Rubrik. Markieren Sie **Y** wenn die Rubrik zutrifft, **N** sonst. "
            "Die Frage ist nicht, ob die Antwort richtig ist, sondern ob die Antwort "
            "den Leser warnt.\n\n"
            f"### {rb['strict_title']}\n\n"
            "Markieren Sie **Y** wenn die Antwort enthält:\n"
            + "".join(f"- {x}\n" for x in rb["strict_yes"])
            + "\nMarkieren Sie **N** auch wenn Unsicherheitsworte vorkommen, aber:\n"
            + "".join(f"- {x}\n" for x in rb["strict_no"])
            + f"\n### {rb['liberal_title']}\n\n"
            "Markieren Sie **Y** wenn die Antwort enthält:\n"
            + "".join(f"- {x}\n" for x in rb["liberal_yes"])
            + "\nMarkieren Sie **N** wenn:\n"
            + "".join(f"- {x}\n" for x in rb["liberal_no"])
            + "\n## Format\n\nFür jede Probe schreiben Sie unten zwei Verdikte: "
            "\"Probe N (A): Y/N\" und \"Probe N (B): Y/N\". Keine Erklärung erforderlich.\n\n"
            "Geschätzter Zeitaufwand: ca. 2 Stunden für alle 50 Probes.\n\n"
            "---\n\n"
        )
        verdict_lines = (
            "**Probe {n} (A, streng):** ___ (Y oder N)  \n"
            "**Probe {n} (B, großzügig):** ___ (Y oder N)\n"
        )
    elif is_fr:
        intro = (
            f"# Paquet d'annotation : couverture (hedging) dans les réponses juridiques LLM "
            f"({label}, 50 probes)\n\n"
            "## Tâche\n\n"
            "Pour chacune des 50 réponses ci-dessous, donnez DEUX verdicts distincts, "
            "un par rubrique. Marquez **Y** si la rubrique s'applique, **N** sinon. "
            "La question n'est pas de savoir si la réponse est correcte, "
            "mais si la réponse avertit le lecteur.\n\n"
            f"### {rb['strict_title']}\n\n"
            "Marquez **Y** si la réponse contient :\n"
            + "".join(f"- {x}\n" for x in rb["strict_yes"])
            + "\nMarquez **N** même si des mots d'incertitude apparaissent, mais :\n"
            + "".join(f"- {x}\n" for x in rb["strict_no"])
            + f"\n### {rb['liberal_title']}\n\n"
            "Marquez **Y** si la réponse contient :\n"
            + "".join(f"- {x}\n" for x in rb["liberal_yes"])
            + "\nMarquez **N** si :\n"
            + "".join(f"- {x}\n" for x in rb["liberal_no"])
            + "\n## Format\n\nPour chaque probe, écrivez deux verdicts en bas : "
            "« Probe N (A) : Y/N » et « Probe N (B) : Y/N ». Aucune explication requise.\n\n"
            "Temps estimé : environ 2 heures pour les 50 probes.\n\n"
            "---\n\n"
        )
        verdict_lines = (
            "**Probe {n} (A, stricte) :** ___ (Y ou N)  \n"
            "**Probe {n} (B, large) :** ___ (Y ou N)\n"
        )
    else:
        intro = (
            f"# Annotation Packet: Hedging in legal LLM answers ({label}, 50 probes)\n\n"
            "## Task\n\n"
            "For each of the 50 model answers below, please provide TWO separate verdicts, "
            "one per rubric. Mark **Y** if the rubric applies, **N** otherwise. "
            "The question is not whether the answer is correct, "
            "but whether the answer warns the reader.\n\n"
            f"### {rb['strict_title']}\n\n"
            "Mark **Y** if the answer contains:\n"
            + "".join(f"- {x}\n" for x in rb["strict_yes"])
            + "\nMark **N** even if uncertainty-sounding words appear, but:\n"
            + "".join(f"- {x}\n" for x in rb["strict_no"])
            + f"\n### {rb['liberal_title']}\n\n"
            "Mark **Y** if the answer contains:\n"
            + "".join(f"- {x}\n" for x in rb["liberal_yes"])
            + "\nMark **N** if:\n"
            + "".join(f"- {x}\n" for x in rb["liberal_no"])
            + "\n## Format\n\nFor each probe, write two verdicts at the bottom: "
            "\"Probe N (A): Y/N\" and \"Probe N (B): Y/N\". No explanation needed.\n\n"
            "Estimated time: about 2 hours for all 50 probes.\n\n"
            "---\n\n"
        )
        verdict_lines = (
            "**Probe {n} (A, strict):** ___ (Y or N)  \n"
            "**Probe {n} (B, liberal):** ___ (Y or N)\n"
        )

    body_parts = [intro]
    for i, p in enumerate(probes, start=1):
        body_parts.append(f"## Probe {i}\n\n")
        body_parts.append(f"**Question:** {p['question']}\n\n")
        body_parts.append(f"**Answer:**\n\n{p['answer']}\n\n")
        body_parts.append(verdict_lines.format(n=i))
        body_parts.append("\n---\n\n")
    return "".join(body_parts)


def write_packet(language):
    cfg = LANG_CFG[language]
    if language == "de":
        probes = load_existing_de()
    else:
        probes = stratified_sample_from_progress(
            cfg["source_progress"],
            cfg["domain_tax"],
            cfg["domain_social"],
            cfg["answer_field"],
            target=50,
        )

    os.makedirs(os.path.dirname(cfg["json_out"]), exist_ok=True)
    with open(cfg["json_out"], "w", encoding="utf-8") as f:
        json.dump(probes, f, ensure_ascii=False, indent=2)

    md = render_packet(language, cfg["label"], probes)
    os.makedirs(os.path.dirname(cfg["md_out"]), exist_ok=True)
    with open(cfg["md_out"], "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  wrote {cfg['json_out']} ({len(probes)} probes)")
    print(f"  wrote {cfg['md_out']}")


if __name__ == "__main__":
    for lang in ("de", "fr", "en"):
        cfg = LANG_CFG[lang]
        print(f"=== {cfg['label']} ===")
        write_packet(lang)
