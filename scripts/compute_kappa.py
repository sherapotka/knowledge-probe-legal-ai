"""Parse human-annotated packets and compute Cohen's kappa: human-human, human-vs-judge, per rubric/language.

The 50-probe sample is fixed per language (data/validation/human_validation_probes_50_<lang>.json).
Each annotator labels the same probes with two verdicts each (Rubric A = strict, Rubric B = liberal).
The Llama judge labels under matched rubrics live in *_judged.json.

For German, Alex annotated the English-translated packet (de-en); Shera annotated the German original (de).
For French, both annotated both versions (fr and fr-en).
For Canadian English, both annotated the same en packet.
"""
import json, os, re, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# Maps from packet filename -> (rater, language, rubric_source)
ANNOTATIONS = {
    # German: Alex on translated, Shera on original
    "alex/de-en":   "validation/packets/alex/human-validation-packet-de-en.md",
    "shera/de":     "validation/packets/shera/human-validation-packet-de.md",

    # French: both raters did both versions (we will pick the original FR for each rater)
    "alex/fr":      "validation/packets/alex/human-validation-packet-fr.md",
    "alex/fr-en":   "validation/packets/alex/human-validation-packet-fr-en.md",
    "shera/fr":     "validation/packets/shera/human-validation-packet-fr.md",
    "shera/fr-en":  "validation/packets/shera/human-validation-packet-fr-en.md",

    # Canadian English: both raters did the original
    "alex/en":      "validation/packets/alex/human-validation-packet-en.md",
    "shera/en":     "validation/packets/shera/human-validation-packet-en.md",
}


# Verdict regex: handles all the rubric labels in any language.
# Examples:
#   **Probe 7 (A, strict):** Y
#   **Probe 7 (A, streng):** Y
#   **Probe 7 (A, stricte) :** Y
#   **Probe 7 (B, liberal):** N
VERDICT_RE = re.compile(
    r"\*\*Probe\s+(\d+)\s*\(\s*([AB])(?:\s*,\s*[^)]*)?\)\s*:?\*\*\s*([YyNn])",
    re.UNICODE,
)


def parse_packet(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # results[probe_idx] = {"A": "y/n", "B": "y/n"}
    results = {}
    for m in VERDICT_RE.finditer(text):
        probe_idx = int(m.group(1)) - 1  # 1-indexed in md, 0-indexed in JSON
        rubric = m.group(2)  # "A" or "B"
        verdict = m.group(3).lower()
        verdict = "yes" if verdict == "y" else "no"
        if probe_idx not in results:
            results[probe_idx] = {}
        results[probe_idx][rubric] = verdict
    return results


def cohen_kappa(labels_a, labels_b):
    """Both lists same length; values "yes" or "no". Returns kappa, agreement, n."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return None, None, 0
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n

    p_a_yes = sum(1 for a in labels_a if a == "yes") / n
    p_b_yes = sum(1 for b in labels_b if b == "yes") / n
    pe = p_a_yes * p_b_yes + (1 - p_a_yes) * (1 - p_b_yes)
    if pe == 1.0:
        kappa = float("nan")
    else:
        kappa = (po - pe) / (1 - pe)
    return kappa, po, n


def load_judge(lang):
    with open(f"data/validation/human_validation_probes_50_{lang}_judged.json", encoding="utf-8") as f:
        d = json.load(f)
    return [{"strict": p.get("judge_strict"), "liberal": p.get("judge_liberal")} for p in d]


def confusion(a, b):
    c = defaultdict(int)
    for x, y in zip(a, b):
        c[(x, y)] += 1
    return dict(c)


def report_pair(name_a, labels_a, name_b, labels_b):
    kappa, agree, n = cohen_kappa(labels_a, labels_b)
    if kappa is None:
        print(f"  {name_a} vs {name_b}: no data")
        return
    conf = confusion(labels_a, labels_b)
    print(f"  {name_a} vs {name_b}:  k={kappa:6.3f}  agree={agree:.1%}  (n={n})  yy={conf.get(('yes','yes'),0)}  nn={conf.get(('no','no'),0)}  yn={conf.get(('yes','no'),0)}  ny={conf.get(('no','yes'),0)}")


# Parse all packets
parsed = {key: parse_packet(path) for key, path in ANNOTATIONS.items()}

# Sanity: every parsed file has 50 probes with both A and B
for key, results in parsed.items():
    missing = [i for i in range(50) if i not in results or "A" not in results[i] or "B" not in results[i]]
    if missing:
        print(f"WARN: {key} missing {len(missing)} verdicts (probes {missing[:5]}...)")


def labels_for(parsed_results, rubric):
    return [parsed_results[i][rubric] for i in range(50)]


def run_language(lang_code, lang_label, alex_key, shera_key, judge_lang):
    print(f"\n=== {lang_label} ===")
    judge = load_judge(judge_lang)
    judge_strict = [j["strict"] for j in judge]
    judge_liberal = [j["liberal"] for j in judge]

    alex = parsed[alex_key]
    shera = parsed[shera_key]
    a_strict = labels_for(alex, "A")
    a_liberal = labels_for(alex, "B")
    s_strict = labels_for(shera, "A")
    s_liberal = labels_for(shera, "B")

    print(" Strict (Rubric A):")
    report_pair("Alex",  a_strict, "Shera", s_strict)
    report_pair("Alex",  a_strict, "Judge", judge_strict)
    report_pair("Shera", s_strict, "Judge", judge_strict)
    print(" Liberal (Rubric B):")
    report_pair("Alex",  a_liberal, "Shera", s_liberal)
    report_pair("Alex",  a_liberal, "Judge", judge_liberal)
    report_pair("Shera", s_liberal, "Judge", judge_liberal)


# German: Alex used translated packet, Shera used original
run_language("de", "GERMAN", "alex/de-en", "shera/de", "de")

# French: both raters did both versions; primary comparison uses original FR for both
run_language("fr", "FRENCH (both on original FR)", "alex/fr", "shera/fr", "fr")

# Sanity: also report French translated-vs-original within each rater
print("\n=== FRENCH (within-rater translated vs original) ===")
for rater in ("alex", "shera"):
    orig = parsed[f"{rater}/fr"]
    tran = parsed[f"{rater}/fr-en"]
    for rubric_name, key in [("Strict", "A"), ("Liberal", "B")]:
        a = labels_for(orig, key)
        b = labels_for(tran, key)
        kappa, agree, n = cohen_kappa(a, b)
        print(f"  {rater}, {rubric_name}: orig vs translated  k={kappa:.3f}  agree={agree:.1%}")

# Canadian English
run_language("en", "CANADIAN ENGLISH", "alex/en", "shera/en", "en")
