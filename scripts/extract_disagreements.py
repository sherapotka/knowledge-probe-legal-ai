"""Pull out probes where humans agree but Llama judge disagrees, for review."""
import json, re, os, sys
sys.stdout.reconfigure(encoding='utf-8')

VERDICT_RE = re.compile(r"\*\*Probe\s+(\d+)\s*\(\s*([AB])(?:\s*,\s*[^)]*)?\)\s*:?\*\*\s*([YyNn])", re.UNICODE)


def parse_packet(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    for m in VERDICT_RE.finditer(text):
        i = int(m.group(1)) - 1
        r = m.group(2)
        v = "yes" if m.group(3).lower() == "y" else "no"
        out.setdefault(i, {})[r] = v
    return out


JOBS = [
    {
        "lang": "de",
        "label": "German",
        "probes": "data/validation/human_validation_probes_50_de.json",
        "judged": "data/validation/human_validation_probes_50_de_judged.json",
        "alex": "validation/packets/alex/human-validation-packet-de-en.md",
        "shera": "validation/packets/shera/human-validation-packet-de.md",
        "translated": "data/validation/human_validation_probes_50_de_en_translated.json",
    },
    {
        "lang": "en",
        "label": "Canadian English",
        "probes": "data/validation/human_validation_probes_50_en.json",
        "judged": "data/validation/human_validation_probes_50_en_judged.json",
        "alex": "validation/packets/alex/human-validation-packet-en.md",
        "shera": "validation/packets/shera/human-validation-packet-en.md",
        "translated": None,
    },
]


def render(job):
    with open(job["probes"], encoding="utf-8") as f:
        probes = json.load(f)
    with open(job["judged"], encoding="utf-8") as f:
        judged = json.load(f)
    alex = parse_packet(job["alex"])
    shera = parse_packet(job["shera"])
    translated = None
    if job["translated"] and os.path.exists(job["translated"]):
        with open(job["translated"], encoding="utf-8") as f:
            translated = json.load(f)

    blocks = [f"# Probes where humans agree but Llama judge disagrees ({job['label']})\n\n"]
    blocks.append("Filter: both Alex and Shera assigned the same verdict, but the Llama judge gave the opposite verdict.\n\n")

    for rubric_name, rubric_key, judge_field in [("Rubric A (strict)", "A", "judge_strict"),
                                                   ("Rubric B (liberal)", "B", "judge_liberal")]:
        blocks.append(f"## {rubric_name}\n\n")
        cnt = 0
        for i, p in enumerate(probes):
            jv = judged[i].get(judge_field)
            av = alex.get(i, {}).get(rubric_key)
            sv = shera.get(i, {}).get(rubric_key)
            if av != sv:
                continue
            if av == jv:
                continue
            cnt += 1
            blocks.append(f"### Probe {i+1} ({p['domain']}, {p['qtype']})\n\n")
            blocks.append(f"**Verdicts.** Alex: **{av.upper()}** | Shera: **{sv.upper()}** | Llama judge: **{jv.upper() if jv else 'ERROR'}**  \n")
            judge_reason = judged[i].get(f"{judge_field}_reason", "")
            if judge_reason:
                blocks.append(f"**Judge's reason:** {judge_reason}\n\n")
            blocks.append(f"**Question:** {p['question']}\n\n")
            blocks.append(f"**Answer (original):**\n\n{p['answer']}\n\n")
            if translated:
                t = translated[i]
                if t.get("answer_en"):
                    blocks.append(f"**Answer (English translation):**\n\n{t['answer_en']}\n\n")
            blocks.append("---\n\n")
        if cnt == 0:
            blocks.append("_(none)_\n\n")
        else:
            # Insert a count after the heading
            blocks.insert(-2 * cnt - 1, f"_{cnt} probe(s)._\n\n")
    return "".join(blocks)


for job in JOBS:
    out_path = f"validation/disagreements-judge-vs-humans-{job['lang']}.md"
    md = render(job)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {out_path}")
