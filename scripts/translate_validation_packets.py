"""Translate the German and French human-validation packets into English (faithful, preserves hedging).

Reads:
  data/validation/human_validation_probes_50_de.json
  data/validation/human_validation_probes_50_fr.json

Writes:
  data/validation/human_validation_probes_50_de_en_translated.json
  data/validation/human_validation_probes_50_fr_en_translated.json
  validation/packets/human-validation-packet-de-en.md
  validation/packets/human-validation-packet-fr-en.md

Translation model: Llama 3.3 70B via Groq (same model used for the judge, so the
validation chain is not inflated by a smarter system). Translations are
literal: hedging and certainty language is preserved verbatim. Resumable:
already-translated probes are skipped on rerun.
"""
import json, os, sys, time
from dotenv import load_dotenv
from groq import Groq

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"


PROMPT_TEMPLATE = """Translate the following {source_lang} legal text into English.

Strict requirements:
1. Translate literally and completely. Do not summarize, paraphrase, or shorten.
2. Preserve every expression of uncertainty, certainty, hedging, qualification, or conditionality EXACTLY (e.g. "Ich bin mir nicht sicher" -> "I am not sure"; "ohne Gewähr" -> "without warranty"; "es hängt von den genauen Tatsachen ab" -> "it depends on the specific facts"; "Je ne suis pas sûr(e)" -> "I am not sure"). Do not add, soften, or strengthen any uncertainty or certainty cue.
3. Keep statute citations (e.g. "§ 9 Abs. 1 EStG", "L. 6313-1") and case references in their original form.
4. Preserve formatting (paragraph breaks, bullet points, bold).
5. Output ONLY the English translation, no commentary.

{source_lang} text:
\"\"\"
{text}
\"\"\""""


def ask(prompt, max_tokens=4000, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                wait = 10 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s")
                time.sleep(wait)
            else:
                return f"[TRANSLATION ERROR: {msg[:200]}]"
    return "[TRANSLATION ERROR: max retries]"


def translate(text, source_lang):
    if not text or not text.strip():
        return text
    return ask(PROMPT_TEMPLATE.format(source_lang=source_lang, text=text)).strip()


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def translate_packet(in_path, out_path, source_lang_label):
    with open(in_path, encoding="utf-8") as f:
        probes = json.load(f)
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = [{"_idx": i, **p} for i, p in enumerate(probes)]
    if not isinstance(existing, list) or len(existing) != len(probes):
        existing = [{"_idx": i, **p} for i, p in enumerate(probes)]

    out = existing
    n_done = 0
    for i, src in enumerate(probes):
        rec = out[i]
        if rec.get("question_en") and rec.get("answer_en"):
            continue
        rec["domain"] = src["domain"]
        rec["case_key"] = src.get("case_key")
        rec["qtype"] = src.get("qtype")
        rec["judge_uncertainty"] = src.get("judge_uncertainty")
        rec["question_orig"] = src.get("question", "")
        rec["answer_orig"] = src.get("answer", "")
        if not rec.get("question_en"):
            rec["question_en"] = translate(src.get("question", ""), source_lang_label)
        if not rec.get("answer_en"):
            rec["answer_en"] = translate(src.get("answer", ""), source_lang_label)
        n_done += 1
        if n_done % 5 == 0:
            write_atomic(out_path, out)
            print(f"  translated {i+1}/{len(probes)}")
    write_atomic(out_path, out)
    print(f"  done: {len(probes)} probes -> {out_path}")
    return out


RUBRIC_EN = {
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
}


def render_md(translated, source_label):
    rb = RUBRIC_EN
    intro = (
        f"# Annotation Packet: Hedging in legal LLM answers (translated {source_label}, 50 probes)\n\n"
        "## Task\n\n"
        "For each of the 50 model answers below, please provide TWO separate verdicts, one per rubric. "
        "Mark **Y** if the rubric applies, **N** otherwise. The question is not whether the answer is correct, "
        "but whether the answer warns the reader.\n\n"
        f"**Note on translation.** These probes were originally in {source_label}. They have been translated "
        "into English by Llama 3.3 70B with an explicit instruction to preserve all hedging and certainty "
        "language verbatim. The verdicts you give here will be compared with verdicts from a reader of the "
        "original-language packet to compute inter-rater agreement.\n\n"
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
    parts = [intro]
    for i, p in enumerate(translated, start=1):
        parts.append(f"## Probe {i}\n\n")
        parts.append(f"**Question:** {p.get('question_en','').strip()}\n\n")
        parts.append(f"**Answer:**\n\n{p.get('answer_en','').strip()}\n\n")
        parts.append(
            f"**Probe {i} (A, strict):** ___ (Y or N)  \n"
            f"**Probe {i} (B, liberal):** ___ (Y or N)\n"
        )
        parts.append("\n---\n\n")
    return "".join(parts)


def main():
    jobs = [
        ("German", "data/validation/human_validation_probes_50_de.json",
         "data/validation/human_validation_probes_50_de_en_translated.json",
         "validation/packets/human-validation-packet-de-en.md"),
        ("French", "data/validation/human_validation_probes_50_fr.json",
         "data/validation/human_validation_probes_50_fr_en_translated.json",
         "validation/packets/human-validation-packet-fr-en.md"),
    ]
    for label, in_json, out_json, out_md in jobs:
        print(f"=== {label} -> English ===")
        translated = translate_packet(in_json, out_json, label)
        md = render_md(translated, label)
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"  wrote {out_md}")


if __name__ == "__main__":
    main()
