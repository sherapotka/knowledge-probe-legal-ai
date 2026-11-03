"""Judge the 50-probe human-validation subsets under the updated (human-matched) rubrics.

Inputs (one per language):
  data/validation/human_validation_probes_50_de.json  (German originals)
  data/validation/human_validation_probes_50_fr.json  (French originals)
  data/validation/human_validation_probes_50_en.json  (Canadian English)

Outputs:
  data/validation/human_validation_probes_50_<lang>_judged.json
  containing each probe with two new judge labels: judge_strict and judge_liberal,
  alongside the existing judge_uncertainty (the original unrubricked label).
"""
import json, os, sys, time
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerun_judge_rubrics import RUBRICS  # reuse updated rubrics

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"


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
                time.sleep(10 * (attempt + 1))
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


def judge(language, rubric_name, question, answer):
    block = RUBRICS[language]
    prompt = block["template"].format(
        rubric=block[rubric_name],
        question=question or "",
        blind_answer=answer or "",
    )
    return parse_label(ask(prompt))


JOBS = {
    "de": "data/validation/human_validation_probes_50_de.json",
    "fr": "data/validation/human_validation_probes_50_fr.json",
    "en": "data/validation/human_validation_probes_50_en.json",
}


def main():
    for lang, in_path in JOBS.items():
        out_path = in_path.replace(".json", "_judged.json")
        print(f"=== {lang}: {in_path} ===")
        with open(in_path, encoding="utf-8") as f:
            probes = json.load(f)
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                out = json.load(f)
        else:
            out = [dict(p) for p in probes]
        for i, p in enumerate(probes):
            rec = out[i]
            for rubric in ("strict", "liberal"):
                key = f"judge_{rubric}"
                if rec.get(key) in ("yes", "no"):
                    continue
                label, reason = judge(lang, rubric, p.get("question", ""), p.get("answer", ""))
                rec[key] = label or "ERROR"
                rec[f"judge_{rubric}_reason"] = reason
            if (i + 1) % 10 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                print(f"  judged {i+1}/{len(probes)}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  done -> {out_path}")


if __name__ == "__main__":
    main()
