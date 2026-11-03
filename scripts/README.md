# Scripts

Every pipeline script calls the Groq API with `GROQ_API_KEY` from `.env`. The pipeline is
the same in every jurisdiction: fetch decisions, extract ground truth from each decision,
generate three probe questions per case, answer them blind (no case text), judge the blind
answer against the ground truth. All four steps use Llama 3.3 70B unless noted.

Run everything from the repository root. Scripts marked "uses `data/` paths" open the
shipped files directly. The older ones open bare file names such as `bsg_cases.json`; run
those from inside the directory that holds the file, or edit the path. Pipeline scripts need
the decision texts, which `refetch_decisions.py` restores (see the top-level README).

## Getting the decision texts back

| Script | What it does |
|---|---|
| `refetch_decisions.py` (uses `data/` paths) | Downloads the court decisions listed in every `*_cases.json` file and `german_rag_corpus.json` from Open Legal Data, the Conseil d'Etat open-data portal and A2AJ, and writes them into the `text` field in place |

## Final experiments (the numbers in the AIES 2026 paper)

| Script | What it does | Writes |
|---|---|---|
| `german_fetch_500.py` | Fetch BFH (tax) decisions from Open Legal Data (it also fetched BVerwG asylum decisions for an early design; those records are not released) | `data/german/german_500_cases.json` |
| `german_auto_probe_500.py` | Full pipeline on the German tax side (500 cases) | `data/german/german_500_progress.json` |
| `subsample_german_tax_200.py` (uses `data/` paths) | Draws the 200 tax cases the paper reports from the 500-case run (`random.seed(42)`, `random.sample`); run without `--write` it checks that the draw matches the shipped file | `data/german/german_tax_200_seed42.json` |
| `fetch_probe_bsg.py`, `extend_bsg.py`, `extend_bsg_200.py` | German social side, built up 50 → 100 → 200 cases | `data/german/bsg_cases_200.json`, `bsg_probe_200_progress.json` |
| `canadian_fetch_500.py` | Fetch Tax Court of Canada decisions from A2AJ (it also fetched Refugee Protection Division decisions for an early design; those records are not released) | `data/canadian/canadian_500_cases.json` |
| `canadian_probe_split.py`, `extend_canadian_split.py` | First Canadian tax run in the split-model design (Llama 3.1 8B answers, 70B judges) on the first 100 cases of the fetch; its ground truths were kept and the answers redone with 70B | `data/canadian/canadian_split_progress.json` |
| `rerun_canadian_70b.py`, `extend_canadian_200.py`, `extend_canadian_sst_fix.py` | Canadian tax and SST (social) with 70B throughout, built up to 200 + 200: the first 100 tax and 100 SST cases re-answered, then 100 more of each fetched (the last 57 SST and 7 tax with different queries and older cases, to avoid duplicates) | `data/canadian/canadian_70b_progress.json`, `canadian_200_progress.json` |
| `fetch_probe_sst.py` | First SST (Social Security Tribunal) fetch and probe | `data/canadian/sst_cases.json`, `sst_probe_progress.json` |
| `french_fetch_pilot.py`, `french_probe_pilot.py`, `french_fix_2.py` | French tax and social security from Conseil d'Etat open data, pilot then 200 + 200 | `data/french/french_200_cases.json`, `french_200_progress.json` |
| `german_qwen_experiment.py` | Replication on Qwen3 32B, reusing German ground truths | `data/german/german_qwen_progress.json` |
| `german_gptoss_experiment.py` | Replication on gpt-oss 120B, reusing German ground truths | `data/german/german_gptoss_progress.json` |
| `german_rag_experiment.py` | Retrieval intervention (BM25, disjoint corpus) on Germany | `data/german/german_rag_corpus.json`, `german_rag_progress.json` |
| `recompute_no_citation.py` (uses `data/` paths) | Recompute every headline number with citation-recall probes dropped | stdout |

## Judge validation (appendix table)

| Script | What it does | Writes |
|---|---|---|
| `rerun_judge_rubrics.py` | Re-judge every probe under explicit strict and liberal rubrics | `*_strict.json`, `*_liberal.json` next to each progress file |
| `compare_rubrics_all.py`, `compare_rubrics_german.py` (use `data/` paths) | Hedging rates under original vs strict vs liberal rubric, all experiments | stdout |
| `generate_validation_packets.py` (uses `data/` paths) | Draw the 50-probe stratified samples and write annotator packets | `data/validation/human_validation_probes_50_*.json`, `validation/packets/` |
| `translate_validation_packets.py` (uses `data/` paths) | English translations of the German and French packets | `data/validation/*_en_translated.json` |
| `judge_50_subset.py` (uses `data/` paths) | Judge the 50-probe samples under the human-matched rubrics | `data/validation/*_judged.json` |
| `compute_kappa.py` (uses `data/` paths) | Cohen's kappa, human vs human and human vs judge | stdout |
| `extract_disagreements.py` (uses `data/` paths) | Probes where both humans agree and the judge differs | `validation/disagreements-judge-vs-humans-*.md` |

## Intermediate files

The runs were built up in stages (50, 100, 200 cases; 8B then 70B answers), and each stage
reads the previous one, so the intermediate files are kept: `bsg_cases.json` and
`bsg_probe_progress.json` (first 100 BSG cases), `french_pilot_*` and `french_50_*`,
`canadian_split_progress.json`, `canadian_70b_progress.json`, `sst_cases.json` and
`sst_probe_progress.json`. Only the 200-case files carry the numbers in the paper.
