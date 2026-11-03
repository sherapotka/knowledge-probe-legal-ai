# knowledge-probe-legal-ai

Code and data for

> Shera Potka, Alex Thomo, Paula Helm, Wulf Loh. *Selective Overconfidence: How Large
> Language Models Underserve Social Welfare Law Across Jurisdictions.* Proceedings of the
> AAAI/ACM Conference on AI, Ethics, and Society (AIES 2026), 9(3), 2130-2141.
> https://doi.org/10.1609/aies.v9i3.43294

Companion site: https://selective-overconfidence.vercel.app

## What the pipeline does

A knowledge-probe pipeline turns published court decisions into legal questions and
measures whether a language model admits uncertainty when it answers them without seeing
the decision. Four steps, all run with Llama 3.3 70B unless a script says otherwise:

1. Extract ground truth from a court decision (legal test, statutes, outcome).
2. Generate three probe questions per case (legal test, statute, procedure).
3. Answer each question blind, with the instruction "If unsure, say so".
4. Judge the blind answer against the ground truth: admitted uncertainty (yes or no) and
   accuracy (1 to 5).

If the model hedges less in one legal domain than in another of similar difficulty, that is
selective overconfidence. The paper compares tax law with social welfare law in Germany,
France and Canada (200 cases per domain per country) and finds a large gap in Germany, a
smaller one in France and none in Canada; the German gap replicates on Qwen3 32B and closes
by 87 percent when related legal text is retrieved at query time.

## Layout

```
scripts/        pipeline, replication and analysis scripts; scripts/README.md maps each
                script to the data file it writes and the paper table it feeds
data/
  german/       BFH tax and BSG social runs; Qwen3, gpt-oss and retrieval replications
  french/       Conseil d'Etat tax and social security runs
  canadian/     Tax Court of Canada and Social Security Tribunal runs
  validation/   the 50-probe stratified samples used for human validation, with the
                judge's verdicts and English translations
validation/     the annotated human-validation packets of both annotators and the
                judge-versus-human disagreement analyses behind the paper's appendix
```

## What is in the data files

Each `*_progress.json` holds, per case, the ground truth the model extracted, the three
probes, the blind answers and the judge's scores. The `*_strict.json` and `*_liberal.json`
siblings are re-judgements under the two explicit hedging rubrics described in the paper's
appendix. The `*_cases.json` files and `german_rag_corpus.json` list the decisions each run
used, with court, date, file number, ECLI or citation, source identifier and character count.

**The decision texts themselves are not in this repository.** They are available from the
portals that publish them, each under its own terms of use. Everything a run needs to be
repeated is here except those texts, and `scripts/refetch_decisions.py`
downloads them again from the original sources into the `text` field of each record:

| Jurisdiction | Source | Identifier kept |
|---|---|---|
| Germany | Open Legal Data, https://de.openlegaldata.io | `case_id`, file number, ECLI |
| France | Conseil d'Etat open data, https://opendata.justice-administrative.fr | `source_file`, `ecli`, `date` |
| Canada | A2AJ, https://a2aj.ca | `citation` |

Not included, because none of their numbers are in the paper: the pilot runs on a handful
of cases, an exploratory Indigenous-rights domain, and the refugee and asylum decisions of
an early tax-versus-refugee design (the fetch scripts still retrieve them, but their records
were removed from every released file; the Canadian and German social-welfare domains the
paper reports are the Social Security Tribunal and the Bundessozialgericht).

## Conventions

- The paper's numbers come from `german_tax_200_seed42.json` (200 of the 500 BFH cases,
  drawn by `scripts/subsample_german_tax_200.py`) and `bsg_probe_200_progress.json` for
  Germany, `french_200_progress.json` for France, `canadian_200_progress.json` for Canada,
  and `german_qwen_progress.json`, `german_gptoss_progress.json` and
  `german_rag_progress.json` for the replications and the retrieval experiment.
  `scripts/recompute_no_citation.py` prints them.
- Citation-recall probes are excluded from all hedging rates. Each case has three other
  probes, so a 200-case domain contributes 600.
- A judge call that returned no admitted-uncertainty label counts as not hedging in
  `recompute_no_citation.py`; `compare_rubrics_all.py` uses labelled probes only.
- `compute_kappa.py` prints each annotator's agreement with the judge; the paper's appendix
  table reports the mean of the two. For German, one annotator used the original packet and
  the other an English translation (`validation/packets/`).

## Reproducing

Python 3.10 or later. Install the dependencies and provide a Groq API key:

```
pip install -r requirements.txt
cp .env.example .env        # then put your key in it
python scripts/refetch_decisions.py
```

Run scripts from the repository root. The analysis scripts (`recompute_no_citation.py`,
`compare_rubrics_all.py`, `compute_kappa.py`, `extract_disagreements.py`) work on the
shipped data and need no model calls or decision texts. The pipeline scripts call the Groq
API and need the decision texts; several older ones open bare file names such as
`bsg_cases.json`, so run those from inside the directory that holds the file, or edit the
path. `scripts/README.md` says which is which.

Model outputs are not deterministic. Re-running a pipeline script produces new answers and
new scores; the shipped files are the ones the paper reports.

## Licences

Code is released under the MIT licence (`LICENSE`). The derived data (ground-truth
extracts, probes, model answers, judge scores, human annotations) is released under
CC BY 4.0 (`DATA_LICENSE.md`). The court decisions are not part of this release and
remain subject to the terms of the portals that publish them.

## Citing

See `CITATION.cff`, or cite the AIES 2026 paper above.
