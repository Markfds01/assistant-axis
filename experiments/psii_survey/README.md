# PSII (core) on a survey — Llama-3.1-8B-Instruct / Qwen2.5-32B-Instruct (code only)

The survey data, question texts, label maps, persona/question lists for vector building, vectors and all results are private and not in this repository.
Provide them locally; configuration by environment variable (defaults are relative paths):

- `PSII_SURVEY_CSV` — survey CSV (default `../../survey_results/1_data/clean_data/ES_translated.csv`)
- `PSII_QUESTIONS_FILE` — JSON list of `{id, block, text, options}` (default `data/questions.json`)
- `PSII_LABEL_MAPS` — JSON of raw label -> value key per attribute (default `data/label_maps.json`)
- `data/dv_questions.json`, `data/dv_personas.json` — inputs of `build_vectors*.py`
- `PSII_OUT` — directory for per-respondent outputs; `PSII_DIR` — clone of the authors' repo; `PSII_VALUE_FILE` — value vectors

Steps: `build_vectors.py` / `build_vectors_perlayer.py gen|merge` -> `run_methods.py`, `followup.py sens|run`, `perlayer.py sens|run`
-> `analyze*.py`. `scale*.py` and `layer_check.py` are diagnostics. `run_followup_*.sh` hold the command sequences.
Needs the HF cache of the models.
