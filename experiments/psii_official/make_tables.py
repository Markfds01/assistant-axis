"""Markdown results table from arm_summary.csv + run_metrics.csv. usage: make_tables.py TABLE_DIR > table.md"""
import sys
import pandas as pd
T = sys.argv[1]
g = pd.read_csv(f"{T}/arm_summary.csv")
g = g[(g.variant == "main") & (g.runner == "official")].set_index("arm")
paper = {  # Llama-3.1-8B, WVS, Overall (Tables 1 and 2)
    "direct": (1.2856, 0.6326), "pe": (0.8095, 0.3831), "psii": (0.4017, 0.0040), "no_value": (0.4638, 0.0416),
    "no_demo": (0.6681, 0.2729), "no_profile": (0.5575, 0.0630), "no_noise": (0.5673, 0.2068), "layer70": (0.5225, 0.1089)}
label = {
    "direct": "Direct", "pe": "PE (profile prompt)", "psii": "PSII (full)",
    "no_value": "PSII w/o value vector", "no_demo": "PSII w/o demographic vectors (as run.sh: value+PE, no noise)",
    "no_profile": "PSII w/o profile prompt", "no_noise": "PSII w/o noise (sigma=0)", "layer70": "PSII w/o layer-wise (all at layer 22)",
    "pe_multi": "[ctrl] PE in 5 random languages, no vectors, no noise",
    "noise_only_value": "[ctrl] noise only (zero demo vectors) + value vector + PE",
    "noise_only_multi": "[ctrl] noise only (zero demo vectors) + random language + PE, no value vector",
    "noise_only": "[ctrl] noise only (zero demo vectors) + PE, English, no value vector",
    "demo_only": "[ctrl] demographic vectors only, sigma=0, + PE (no value vector)",
    "random_value": "[ctrl] random vectors (same norms) + noise + value vector + PE",
    "random": "[ctrl] random vectors (same norms) + noise + PE, no value vector",
    "random_nonoise": "[ctrl] random vectors (same norms), sigma=0, + PE",
    "pe_novllm": "PE via HF generate (no vllm)"}
f = lambda m, s: "-" if pd.isna(m) else (f"{m:.3f}" if pd.isna(s) else f"{m:.3f} ± {s:.3f}")
print("| arm | seeds | paper KL / ED | ours KL | ours ED (per-question) | ours ED (aggregate) | unparsed |")
print("|---|---|---|---|---|---|---|")
for a in label:
    if a not in g.index:
        continue
    r = g.loc[a]
    p = paper.get(a)
    print(f"| {label[a]} | {int(r.n_seeds)} | {'%.3f / %.3f' % p if p else '-'} | {f(r.KL, r.KL_sd)} | {f(r.ED_q, r.ED_q_sd)} | {f(r.ED_agg, r.ED_agg_sd)} | {r.bad*100:.1f}% |")
