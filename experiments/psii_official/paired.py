"""Paired per-question contrasts between arms (bootstrap over the questions). usage: paired.py TABLE_DIR"""
import sys
import numpy as np
import pandas as pd
T = sys.argv[1]
d = pd.read_csv(f"{T}/per_question_metrics.csv")
d = d[d.runner == "official"]
kl = d.groupby(["arm", "q"]).KL.mean().unstack(0)
dh = d.groupby(["arm", "q"]).dH.mean().unstack(0)
rng = np.random.default_rng(0)
pairs = [("noise_only_value", "psii"), ("random_value", "psii"), ("noise_only_multi", "psii"), ("noise_only", "no_value"),
         ("random", "no_value"), ("noise_only_value", "pe_multi"), ("psii", "pe_multi"), ("no_value", "pe"), ("demo_only", "pe"),
         ("noise_only", "pe"), ("no_noise", "psii"), ("no_demo", "psii")]
print("| contrast (A - B) | nq | dKL mean [95% CI] | d|ED_q| mean [95% CI] |")
print("|---|---|---|---|")
for a, b in pairs:
    if a not in kl or b not in kl:
        continue
    ok = kl[a].notna() & kl[b].notna()
    x = (kl[a] - kl[b])[ok].values
    y = (dh[a].abs() - dh[b].abs())[ok].values
    def ci(v):
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        return f"{v.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
    print(f"| {a} - {b} | {ok.sum()} | {ci(x)} | {ci(y)} |")
