"""KL / ED for the PSII reproduction runs.

The authors' repository ships NO metric code, so these are re-implemented from the paper text
(Sec. 4.1): KL divergence between the human and the simulated answer distribution of each question,
and Entropy Deviation = |normalised entropy(simulated) - normalised entropy(human)|.
Unspecified details are exposed as variants and reported side by side:
  KL direction  : KL(human || sim)  [main]  or KL(sim || human)
  smoothing     : additive pseudo-count `alpha` on the simulated (and human) counts   [main: alpha = 1 on both, chosen because it reproduces the paper's PE and Direct KL]
  ED aggregation: 'q'   = mean over questions of |dH|   (strict reading of the text)
                  'agg' = |mean over questions of dH|   (aggregate; the only reading that can give 0.004)
Only per-question distribution summaries are written; no per-agent data leaves the output directory.

usage: analyze.py OUT_DIR TABLE_DIR
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

PSII = os.environ.get("PSII_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "psii_official"))
OUT, TAB = sys.argv[1], sys.argv[2]
os.makedirs(TAB, exist_ok=True)
Q = json.load(open(f"{PSII}/data/questions.json"))
QIDS = list(Q)[:259]
OPT = {q: sorted(int(k) for k in Q[q]["options"]) for q in QIDS}
df = pd.read_csv(f"{PSII}/data/WVS_Cross-National_Wave_7_csv_v6_0_100.csv", usecols=["id"] + QIDS, low_memory=False)
HUM = {q: np.array([(df[q].dropna().astype(int) == o).sum() for o in OPT[q]], float) for q in QIDS}


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def metrics(counts, q, alpha, direction="hs"):
    k = len(OPT[q])
    h, m = HUM[q], counts
    if h.sum() == 0 or m.sum() == 0:
        return None
    ph = (h + alpha) / (h + alpha).sum()
    pm = (m + alpha) / (m + alpha).sum()
    kl = float((ph * np.log(ph / pm)).sum()) if direction == "hs" else float((pm * np.log(pm / ph)).sum())
    # normalised entropy on the unsmoothed distributions
    Hh = ent(h / h.sum()) / math.log(k)
    Hm = ent(m / m.sum()) / math.log(k)
    return kl, Hm - Hh


def load_run(d):
    files = glob.glob(os.path.join(d, "*.json"))
    if len(files) < 100:
        return None
    cnt, bad, n = {}, 0, 0
    for f in files:
        r = json.load(open(f))
        for q, v in r.items():
            if q == "lang" or not isinstance(v, dict):
                continue
            n += 1
            x = v["result"]
            if x == "FAILED_TO_PARSE_NUMBER" or x not in OPT[q]:
                bad += 1
                continue
            cnt.setdefault(q, np.zeros(len(OPT[q])))[OPT[q].index(x)] += 1
    return cnt, bad / max(n, 1)


rows, perq = [], []
run_dirs = sorted(r for r, _, fs in os.walk(OUT) if "/seed_" in r + "/" and sum(f.endswith(".json") for f in fs) >= 100 and "/logs" not in r)
for d in run_dirs:
    if True:
        dirs = [d]
        dd = d
        rel = os.path.relpath(dd, OUT)
        arm = rel.split("/")[0]
        seed = int(re.search(r"seed_(\d+)", rel).group(1))
        runner = "official"
        stride = int(m.group(1)) if (m := re.search(r"qstride_(\d+)", rel)) else 1
        lr = load_run(dd)
        if lr is None:
            continue
        cnt, badfrac = lr
        qs = [q for q in QIDS if q in cnt]
        for variant, (alpha, direction) in {"main": (1.0, "hs"), "a0.5": (0.5, "hs"), "a0.01": (0.01, "hs"), "rev": (1.0, "sh")}.items():
            res = [metrics(cnt[q], q, alpha, direction) for q in qs]
            res = [r for r in res if r]
            kl = np.array([r[0] for r in res])
            dh = np.array([r[1] for r in res])
            rows.append(dict(arm=arm, seed=seed, runner=runner, stride=stride, variant=variant, nq=len(res),
                             KL=kl.mean(), ED_q=np.abs(dh).mean(), ED_agg=abs(dh.mean()),
                             Hsim=float(dh.mean()), bad=badfrac))
        if True:
            for q in qs:
                r = metrics(cnt[q], q, 1.0)
                perq.append(dict(arm=arm, seed=seed, runner=runner, q=q, KL=r[0], dH=r[1]))
R = pd.DataFrame(rows)
R.to_csv(f"{TAB}/run_metrics.csv", index=False)
pd.DataFrame(perq).to_csv(f"{TAB}/per_question_metrics.csv", index=False)
g = R.groupby(["runner", "stride", "arm", "variant"]).agg(
    n_seeds=("seed", "nunique"), nq=("nq", "first"), KL=("KL", "mean"), KL_sd=("KL", "std"),
    ED_q=("ED_q", "mean"), ED_q_sd=("ED_q", "std"), ED_agg=("ED_agg", "mean"), ED_agg_sd=("ED_agg", "std"),
    bad=("bad", "mean")).reset_index()
g.to_csv(f"{TAB}/arm_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(g[g.variant == "main"].round(3).to_string(index=False))
