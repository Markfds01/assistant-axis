"""Metrics from answers_*.csv in OUT. KL = KL(P_human || P_model), natural log, both distributions
smoothed with eps=1e-6 and renormalised. ED = |H(model)-H(human)|/log(k). JS natural log (sqrt not taken).
MAE = mean over options of |p_model - p_human|. Questions -> block mean -> overall mean (mean over questions).
Ground truth: (a) same sampled respondents' real answers, (b) full sample (all respondents)."""
import glob, os, sys, json
import numpy as np, pandas as pd
from common import OUT, HERE, load_survey
from questions import QUESTIONS, QBY, BLOCKS
EPS = 1e-6


def dist(idx, k):
    c = np.bincount(idx, minlength=k).astype(float); return c / c.sum()


def metrics(p_h, p_m):
    k = len(p_h)
    ph = (p_h + EPS) / (p_h + EPS).sum(); pm = (p_m + EPS) / (p_m + EPS).sum()
    kl = float((ph * np.log(ph / pm)).sum())
    H = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    ed = abs(H(p_m) - H(p_h)) / np.log(k)
    m = 0.5 * (ph + pm)
    js = float(0.5 * (ph * np.log(ph / m)).sum() + 0.5 * (pm * np.log(pm / m)).sum())
    return dict(KL=kl, ED=ed, JS=js, MAE=float(np.abs(p_m - p_h).mean()))




def build_tables(files, outdir, name_fn=lambda d: d.method[0], results_prefix="table"):
    """files: answers csv paths. Writes per-question metrics + <outdir>/<results_prefix>_{sample,full}.md; returns long table."""
    df = load_survey()
    full = {q["id"]: dist(np.array([q["options"].index(v) for v in df[q["id"]] if v in q["options"]]), len(q["options"]))
            for q in QUESTIONS}
    rows = []
    for f in files:
        d = pd.read_csv(f, lineterminator="\n"); name = name_fn(d)
        for q in QUESTIONS:
            s = d[d.qid == q["id"]]; k = len(q["options"])
            pm = dist(s.answer.values, k)
            for gt, ph in (("sample", dist(s.human.values, k)), ("full", full[q["id"]])):
                rows.append(dict(method=name, seed=d.seed[0], qid=q["id"], block=q["block"], gt=gt, **metrics(ph, pm)))
    R = pd.DataFrame(rows)
    blk = R.groupby(["gt", "method", "seed", "block"])[["KL", "ED", "JS", "MAE"]].mean().reset_index()
    ovr = R.groupby(["gt", "method", "seed"])[["KL", "ED", "JS", "MAE"]].mean().reset_index().assign(block="overall")
    A = pd.concat([blk, ovr])
    g = A.groupby(["gt", "method", "block"])[["KL", "ED", "JS", "MAE"]].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return R, g.reset_index()


def write_md(g, gt, path, order=None, title=None):
    sub = g[g["gt"] == gt]
    lines = [f"### {title or 'Ground truth: ' + gt}\n", "| method | block | KL | ED | JS | MAE |", "|---|---|---|---|---|---|"]
    meths = list(order) if order else sorted(sub.method.unique())
    for m in meths:
        if m not in set(sub.method): continue
        for b in ["overall"] + BLOCKS:
            r = sub[(sub.method == m) & (sub.block == b)].iloc[0]
            lines.append(f"| {m} | {b} | " + " | ".join(f"{r[c+'_mean']:.3f} ± {r[c+'_std']:.3f}" for c in ["KL", "ED", "JS", "MAE"]) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    df = load_survey()
    full = {q["id"]: dist(np.array([q["options"].index(v) for v in df[q["id"]] if v in q["options"]]), len(q["options"]))
            for q in QUESTIONS}
    rows = []
    for f in sorted(glob.glob(f"{OUT}/answers_*_seed*.csv")):
        d = pd.read_csv(f)
        for q in QUESTIONS:
            s = d[d.qid == q["id"]]; k = len(q["options"])
            pm = dist(s.answer.values, k)
            for gt, ph in (("sample", dist(s.human.values, k)), ("full", full[q["id"]])):
                rows.append(dict(method=d.method[0], seed=d.seed[0], qid=q["id"], block=q["block"], gt=gt, **metrics(ph, pm)))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/per_question_metrics.csv", index=False)
    blk = R.groupby(["gt", "method", "seed", "block"])[["KL", "ED", "JS", "MAE"]].mean().reset_index()
    ovr = R.groupby(["gt", "method", "seed"])[["KL", "ED", "JS", "MAE"]].mean().reset_index().assign(block="overall")
    A = pd.concat([blk, ovr])
    g = A.groupby(["gt", "method", "block"])[["KL", "ED", "JS", "MAE"]].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    g = g.reset_index(); g.to_csv(f"{HERE}/results/metrics_long.csv", index=False)
    order = {"direct": 0, "pe": 1, "pv": 2}
    for gt in ("sample", "full"):
        sub = g[g["gt"] == gt]
        lines = [f"### Ground truth: {gt}\n", "| method | block | KL | ED | JS | MAE |", "|---|---|---|---|---|---|"]
        meths = sorted(sub.method.unique(), key=lambda m: (order.get(m, 3), m))
        for m in meths:
            for b in ["overall"] + BLOCKS:
                r = sub[(sub.method == m) & (sub.block == b)].iloc[0]
                lines.append(f"| {m} | {b} | " + " | ".join(f"{r[c+'_mean']:.3f} ± {r[c+'_std']:.3f}" for c in ["KL", "ED", "JS", "MAE"]) + " |")
        open(f"{HERE}/results/table_{gt}.md", "w").write("\n".join(lines) + "\n")
    print(open(f"{HERE}/results/table_sample.md").read())
