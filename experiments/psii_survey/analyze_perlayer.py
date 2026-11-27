"""Tables for the per-layer PSII runs. Reads OUT/perlayer/runs/answers_<arm>_seed<k>.csv, writes results/perlayer/*.md|csv.
Readouts: G (main; sampled answer, parsed number, unparsed dropped) and R1 (option-logit sample of the same prefill).
Metrics as in analyze.py: KL(human || model) with eps=1e-6, ED, JS, MAE; question -> block -> overall; ground truth = the sampled
respondents' real answers (all items, whether or not the model's answer parsed).
"Item agreement" = share of items whose sampled answer equals that respondent's real answer (individual-level diagnostic; a model that answers at random scores near the chance level, a low KL with falling agreement means the distribution is matched by scrambling).
Paired contrasts: per-seed difference of the overall KL (mean +- sd over seeds) and a bootstrap over questions (2,000 draws, seed-averaged
per-question KL) -- the bootstrap covers question sampling, not seed variation."""
import glob, os, re, json
import numpy as np, pandas as pd, torch
from common import OUT, HERE, TEMP
from questions import QUESTIONS, QBY, BLOCKS
from analyze import dist, metrics

RES = f"{HERE}/results/perlayer"; os.makedirs(RES, exist_ok=True)
RUNS = f"{OUT}/perlayer/runs"
sig = {m: json.load(open(f"{RES}/sigma_perlayer_{m}.json"))["sigma_best"] if os.path.exists(f"{RES}/sigma_perlayer_{m}.json") else float("nan") for m in ("attr", "layer")}
sig22 = json.load(open(f"{HERE}/results/followup/sigma_llama_r1.json"))["sigma_best"]


STATS = {}  # arm -> list of (option mass, unparsed share) per seed


def load():
    rows = []
    for f in sorted(glob.glob(f"{RUNS}/answers_*_seed*.csv")):
        d = pd.read_csv(f, lineterminator="\n", usecols=["method", "seed", "resp", "qid", "ans_g", "ans_r1", "human", "opt_mass", "attempts"])
        name, seed = d.method[0], int(d.seed[0])
        ok = d.ans_g >= 0
        STATS.setdefault(name, []).append((float(d.opt_mass.mean()), float((d.ans_g < 0).mean()), float((d.ans_g[ok] == d.human[ok]).mean()), float((d.ans_r1 == d.human).mean())))
        for ro, col in (("G", "ans_g"), ("R1", "ans_r1")):
            for q in QUESTIONS:
                s = d[d.qid == q["id"]]; k = len(q["options"])
                ans = s[col].values; ans = ans[ans >= 0]
                pm = dist(ans, k) if len(ans) else np.full(k, 1.0 / k)
                rows.append(dict(arm=name, seed=seed, readout=ro, qid=q["id"], block=q["block"], n=len(ans), **metrics(dist(s.human.values, k), pm)))
    return pd.DataFrame(rows)


def aggregate(Q):
    blk = Q.groupby(["readout", "arm", "seed", "block"])[["KL", "ED", "JS", "MAE"]].mean().reset_index()
    ovr = Q.groupby(["readout", "arm", "seed"])[["KL", "ED", "JS", "MAE"]].mean().reset_index().assign(block="overall")
    A = pd.concat([blk, ovr])
    g = A.groupby(["readout", "arm", "block"])[["KL", "ED", "JS", "MAE"]].agg(["mean", "std", "count"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    return A, g.reset_index()


LABEL = {"pe": "PE",
         "pl_pe_noise_s007": "PE + noise σ=0.07 (same layers)", "pl_pe_noise_cal": "PE + noise σ=cal (same layers)",
         "pl_a1_psii_s007": "per-layer PSII σ=0.07", "pl_a1_psii_calattr": "per-layer PSII σ=cal, per-attribute draws", "pl_pe_noise_calattr": "PE + noise σ=cal, per-attribute draws", "pl_a1_psii_cal": "per-layer PSII σ=cal", "pl_a1_psii_nonoise": "per-layer PSII, no noise",
         "pl_a1_rand_s007": "random dir, σ=0.07", "pl_a1_rand_cal": "random dir, σ=cal", "pl_a1_rand_nonoise": "random dir, no noise",
         "pl_a2_psii_s007": "per-layer PSII σ=0.07 (coef 2)", "pl_a2_psii_cal": "per-layer PSII σ=cal (coef 2)", "pl_a2_psii_nonoise": "per-layer PSII, no noise (coef 2)",
         "pl_a2_rand_s007": "random dir, σ=0.07 (coef 2)", "pl_a2_rand_cal": "random dir, σ=cal (coef 2)", "pl_a2_rand_nonoise": "random dir, no noise (coef 2)",
         "pl_a1_psii_cal_B": "per-layer PSII σ=cal, chat-format vectors", "pl_a1_psii_cal_age21": "per-layer PSII σ=cal, age at layer 21",
         "pl_a1_randattr_cal": "random dirs, per-attribute norms, σ=cal",
         "pe_value": "PE + es value vector", "pl_a1_psii_cal_value": "per-layer PSII σ=cal + es value vector", "pl_pe_noise_cal_value": "PE + noise σ=cal + es value vector",
         "l22_psii_s007": "single-layer-22 PSII σ=0.07", "l22_psii_cal": "single-layer-22 PSII σ=cal", "l22_psii_nonoise": "single-layer-22 PSII, no noise",
         "l22_pe_noise_cal": "single-layer-22 PE + noise σ=cal", "l22_pe_noise_s007": "single-layer-22 PE + noise σ=0.07",
         "l22_rand_cal": "single-layer-22 random dir σ=cal", "l22_rand_nonoise": "single-layer-22 random dir, no noise"}
ORDER = list(LABEL)


def main_table(g, ro, arms):
    sub = g[(g.readout == ro) & (g.block == "overall")].set_index("arm")
    L = [f"| arm | seeds | KL | ED | JS | MAE | option mass | unparsed (G) | item agreement G / R1 |", "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        if a not in sub.index: continue
        r = sub.loc[a]
        L.append(f"| {LABEL.get(a, a)} | {int(r.KL_count)} | " + " | ".join(f"{r[c+'_mean']:.3f} ± {r[c+'_std']:.3f}" for c in ["KL", "ED", "JS", "MAE"]) + f" | {np.mean([x[0] for x in STATS[a]]):.3f} | {np.mean([x[1] for x in STATS[a]]):.1%} | {np.mean([x[2] for x in STATS[a]]):.3f} / {np.mean([x[3] for x in STATS[a]]):.3f} |")
    return "\n".join(L)


def block_table(g, ro, arms):
    sub = g[(g.readout == ro)]
    L = ["| arm | " + " | ".join(BLOCKS) + " |", "|---|" + "---|" * len(BLOCKS)]
    for a in arms:
        s = sub[sub.arm == a]
        if not len(s): continue
        L.append(f"| {LABEL.get(a, a)} | " + " | ".join(f"{s[s.block == b].KL_mean.iloc[0]:.2f} ± {s[s.block == b].KL_std.iloc[0]:.2f}" for b in BLOCKS) + " |")
    return "\n".join(L)


def contrasts(Q, A, ro, pairs, nb=2000):
    rng = np.random.default_rng(0)
    q = Q[Q.readout == ro].groupby(["arm", "qid"]).KL.mean().unstack(0)
    a_ = A[(A.readout == ro) & (A.block == "overall")].pivot(index="seed", columns="arm", values="KL")
    L = ["| contrast (A − B) | ΔKL, per-seed mean ± sd | ΔKL bootstrap over questions [95% CI] |", "|---|---|---|"]
    for x, y in pairs:
        if x not in a_.columns or y not in a_.columns: continue
        dd = (a_[x] - a_[y]).dropna()
        dq = (q[x] - q[y]).dropna().values
        bs = [dq[rng.integers(0, len(dq), len(dq))].mean() for _ in range(nb)]
        L.append(f"| {LABEL.get(x, x)} − {LABEL.get(y, y)} | {dd.mean():+.3f} ± {dd.std():.3f} (n={len(dd)}) | {dq.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] |")
    return "\n".join(L)


def movement(arms):
    """R1 item TV and argmax flips vs PE (same items), from the saved option logits."""
    L = ["| arm | item TV vs PE (R1) | argmax flips vs PE | option mass (R1) | unparsed (G) | re-asked (G) |", "|---|---|---|---|---|---|"]
    for a in arms:
        tv, fl, ms, un, ra = [], [], [], [], []
        for s in (0, 1, 2):
            f, f0 = f"{RUNS}/logits_{a}_seed{s}.npy", f"{RUNS}/logits_pe_seed{s}.npy"
            if not (os.path.exists(f) and os.path.exists(f0)): continue
            l, l0 = torch.tensor(np.load(f)), torch.tensor(np.load(f0))
            tv.append(float(0.5 * (torch.softmax(l / TEMP, -1) - torch.softmax(l0 / TEMP, -1)).abs().sum(-1).mean()))
            fl.append(float((l.argmax(-1) != l0.argmax(-1)).float().mean()))
            d = pd.read_csv(f"{RUNS}/answers_{a}_seed{s}.csv", usecols=["ans_g", "opt_mass", "attempts"])
            ms.append(float(d.opt_mass.mean())); un.append(float((d.ans_g < 0).mean())); ra.append(float((d.attempts > 1).mean()))
        if tv: L.append(f"| {LABEL.get(a, a)} | {np.mean(tv):.3f} | {np.mean(fl):.3f} | {np.mean(ms):.3f} | {np.mean(un):.1%} | {np.mean(ra):.1%} |")
    return "\n".join(L)


if __name__ == "__main__":
    Q = load(); Q.to_csv(f"{RES}/metrics_per_question.csv", index=False)
    A, g = aggregate(Q); g.to_csv(f"{RES}/metrics_long.csv", index=False)
    arms = [a for a in ORDER if a in set(g.arm)] + sorted(set(g.arm) - set(ORDER))
    out = [f"σ_cal per-layer, one draw per layer = {sig['layer']:.3f}; σ_cal per-layer with per-attribute draws (calattr) = {sig['attr']:.3f}; σ_cal single-layer-22 (one draw) = {sig22:.3f}\n"]
    for ro, title in (("G", "G: sampled answer (T=0.7, top_k=20, top_p=0.8, parsed number) — MAIN"), ("R1", "R1: option-logit sample (T=0.7) of the same prefill — SECONDARY")):
        out += [f"## {title}", "", "### Overall (sample ground truth; mean ± sd over seeds)", main_table(g, ro, arms), "", "### KL by block", block_table(g, ro, arms), ""]
        P = [("pl_a1_psii_cal", "pe"), ("pl_a1_psii_cal", "pl_pe_noise_cal"), ("pl_a1_psii_cal", "pl_a1_rand_cal"),
             ("pl_a1_psii_s007", "pe"), ("pl_a1_psii_s007", "pl_pe_noise_s007"), ("pl_a1_psii_s007", "pl_a1_rand_s007"),
             ("pl_a1_psii_nonoise", "pe"), ("pl_a1_psii_nonoise", "pl_a1_rand_nonoise"),
             ("pl_a2_psii_cal", "pl_pe_noise_cal"), ("pl_a2_psii_cal", "pl_a2_rand_cal"), ("pl_a2_psii_nonoise", "pl_a2_rand_nonoise"), ("pl_a2_psii_nonoise", "pe"),
             ("pl_a1_psii_cal", "l22_psii_cal"), ("pl_a1_psii_cal", "pl_a1_randattr_cal"),
             ("l22_psii_cal", "l22_pe_noise_cal"), ("l22_psii_cal", "l22_rand_cal"), ("l22_psii_nonoise", "pe"),
             ("pl_a1_psii_cal_value", "pl_pe_noise_cal_value"), ("pe_value", "pe")]
        out += ["### Paired contrasts (KL)", contrasts(Q, A, ro, P), ""]
    out += ["## Movement vs PE (R1 logits) and parse statistics", movement(arms)]
    open(f"{RES}/tables.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))
