"""Tables for the follow-up (R1 / R3 readouts). Reads OUT/followup/<spec>/<readout>/, writes results/followup/*.md.
Usage: python analyze_followup.py --spec llama
Movement metrics (vs the unedited PE run, same items): item TV = mean_items 0.5*sum|P - P_pe| with P = softmax(logits/0.7);
population TV = mean over questions of 0.5*sum|mean_items P - mean_items P_pe| (expected answer distribution, no sampling noise);
flip = share of items whose argmax option changes."""
import argparse, glob, os, re, json
import numpy as np, pandas as pd, torch
from common import OUT, HERE, TEMP
from questions import QUESTIONS, QBY, BLOCKS
from analyze import build_tables, write_md

ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="llama"); a = ap.parse_args()
RES = f"{HERE}/results/followup"; os.makedirs(RES, exist_ok=True)
MAIN = ["direct", "pe", "pv", "psii", "psii_nonoise", "pe_noise"]
LABEL = {"direct": "Direct", "pe": "PE", "pv": "PV (coeff 2)", "psii": "PSII (calibrated σ)", "psii_nonoise": "PSII − noise",
         "pe_noise": "PE + noise (no vectors)"}
sig = {r: json.load(open(f"{RES}/sigma_{a.spec}_{r}.json"))["sigma_best"] for r in ("r1", "r3") if os.path.exists(f"{RES}/sigma_{a.spec}_{r}.json")}


def files(readout, names=None):
    fs = sorted(glob.glob(f"{OUT}/followup/{a.spec}/{readout}/answers_*_seed*.csv"))
    return [f for f in fs if names is None or re.sub(r"_seed\d+\.csv", "", os.path.basename(f))[len("answers_"):] in names]


def nice(name, readout):
    sg = sig.get(readout, float("nan"))
    return {"psii": f"PSII (σ={sg:.3f})", "pe_noise": f"PE + noise (σ={sg:.3f}, no vectors)"}.get(name, LABEL.get(name, name))


def alpha_sigma(n, readout):
    """name -> (alpha, sigma label, flag) for sweep tables; named methods map to their sweep cell."""
    s_cal = f"{sig[readout]:.3f}" if readout in sig else "cal"
    fixed = {"pe": (0.0, "0", ""), "psii_nonoise": (1.0, "0", ""), "pe_noise": (0.0, s_cal, ""), "psii": (1.0, s_cal, "")}
    if n in fixed: return fixed[n]
    m = re.match(r"lin_a([\d.]+)_s([\d.]+|cal)(_np|_rand)?$", n)
    return (float(m.group(1)), s_cal if m.group(2) == "cal" else m.group(2), m.group(3) or "") if m else None


def load_logits(readout, name, seed):
    f = f"{OUT}/followup/{a.spec}/{readout}/logits_{name}_seed{seed}.npy"
    return torch.tensor(np.load(f)) if os.path.exists(f) else None


def movement(readout, name):
    tv, pop, flip = [], [], []
    for seed in (0, 1, 2):
        L, L0 = load_logits(readout, name, seed), load_logits(readout, "pe", seed)
        if L is None or L0 is None: continue
        d = pd.read_csv(f"{OUT}/followup/{a.spec}/{readout}/answers_pe_seed{seed}.csv", usecols=["qid"])
        P, P0 = torch.softmax(L / TEMP, -1), torch.softmax(L0 / TEMP, -1)
        tv.append(float(0.5 * (P - P0).abs().sum(-1).mean()))
        flip.append(float((L.argmax(-1) != L0.argmax(-1)).float().mean()))
        pv = []
        for q in QUESTIONS:
            idx = torch.tensor((d.qid == q["id"]).values)
            pv.append(float(0.5 * (P[idx].mean(0) - P0[idx].mean(0)).abs().sum()))
        pop.append(np.mean(pv))
    return (np.mean(tv), np.mean(pop), np.mean(flip)) if tv else (np.nan,) * 3


# ---- coherence heuristic (R3) ----
ES_STOP = set("de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos todos uno les ni otros ese eso ante e esto mi antes qué yo otro otra él tanto esa mucho nada poco ella estar algo mis tú te tu soy creo estoy tengo siento si es son era fue ser tiene tienen puede hace".split())
EN_STOP = set("the and i to of is that it my but with for in on are was this have be not as".split())
WORD = re.compile(r"[^\W\d_]+")


def lexicon(readout):
    cnt = {}
    for f in files(readout, {"direct", "pe"}):
        for t in pd.read_csv(f, usecols=["rationale"], lineterminator="\n").rationale.fillna(""):
            for w in WORD.findall(t.lower()): cnt[w] = cnt.get(w, 0) + 1
    return {w for w, c in cnt.items() if c >= 3}


def coherent(t, lex):
    """Coherent Spanish = >=6 words, Spanish function words >=15%, English function words <15%, no non-Latin script,
    <=20% of words outside the lexicon of words used >=3 times in the unedited Direct/PE rationales, no long repetition."""
    if not isinstance(t, str): return False
    w = WORD.findall(t.lower())
    if len(w) < 6: return False
    if re.search(r"[Ѐ-ӿ　-鿿؀-ۿ]", t): return False
    es = sum(x in ES_STOP for x in w) / len(w); en = sum(x in EN_STOP for x in w) / len(w)
    oov = sum(x not in lex for x in w) / len(w); uniq = len(set(w)) / len(w)
    return es >= 0.15 and en < 0.15 and oov <= 0.20 and uniq >= 0.5


def fmt(m, s): return f"{m:.3f} ± {s:.3f}"


def agg(g, name, gt="sample", block="overall"):
    r = g[(g["gt"] == gt) & (g.method == name) & (g.block == block)]
    return None if r.empty else r.iloc[0]


for readout in ("r1", "r3"):
    fs = files(readout)
    if not fs: continue
    R, g = build_tables(fs, RES)
    g.to_csv(f"{RES}/metrics_long_{a.spec}_{readout}.csv", index=False)
    present = set(g.method)
    # main table
    for gt in ("sample", "full"):
        write_md(g, gt, f"{RES}/main_{a.spec}_{readout}_{gt}.md", order=[m for m in MAIN if m in present],
                 title=f"{a.spec} {readout.upper()} — ground truth: {gt}")
    # sweep table
    rows = []
    mass = {}
    for f in fs:
        d = pd.read_csv(f, usecols=["method", "opt_mass"], lineterminator="\n"); mass.setdefault(d.method[0], []).append(d.opt_mass.mean())
    lines = [f"### {a.spec} {readout.upper()} — scale sweep (profile prompt; sample ground truth; mean ± sd over the available seeds; sd = nan with 1 seed)", "",
             "| α | σ | name | KL | ED | JS | MAE | item TV vs PE | pop. TV vs PE | argmax flips | option mass |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    cells = []
    for n in present:
        k = alpha_sigma(n, readout)
        if k is None or (n == "direct") or (n == "pv"): continue
        cells.append((k[2], k[0], 0.0 if k[1] == "0" else float(k[1]), n, k))
    for flag, al, sg, n, k in sorted(cells):
        r = agg(g, n); mv = movement(readout, n)
        lines.append(f"| {al:g} | {k[1]} | {n}{' ('+flag[1:]+')' if flag else ''} | " + " | ".join(fmt(r[c + '_mean'], r[c + '_std']) for c in ["KL", "ED", "JS", "MAE"]) +
                     f" | {mv[0]:.3f} | {mv[1]:.3f} | {mv[2]:.3f} | {np.mean(mass[n]):.3f} |")
    open(f"{RES}/sweep_{a.spec}_{readout}.md", "w").write("\n".join(lines) + "\n")
    # compact main summary
    lines = [f"| method | KL | ED | JS | MAE | option mass |", "|---|---|---|---|---|---|"]
    for m in MAIN:
        r = agg(g, m)
        if r is not None:
            lines.append(f"| {nice(m, readout)} | " + " | ".join(fmt(r[c + '_mean'], r[c + '_std']) for c in ["KL", "ED", "JS", "MAE"]) + f" | {np.mean(mass[m]):.3f} |")
    open(f"{RES}/summary_{a.spec}_{readout}.md", "w").write("\n".join(lines) + "\n")

# ---- R3: coherence + example rationales ----
fs = files("r3")
if fs:
    lex = lexicon("r3"); rng = np.random.default_rng(0)
    out = {}; ex = [f"# Example R3 rationales ({a.spec}); text only, 5 random per config (seed 0), no respondent identifiers", ""]
    for f in fs:
        d = pd.read_csv(f, lineterminator="\n"); name = d.method[0]
        ok = d.rationale.map(lambda t: coherent(t, lex)); out.setdefault(name, []).append(1 - ok.mean())
        if d.seed[0] == 0:
            ex += [f"## {name}", ""]
            for i in rng.choice(len(d), 5, replace=False):
                ex.append(f"- [{d.qid[i]}] {str(d.rationale[i])!r}")
            ex.append("")
    open(f"{RES}/rationales_{a.spec}.md", "w").write("\n".join(ex))
    lines = ["| config | share not coherent Spanish (mean over seeds) | seeds |", "|---|---|---|"]
    for n in sorted(out, key=lambda n: (n not in MAIN, n)):
        lines.append(f"| {n} | {np.mean(out[n]):.1%} | {len(out[n])} |")
    open(f"{RES}/coherence_{a.spec}.md", "w").write("\n".join(lines) + "\n")
    print(open(f"{RES}/coherence_{a.spec}.md").read())
for r in ("r1", "r3"):
    for fn in (f"summary_{a.spec}_{r}.md", f"sweep_{a.spec}_{r}.md"):
        if os.path.exists(f"{RES}/{fn}"): print(fn); print(open(f"{RES}/{fn}").read())

# ---- R3 sampling floor: the same PE configuration rerun with other rationale seeds (item/pop TV is not 0 even with no edit) ----
fl = f"{OUT}/followup/{a.spec}/r3/logits_pe_floor_seed0.npy"
if os.path.exists(fl):
    L, L0 = torch.tensor(np.load(fl)), load_logits("r3", "pe", 0)
    d = pd.read_csv(f"{OUT}/followup/{a.spec}/r3/answers_pe_seed0.csv", usecols=["qid"], lineterminator="\n")
    P, P0 = torch.softmax(L / TEMP, -1), torch.softmax(L0 / TEMP, -1)
    pop = np.mean([float(0.5 * (P[torch.tensor((d.qid == q["id"]).values)].mean(0) - P0[torch.tensor((d.qid == q["id"]).values)].mean(0)).abs().sum()) for q in QUESTIONS])
    txt = (f"R3 sampling floor ({a.spec}, seed 0): PE vs PE rerun with different rationale seeds: item TV {float(0.5 * (P - P0).abs().sum(-1).mean()):.3f}, "
           f"population TV {pop:.3f}, argmax flips {float((L.argmax(-1) != L0.argmax(-1)).float().mean()):.3f}\n")
    open(f"{RES}/floor_{a.spec}_r3.md", "w").write(txt); print(txt)
