"""Run Direct / PE / PV / core PSII (+ablations) on a sample of respondents.
Each agent answers each question it has a real answer for, once. The answer is read from a single
constrained decoding step: logits of the option-number tokens at the first answer position, sampled with
temperature 0.7 (top_k 20 is a no-op over <= 6 options).
Usage: python run_methods.py --device cuda:0 --seeds 0 1 2 --methods direct pe pv psii psii_nonoise psii_noprofile --sigma 0.39 --tag cal
       python run_methods.py --sensitivity --sigma 0.07"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--methods", nargs="+", default=["direct", "pe", "pv", "psii", "psii_nonoise", "psii_noprofile"])
ap.add_argument("--sigma", type=float, default=0.07); ap.add_argument("--tag", default="")
ap.add_argument("--n", type=int, default=100); ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--pv_coeff", type=float, default=2.0)
ap.add_argument("--sensitivity", action="store_true")
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

df = load_survey()
idents = [respondent_identity(r) for _, r in df[ATTRIBUTES].iterrows()]
complete = np.array([all(v is not None for v in i.values()) for i in idents])
vecs = torch.load(f"{HERE}/data/demographic_vectors_layer{LAYER}.pt")
model, tok = load_model(a.device)
inj = Injector(model)
opt_ids = option_token_ids(tok)


def sample_ids(seed):
    return np.random.default_rng(seed).choice(np.where(complete)[0], a.n, replace=False)


def respondent_vec(ident):
    return sum(vecs[f"{at}/{v}"] for at in ATTRIBUTES for v in [ident[at]] if v is not None).float()


CFG = {  # profile, vector mode, coefficient, noise
    "direct": dict(profile=False, vec=None, noise=False),
    "pe": dict(profile=True, vec=None, noise=False),
    "pv": dict(profile=False, vec="last", noise=False, coeff=a.pv_coeff),
    "psii": dict(profile=True, vec="all", noise=True),
    "psii_nonoise": dict(profile=True, vec="all", noise=False),
    "psii_noprofile": dict(profile=False, vec="all", noise=True),
    "pe_noise": dict(profile=True, vec=None, noise=True),  # extra diagnostic: profile + noise, NO vectors
}


@torch.no_grad()
def option_logits(items, cfg, sigma, noise_seed):
    """items: list of (respondent_row, qid). Returns (N, 6) float logits over option tokens (-inf if unused)."""
    out = torch.full((len(items), 6), float("-inf"))
    inj.gen = torch.Generator(device=a.device); inj.gen.manual_seed(noise_seed)
    inj.mode = cfg["vec"] if cfg["vec"] == "last" else "all"
    inj.sigma = sigma if cfg["noise"] else 0.0
    with inj:
        for b in range(0, len(items), a.bs):
            chunk = items[b:b + a.bs]
            prompts = []
            for r, qid in chunk:
                sysm = DIRECT_SYS + ("\n" + profile_text(idents[r]) if cfg["profile"] else "")
                prompts.append(chat_prompt(tok, sysm, question_text(QBY[qid])))
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
            inj.mask = enc.attention_mask
            inj.vec = (torch.stack([respondent_vec(idents[r]) for r, _ in chunk]).to(a.device) * cfg.get("coeff", 1.0)
                       if cfg["vec"] else None)
            lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            for i, (_, qid) in enumerate(chunk):
                k = len(QBY[qid]["options"])
                out[b + i, :k] = lg[i, opt_ids[:k]].cpu()
    return out


def items_for(ids):
    return [(int(r), q["id"]) for r in ids for q in QUESTIONS if df.at[r, q["id"]] in q["options"]]


def sample_answers(logits, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(logits / TEMP, -1)
    return torch.multinomial(p, 1, generator=g)[:, 0].numpy()


if a.sensitivity:
    ids = sample_ids(0); items = items_for(ids)
    cfg = CFG["psii"]
    clean = option_logits(items, dict(cfg, noise=False), 0.0, 1)
    noisy = option_logits(items, cfg, a.sigma, 1)
    maes = []
    for c, n in zip(clean, noisy):
        k = int(torch.isfinite(c).sum())
        rc = torch.argsort(torch.argsort(-c[:k])).float(); rn = torch.argsort(torch.argsort(-n[:k])).float()
        maes.append(float((rc - rn).abs().mean() / k))
    s = float(np.mean(maes))
    res = {"probe_sigma": a.sigma, "sensitivity": s, "sigma_best": max(0.0, 0.4 - s), "n_pairs": len(items)}
    json.dump(res, open(f"{HERE}/results/sensitivity_{a.sigma}.json", "w"), indent=1); print(res)
    raise SystemExit

rows = []
for seed in a.seeds:
    ids = sample_ids(seed); items = items_for(ids)
    for m in a.methods:
        t = time.time()
        lg = option_logits(items, CFG[m], a.sigma, 1000 * seed + 7)
        ans = sample_answers(lg, 1000 * seed + 13)
        name = m + (f"_{a.tag}" if a.tag and CFG[m]["noise"] else "")
        d = pd.DataFrame({"method": name, "seed": seed, "resp": [r for r, _ in items], "qid": [q for _, q in items],
                          "answer": ans, "human": [QBY[q]["options"].index(df.at[r, q]) for r, q in items]})
        d.to_csv(f"{OUT}/answers_{name}_seed{seed}.csv", index=False)
        print(name, seed, len(d), f"{time.time()-t:.0f}s", flush=True)
