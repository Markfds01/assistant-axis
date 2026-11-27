"""Scale check: mean residual-stream norm at the prompt's final token vs. the norm of each d_k and of the summed vector.
Usage: python scale.py --spec llama --layers 16 18 20 22 24 26 [--device cuda:0]
Residual stream after decoder layer l = hidden_states[l+1]; vectors from the all-layer file are indexed the same way."""
import argparse, json
import numpy as np, torch
from followup import Runner, NAMED
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="llama"); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--layers", type=int, nargs="+", required=True); ap.add_argument("--n_items", type=int, default=400)
ap.add_argument("--sigmas", type=float, nargs="*", default=[0.07])
a = ap.parse_args()
R = Runner(a.spec, a.device)
S = SPECS[a.spec]
allv = torch.load(f"{OUT}/{S['all_layers']}")  # key -> (n_layers, H)
ids = R.sample_ids(0); items = R.items_for(ids)
rng = np.random.default_rng(0); sel = rng.choice(len(items), min(a.n_items, len(items)), replace=False)
items = [items[i] for i in sel]
H = R.model.config.hidden_size
res = {l: [] for l in a.layers}
with torch.no_grad():
    for b in range(0, len(items), 40):
        chunk = items[b:b + 40]
        enc = R.tok([R.prompt(r, q, True, "r1") for r, q in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(R.dev)
        hs = R.model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
        for l in a.layers:
            res[l] += hs[l + 1][:, -1].float().norm(dim=-1).cpu().tolist()
out = {"spec": a.spec, "hidden_size": H, "n_items": len(items), "layers": {}}
resp = sorted({r for r, _ in items})
rows = []
for l in a.layers:
    dn = np.array([float(v[l].norm()) for v in allv.values()])
    sums = np.array([float(sum(allv[f"{at}/{R.idents[r][at]}"][l] for at in ATTRIBUTES if R.idents[r][at] is not None).norm()) for r in resp])
    h = np.array(res[l])
    out["layers"][l] = dict(depth=round((l + 1) / S["n_layers"], 2), resid_mean=h.mean(), resid_sd=h.std(), dk_mean=dn.mean(), dk_min=dn.min(),
                            dk_max=dn.max(), dsum_mean=sums.mean(), dsum_sd=sums.std(), ratio_dsum_over_h=sums.mean() / h.mean(),
                            ratio_dk_over_h=dn.mean() / h.mean())
    rows.append(f"| {l} | {(l+1)/S['n_layers']:.0%} | {h.mean():.1f} ± {h.std():.1f} | {dn.mean():.2f} ({dn.min():.2f}–{dn.max():.2f}) | "
                f"{sums.mean():.2f} ± {sums.std():.2f} | {sums.mean()/h.mean():.1%} |")
noise = {s: s * H ** 0.5 for s in a.sigmas}
out["noise_norm"] = noise
json.dump(out, open(f"{HERE}/results/followup/scale_{a.spec}.json", "w"), indent=1)
md = [f"Scale check, {S['name']} (hidden size {H}); residual norm at the final prompt token over {len(items)} PE prompts (R1 prompts); "
      f"d_k over all {len(allv)} attribute values; sum over the 100 seed-0 respondents' attributes.", "",
      "| layer | depth | residual norm | norm d_k, mean (min–max) | norm Σd_k | Σd / residual |", "|---|---|---|---|---|---|"] + rows
md += ["", "Noise norm sigma*sqrt(H): " + ", ".join(f"σ={s}: {v:.1f}" for s, v in noise.items())]
open(f"{HERE}/results/followup/scale_{a.spec}.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
