"""Scale check of the per-layer vectors vs the residual stream and the noise at every injection layer.
Residual = hidden_states[h] (= output of decoder layer h-1) of the PE prompts (400 random seed-0 items), norm at the last prompt token and
mean over prompt tokens (the real, non-pad tokens). Vectors: authors-format per-layer vectors at the assigned layers; the
single-layer-22 reference uses the earlier vectors (hidden_states index 23).
Noise: sigma*sqrt(H) per draw; a layer receiving n attribute vectors gets n independent draws -> sigma*sqrt(n*H).
Usage: python scale_perlayer.py [--device cuda:0] -> results/perlayer/scale.{json,md}"""
import argparse, json
import numpy as np, torch
from common import *
from followup import Runner
from perlayer import LAYER_MAP, SINGLE_LAYER, MID, UPPER

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n_items", type=int, default=400)
a = ap.parse_args()
R = Runner("llama", a.device); H = R.model.config.hidden_size
vecs = torch.load(f"{OUT}/perlayer/demographic_vectors_perlayer_authors.pt")
sg_pl = json.load(open(f"{HERE}/results/perlayer/sigma_perlayer_layer.json"))["sigma_best"]
sg_attr = json.load(open(f"{HERE}/results/perlayer/sigma_perlayer_attr.json"))["sigma_best"]
sg_22 = json.load(open(f"{HERE}/results/followup/sigma_llama_r1.json"))["sigma_best"]
ids = R.sample_ids(0); items = R.items_for(ids)
sel = np.random.default_rng(0).choice(len(items), min(a.n_items, len(items)), replace=False); items = [items[i] for i in sel]
LAY = sorted({16, MID, UPPER, SINGLE_LAYER + 1})
last = {h: [] for h in LAY}; mean = {h: [] for h in LAY}
with torch.no_grad():
    for b in range(0, len(items), 40):
        chunk = items[b:b + 40]
        enc = R.tok([R.prompt(r, q, True, "r1") for r, q in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(R.dev)
        hs = R.model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
        m = enc.attention_mask.float()
        for h in LAY:
            x = hs[h].float()
            last[h] += x[:, -1].norm(dim=-1).cpu().tolist()
            mean[h] += ((x.norm(dim=-1) * m).sum(1) / m.sum(1)).cpu().tolist()
resp = sorted({r for r, _ in items})
out = {"hidden_size": H, "n_items": len(items), "sigma_cal_perlayer": sg_pl, "sigma_cal_single": sg_22, "layers": {}}
rows = []
for h in LAY:
    ats = [at for at in ATTRIBUTES if LAYER_MAP[at] == h]
    single = h == SINGLE_LAYER + 1
    if single:
        ats = list(ATTRIBUTES)
        dk = np.array([float(R.vecs[f"{at}/{v}"].norm()) for at in ATTRIBUTES for v in VALUES[at]])
        sums = np.array([float(sum(R.vecs[f"{at}/{R.idents[r][at]}"] for at in ATTRIBUTES).norm()) for r in resp if all(R.idents[r][at] for at in ATTRIBUTES)])
    elif ats:
        dk = np.array([float(vecs[f"{at}/{v}"][h].norm()) for at in ats for v in VALUES[at]])
        sums = np.array([float(sum(vecs[f"{at}/{R.idents[r][at]}"][h] for at in ats).norm()) for r in resp if all(R.idents[r][at] for at in ats)])
    else:
        dk = sums = None
    n = len(ats) if not single else 1  # single-layer reference: one draw
    n_eff = n if ats else 0
    rl, rm = np.mean(last[h]), np.mean(mean[h])
    d = dict(attrs=ats, n_noise_draws=n_eff, resid_last=rl, resid_mean=rm)
    if dk is not None:
        d.update(dk_mean=dk.mean(), dk_min=dk.min(), dk_max=dk.max(), dsum_mean=sums.mean(), dsum_sd=sums.std())
    nn = 1 if single else max(n_eff, 1)
    d["noise_0.07"] = 0.07 * nn ** 0.5 * H ** 0.5                      # numeric sigma: one draw per attribute vector
    d["noise_cal"] = (sg_22 if single else sg_pl) * H ** 0.5            # calibrated arms: one draw per layer
    d["noise_calattr"] = sg_attr * nn ** 0.5 * H ** 0.5                 # calibrated sigma with per-attribute draws (collapsed arm)
    out["layers"][h] = d
    rows.append(f"| {h}{' (single-layer ref.)' if single else ''} | {','.join(ats) if ats else '(none: no attribute assigned)'} | {rl:.1f} / {rm:.1f} | "
                + (f"{dk.mean():.2f} ({dk.min():.2f}–{dk.max():.2f}) | {sums.mean():.2f} ± {sums.std():.2f} | {sums.mean()/rl:.0%} | " if dk is not None else "– | – | – | ")
                + (f"{d['noise_0.07']:.1f} ({d['noise_0.07']/rl:.0%}) | {d['noise_cal']:.1f} ({d['noise_cal']/rl:.0%}) | {d['noise_calattr']:.1f} ({d['noise_calattr']/rl:.0%})" if n_eff else "– | – | –"))
json.dump(out, open(f"{HERE}/results/perlayer/scale.json", "w"), indent=1, default=float)
md = [f"Residual stream = hidden_states[h] at the injection layer h; norms over {len(items)} PE prompts (R1 format, 100 seed-0 respondents), hidden size {H}. "
      f"Norm of the summed vector (Σ_k d_k at that layer, α=1; ×2 for the official coefficient) and of the noise, in brackets noise norm / residual norm at the last token: "
      f"σ=0.07 with one draw per attribute vector (σ√(nH), n = vectors at the layer); σ_cal one draw per layer (σ_cal = {sg_pl:.3f} per-layer config, {sg_22:.3f} single-layer ref.); σ_cal = {sg_attr:.3f} with per-attribute draws (the collapsed 'calattr' arm).", "",
      "| h | attributes injected | residual norm: last token / mean over tokens | norm d_k, mean (min–max) | norm Σd_k | Σd / residual (last) | noise σ=0.07 (per-attr draws) | noise σ_cal (1 draw/layer) | noise σ_cal per-attr draws |",
      "|---|---|---|---|---|---|---|---|---|"] + rows
open(f"{HERE}/results/perlayer/scale.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
