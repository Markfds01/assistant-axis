"""Scale check: authors' vectors / noise vs the residual stream at their injection layers.

Norms of every per-layer demographic vector at the layers used in run.sh (16 / 21 / 26 and the 70% layer 22),
the norm of the injected noise (sigma * sqrt(4096) per hook, sqrt(n) of that for n hooks at one layer),
the value-vector norm (alpha * ||l_s||) and the residual-stream norm at those layers for the authors' PE prompts.
usage: scale_check.py OUT_JSON
"""
import glob
import json
import os
import sys

PSII = os.environ.get("PSII_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "psii_official"))
sys.path.insert(0, PSII)
os.chdir(PSII)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import run as R  # noqa: E402
import utils as U  # noqa: E402

out = {}
VD = "demographic_vectors/vectors_llama3.1-8b"
files = sorted(glob.glob(f"{VD}/Q*_*.pt"))
cfg = R.parse_vector_q_codes("271:16,285:16,263:21,289:21,260:26,273:26,275:26,279:26,281:26,284:26,286:26,287:26,288:26")
norms = {}
for q, l in cfg.items():
    ns = [float(torch.load(f, weights_only=False)[l].float().norm()) for f in files if os.path.basename(f).startswith(f"Q{q}_")]
    norms[f"Q{q}@{l}"] = dict(min=min(ns), mean=float(np.mean(ns)), max=max(ns), n_values=len(ns))
out["vector_norms_run_sh"] = norms
l22 = [float(torch.load(f, weights_only=False)[22].float().norm()) for f in files]
out["vector_norms_layer22_all_files"] = dict(min=min(l22), mean=float(np.mean(l22)), max=max(l22))
vn = {}
for lang in R.LANGS:
    v = torch.load(f"value_vectors/vectors_llama3.1-8b/language_embedding_{lang}.pt")
    vn[lang] = float(v.float().norm())
out["value_vector_norm"] = vn

model, tok, gc = U.get_model("llama3.1-8b", 5, 0.7)
model.eval()
agents = U.load_json("agent_profile/descriptions/wvs_demographic_descriptions_100.json")
qs = U.load_json("data/questions.json")
layers = [15, 20, 21, 25]  # hook index = layer - 1 (vector_q_codes layer 16, 21, 22, 26)
store = {l: [] for l in layers + [31]}
last = {l: [] for l in layers + [31]}


def mk(l):
    def hook(m, i, o):
        t = o[0] if isinstance(o, (tuple, list)) else o
        n = t[0].float().norm(dim=-1)
        store[l].append(float(n.mean()))   # mean over prompt tokens (includes BOS / sink token)
        last[l].append(float(n[-1]))       # last prompt token
    return hook


hs = [model.model.layers[l].register_forward_hook(mk(l)) for l in store]
import random  # noqa: E402
random.seed(0)
with torch.no_grad():
    for _ in range(60):
        aid = str(random.randint(1, 100))
        qid = random.choice(list(qs)[:259])
        q = qs[qid]
        role = R.system_prompt_map["en"] + "\n" + R.profile_prompt_map["en"] + "\n" + agents[aid]
        txt = "Question: " + q["question_text"] + "\nOptions:\n" + "\n".join(f"{k}: {v}" for k, v in sorted(q["options"].items(), key=lambda x: int(x[0]))) + R.number_prompt_map["en"]
        ids = U.get_prompt(tok, role.strip(), txt.strip())["input_ids"].to(model.device)
        model(input_ids=ids)
for h in hs:
    h.remove()
out["residual_norm"] = {f"layer{l + 1}": dict(mean_over_tokens=float(np.mean(store[l])), last_token=float(np.mean(last[l]))) for l in store}
sigma, coef, alpha = 0.07, 2.0, 0.1
out["noise_norm_per_hook"] = sigma * 64.0
out["coef"] = coef
out["alpha"] = alpha
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
