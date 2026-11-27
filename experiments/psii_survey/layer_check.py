"""Per-attribute layer check: how much does each attribute's value vector move the model's answer to a question about that
very attribute ("What is your <attribute>?", options = the attribute's values), as a function of the injection layer?
No profile, no noise. For every (attribute, value v) the vector d_v of that value (all-layer file, authors-format) times alpha is
added to every prompt token at hidden_states index h; metric = p(v) under softmax(option logits) minus p(v) without a vector,
averaged over the attribute's values (chance = 1/k); random control = random direction with the same norm.
Usage: python layer_check.py [--device cuda:0] -> results/perlayer/layer_check.{json,md}"""
import argparse, json
import numpy as np, torch
from common import *
from followup import Runner
from intervention import LinearEdit, option_logits_from
from perlayer import MultiHook, LAYER_MAP

OWN_Q = {"sex": "What is your sex?", "age": "How old are you?", "education": "What is your highest level of education?",
         "income": "What is the monthly income of your household?", "religion": "What is your religion?",
         "marital_status": "What is your marital status?", "politics": "Which Spanish political party do you feel closest to?"}
GRID = [4, 8, 12, 14, 16, 18, 20, 21, 22, 24, 26, 28, 30, 32]

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
a = ap.parse_args()
R = Runner("llama", a.device); mh = MultiHook(R.model)
vecs = torch.load(f"{OUT}/perlayer/demographic_vectors_perlayer_authors.pt")


def prompt(at):
    opts = "\n".join(f"{i+1}: {VALUE_DESC[at][v]}" for i, v in enumerate(VALUES[at]))
    return chat_prompt(R.tok, DIRECT_SYS, f"Question: {OWN_Q[at]}\nOptions:\n{opts}{NUM_SUFFIX}")


@torch.no_grad()
def probs(at, h, alpha, kind, seed=0):
    vs = VALUES[at]; k = len(vs)
    enc = R.tok([prompt(at)] * k, return_tensors="pt", padding=True, add_special_tokens=False).to(R.dev)
    if h is None or alpha == 0:
        mh.edits = {}
    else:
        V = torch.stack([vecs[f"{at}/{v}"][h] for v in vs]).float()
        if kind == "rand":
            z = torch.randn(V.shape, generator=torch.Generator().manual_seed(seed)); V = z / z.norm(dim=-1, keepdim=True) * V.norm(dim=-1, keepdim=True)
        mh.edits = {h: [LinearEdit(V, alpha, 0.0, "all", "call")]}
    mh.ctx = dict(mask=enc.attention_mask.float(), prefill=True)
    lg = R.model(**enc, logits_to_keep=1).logits[:, -1]
    ids = R.opt_ids[:k]
    p = torch.softmax(lg.float()[:, ids], -1).cpu()  # (k values, k options); row i = vector of value i
    return p


res = {}
with mh:
    for at in ATTRIBUTES:
        k = len(VALUES[at]); base = probs(at, None, 0, "real")
        b_p = float(base.diag().mean()); b_acc = float((base.argmax(1) == torch.arange(k)).float().mean())
        res[at] = {"k": k, "baseline_p": b_p, "baseline_acc": b_acc, "alpha": {}}
        for al in a.alphas:
            r = {}
            for h in GRID:
                pr = probs(at, h, al, "real"); pn = np.mean([float(probs(at, h, al, "rand", s).diag().mean()) for s in range(3)])
                r[h] = dict(dp=float(pr.diag().mean()) - b_p, dp_rand=pn - b_p, acc=float((pr.argmax(1) == torch.arange(k)).float().mean()))
            res[at]["alpha"][al] = r
json.dump(res, open(f"{HERE}/results/perlayer/layer_check.json", "w"), indent=1)
L = []
for al in a.alphas:
    L += [f"### alpha = {al:g}: Δp(own value) = p(v | +α·d_v at hidden_states index h) − p(v | no vector), mean over the attribute's values; (random direction, same norm) in brackets",
          "", "| attribute (k, baseline p) | assigned | " + " | ".join(str(h) for h in GRID) + " |", "|---|---|" + "---|" * len(GRID)]
    for at in ATTRIBUTES:
        r = res[at]["alpha"][al]
        L.append(f"| {at} ({res[at]['k']}, {res[at]['baseline_p']:.2f}) | {LAYER_MAP[at]} | " + " | ".join(f"{r[h]['dp']:+.2f} ({r[h]['dp_rand']:+.2f})" for h in GRID) + " |")
    L.append("")
open(f"{HERE}/results/perlayer/layer_check.md", "w").write("\n".join(L)); print("\n".join(L))
