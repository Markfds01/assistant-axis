"""Follow-up runner: every method through ONE intervention hook (intervention.py), two readouts (R1, R3).

  python followup.py sens   --spec llama --readout r1|r3 [--probe 0.07]       # paper's sensitivity rule -> sigma_best
  python followup.py run    --spec llama --readout r1|r3 --configs direct pe pv psii psii_nonoise pe_noise --seeds 0 1 2
  python followup.py run    ... --configs lin:a=4:s=0.07 lin:a=8:s=cal:np    # sweep configs

Config grammar: a named method (direct pe pv psii psii_nonoise pe_noise) or
  lin:a=<alpha>:s=<sigma|cal>[:np][:rand]   = profile prompt + h + alpha*sum(d_k) + sigma*eps
  (:np = no profile sentence, :rand = control: random direction with the same norm as sum(d_k) instead of the vectors).
Per-respondent outputs go to OUT/followup/<spec>/<readout>/ (never committed).
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd, torch
from common import *
from intervention import LinearEdit, Hook, readout_r1, readout_r3, SUFFIX

R3_TAIL = ("\nFirst write 1-2 short sentences in the first person, in Spanish, reasoning about this question as the "
           "person you are. Do NOT state your final answer number yet.")
PV_COEFF = 2.0
NAMED = {  # profile, vec (None | "sum"), alpha, positions, noise
    "direct": dict(profile=False, vec=None, alpha=0.0, positions="all", noise=False),
    "pe": dict(profile=True, vec=None, alpha=0.0, positions="all", noise=False),
    "pv": dict(profile=False, vec="sum", alpha=PV_COEFF, positions="last", noise=False),
    "psii": dict(profile=True, vec="sum", alpha=1.0, positions="all", noise=True),
    "psii_nonoise": dict(profile=True, vec="sum", alpha=1.0, positions="all", noise=False),
    "pe_noise": dict(profile=True, vec=None, alpha=0.0, positions="all", noise=True),
}


def parse_config(s, sigma_cal):
    """-> (name, cfg dict, sigma)"""
    if s in NAMED:
        return s, dict(NAMED[s]), (sigma_cal if NAMED[s]["noise"] else 0.0)
    parts = s.split(":"); assert parts[0] == "lin", s
    kv = dict(p.split("=") for p in parts[1:] if "=" in p); flags = set(p for p in parts[1:] if "=" not in p)
    a = float(kv["a"]); sg = sigma_cal if kv["s"] == "cal" else float(kv["s"])
    cfg = dict(profile="np" not in flags, vec="rand" if "rand" in flags else "sum", alpha=a, positions="all", noise=sg > 0)
    name = f"lin_a{kv['a']}_s{'cal' if kv['s'] == 'cal' else kv['s']}" + ("_np" if "np" in flags else "") + ("_rand" if "rand" in flags else "")
    if a == 0 and cfg["vec"] == "sum":
        cfg["vec"] = None
    if kv["s"] in ("cal", "0", "0.0"):  # same intervention as a named method -> reuse its name (and its outputs)
        for nm in ("pe", "pe_noise", "psii", "psii_nonoise"):
            nc = NAMED[nm]
            if all(nc[k] == cfg[k] for k in ("profile", "vec", "alpha", "positions")) and nc["noise"] == (sg > 0):
                return nm, cfg, sg
    return name, cfg, sg


class Runner:
    def __init__(self, spec, device, bs=None, max_new=60, noise="call"):
        self.spec, self.S = spec, SPECS[spec]
        self.device, self.max_new, self.noise = device, max_new, noise
        self.df = load_survey()
        self.idents = [respondent_identity(r) for _, r in self.df[ATTRIBUTES].iterrows()]
        self.complete = np.array([all(v is not None for v in i.values()) for i in self.idents])
        self.vecs = torch.load(f"{HERE}/data/{self.S['vec_file']}")
        self.model, self.tok = load_model(device, spec)
        self.opt_ids = option_token_ids(self.tok)
        self.suffix_ids = self.tok.encode(SUFFIX, add_special_tokens=False)
        self.stop_ids = sorted({self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|eot_id|>" if spec == "llama" else "<|im_end|>"),
                                self.tok.convert_tokens_to_ids("<|end_of_text|>" if spec == "llama" else "<|endoftext|>")} - {None})
        self.dev = self.model.get_input_embeddings().weight.device
        self.hook = Hook(self.model, self.S["layer"])
        self.outdir = f"{OUT}/followup/{spec}"
        os.makedirs(self.outdir, exist_ok=True)

    # --- data ---
    def sample_ids(self, seed, n=100):
        return np.random.default_rng(seed).choice(np.where(self.complete)[0], n, replace=False)

    def items_for(self, ids):
        return [(int(r), q["id"]) for r in ids for q in QUESTIONS if self.df.at[r, q["id"]] in q["options"]]

    def respondent_vec(self, ident):
        return sum(self.vecs[f"{at}/{v}"] for at in ATTRIBUTES for v in [ident[at]] if v is not None).float()

    def prompt(self, r, qid, profile, readout):
        sysm = DIRECT_SYS + ("\n" + profile_text(self.idents[r]) if profile else "")
        q = QBY[qid]
        if readout == "r1":
            user = question_text(q)
        else:
            opts = "\n".join(f"{i+1}: {o}" for i, o in enumerate(q["options"]))
            user = f"Question: {q['text']}\nOptions:\n{opts}{R3_TAIL}"
        return chat_prompt(self.tok, sysm, user)

    # --- core ---
    @torch.no_grad()
    def logits(self, items, cfg, sigma, noise_seed, readout, bs, gen_seed=0):
        """items: [(resp, qid)]. -> logits (N,6) with -inf for unused options, option mass (N,), rationales (N,) or None"""
        out = torch.full((len(items), 6), float("-inf")); mass = torch.zeros(len(items)); texts = [""] * len(items)
        rng = torch.Generator().manual_seed(noise_seed + 99)
        with self.hook:
            for b in range(0, len(items), bs):
                chunk = items[b:b + bs]
                prompts = [self.prompt(r, qid, cfg["profile"], readout) for r, qid in chunk]
                enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
                vec = None
                if cfg["vec"] == "sum":
                    vec = torch.stack([self.respondent_vec(self.idents[r]) for r, _ in chunk])
                elif cfg["vec"] == "rand":  # control: random direction, same norm as the respondent's summed vector
                    v = torch.stack([self.respondent_vec(self.idents[r]) for r, _ in chunk])
                    z = torch.randn(v.shape, generator=rng); vec = z / z.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)
                edit = LinearEdit(vec, cfg["alpha"], sigma if cfg["noise"] else 0.0, cfg["positions"], self.noise,
                                  seed=noise_seed * 1000003 + b)
                self.hook.edit = edit if (vec is not None or edit.sigma > 0) else None
                if readout == "r1":
                    lg, ms = readout_r1(self.model, self.hook, enc, self.opt_ids); tx = [""] * len(chunk)
                else:
                    lg, ms, tx = readout_r3(self.model, self.tok, self.hook, enc, self.opt_ids, self.suffix_ids, self.stop_ids,
                                            self.max_new, TEMP, TOPK, seed=gen_seed * 7919 + b)
                for i, (_, qid) in enumerate(chunk):
                    k = len(QBY[qid]["options"])
                    out[b + i, :k] = lg[i, :k]; mass[b + i] = ms[i]; texts[b + i] = tx[i]
        self.hook.edit = None
        return out, mass, texts


def sample_answers(logits, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.multinomial(torch.softmax(logits / TEMP, -1), 1, generator=g)[:, 0].numpy()


def rank_mae(clean, noisy):
    """Paper's sensitivity: mean over items of the mean abs rank change of the option logits / k."""
    maes = []
    for c, n in zip(clean, noisy):
        k = int(torch.isfinite(c).sum())
        rc = torch.argsort(torch.argsort(-c[:k])).float(); rn = torch.argsort(torch.argsort(-n[:k])).float()
        maes.append(float((rc - rn).abs().mean() / k))
    return float(np.mean(maes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sens", "run"]); ap.add_argument("--spec", default="llama")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--readout", default="r1", choices=["r1", "r3"])
    ap.add_argument("--configs", nargs="+", default=[]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--probe", type=float, default=0.07); ap.add_argument("--max_new", type=int, default=60)
    ap.add_argument("--noise", default="call", choices=["call", "token"])
    ap.add_argument("--sigma_file", default="", help="json with sigma_best (default results/followup/sigma_<spec>_<readout>.json)")
    ap.add_argument("--gen_offset", type=int, default=0, help="shift of the rationale/noise RNG seeds (noise-floor reruns)")
    ap.add_argument("--tag", default="", help="suffix appended to the output name (use with --gen_offset)")
    a = ap.parse_args()
    R = Runner(a.spec, a.device, max_new=a.max_new, noise=a.noise)
    bs = a.bs or (48 if a.readout == "r1" else 96)
    rdir = f"{R.outdir}/{a.readout}" + ("" if a.noise == "call" else "_tokennoise"); os.makedirs(rdir, exist_ok=True)
    os.makedirs(f"{HERE}/results/followup", exist_ok=True)

    if a.mode == "sens":
        ids = R.sample_ids(0, a.n); items = R.items_for(ids)
        cfg = NAMED["psii"]
        t = time.time()
        clean, _, tx0 = R.logits(items, dict(cfg, noise=False), 0.0, 1, a.readout, bs, gen_seed=5)
        noisy, _, tx1 = R.logits(items, cfg, a.probe, 1, a.readout, bs, gen_seed=5)
        s = rank_mae(clean, noisy)
        same = float(np.mean([x == y for x, y in zip(tx0, tx1)])) if a.readout == "r3" else None
        res = {"spec": a.spec, "readout": a.readout, "noise": a.noise, "probe_sigma": a.probe, "sensitivity": s,
               "sigma_best": max(0.0, 0.4 - s), "n_pairs": len(items), "share_identical_rationales": same, "secs": round(time.time() - t)}
        json.dump(res, open(f"{HERE}/results/followup/sigma_{a.spec}_{a.readout}" + ("" if a.noise == "call" else "_token") + ".json", "w"), indent=1)
        print(res); return

    sf = a.sigma_file or f"{HERE}/results/followup/sigma_{a.spec}_{a.readout}.json"
    sigma_cal = json.load(open(sf))["sigma_best"] if os.path.exists(sf) else None
    for seed in a.seeds:
        ids = R.sample_ids(seed, a.n); items = R.items_for(ids)
        for c in a.configs:
            if "s=cal" in c or (c in NAMED and NAMED[c]["noise"]):
                assert sigma_cal is not None, f"calibrated sigma needed: run `sens` first ({sf})"
            name, cfg, sg = parse_config(c, sigma_cal)
            name += a.tag
            f = f"{rdir}/answers_{name}_seed{seed}.csv"
            if os.path.exists(f):
                print("skip", f); continue
            t = time.time()
            lg, mass, tx = R.logits(items, cfg, sg, 1000 * seed + 7 + a.gen_offset, a.readout, bs, gen_seed=1000 * seed + 21 + a.gen_offset)
            ans = sample_answers(lg, 1000 * seed + 13)
            d = pd.DataFrame({"method": name, "seed": seed, "resp": [r for r, _ in items], "qid": [q for _, q in items],
                              "answer": ans, "human": [QBY[q]["options"].index(R.df.at[r, q]) for r, q in items],
                              "opt_mass": mass.numpy()})
            if a.readout == "r3":
                d["rationale"] = tx
            np.save(f"{rdir}/logits_{name}_seed{seed}.npy", lg.numpy())
            d.to_csv(f, index=False)
            print(name, seed, a.readout, f"sigma={sg:.3f}", len(d), f"mass={float(mass.mean()):.3f}", f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
