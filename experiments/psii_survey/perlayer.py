"""PSII with per-layer demographic vectors and the paper's layer groups, on the Spanish survey (Llama-3.1-8B-Instruct).

  python perlayer.py sens                                  # paper's sigma rule on the per-layer PSII config -> results/perlayer/sigma_perlayer_{layer,attr}.json (--noise)
  python perlayer.py run --arms pe pl_a1_psii_cal ... --seeds 0 1 2 --device cuda:0
  python perlayer.py run --list                            # arm registry

One pass per (arm, seed) yields BOTH readouts for the same noisy forward calls:
  G  (main) official-code style sampled answer: the model generates up to 5 tokens (T=0.7, top_k=20, top_p=0.8 as in the
     authors' generation_config), the first number is parsed (their regex), values outside 1..k count as unparsed and are
     re-asked up to 5 times (their max_attempts), then dropped. Steering is applied to every prompt token and to every
     generated token (fresh noise draw per forward call, as their hook does).
  R1 (secondary) option-digit logits at the last prompt position of the same prefill, answer = sample from softmax(logits/0.7).
Layer indices are hidden_states indices (the authors hook decoder layer index-1, so index h = output of layers[h-1] = hidden_states[h]).
Per-layer noise: the paper/official code draws one eps ~ N(0, sigma^2 I) per injected attribute vector, so a layer that receives
n attribute vectors receives n independent draws = N(0, n sigma^2 I) (implemented as sigma*sqrt(n), mode 'attr'; used for the numeric
sigma=0.07 arms). Calibrated arms ('cal') use ONE draw per injection layer (mode 'layer', as the single-layer arms of followup.py),
because the paper's calibration rule applied with per-attribute draws ('calattr') gives a much larger sigma per draw and a large noise norm
on the layer holding six vectors, and the model collapses.
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd, torch
from common import *
from followup import Runner, sample_answers, rank_mae
from intervention import LinearEdit, option_logits_from, _sample_topk

PL = f"{OUT}/perlayer"
RES = f"{HERE}/results/perlayer"
TOPP, MAXNEW, MAXATT = 0.8, 5, 5
# --- layer assignment (hidden_states index), from run.sh llama3.1-8b layers 16 / 21 / 26 and the paper's rule ---
LOWER, MID, UPPER = 16, 21, 26
LAYER_MAP = {"sex": UPPER, "age": UPPER, "education": UPPER, "income": UPPER, "marital_status": UPPER, "politics": UPPER, "religion": MID}
SINGLE_LAYER = 22  # reference arm: all attributes at decoder layer 22 (hook on layers[22]), earlier vectors (data/demographic_vectors_layer22.pt)
VALUE_ALPHA, VALUE_SIGMA = 0.1, 0.01  # activation_steer.py defaults for the value vector
VALUE_FILE = os.environ.get("PSII_VALUE_FILE", os.path.join(os.environ.get("PSII_DIR", os.path.join(HERE, "..", "..", "..", "psii_official")), "value_vectors", "vectors_llama3.1-8b", "language_embedding_es.pt"))

# arm registry: layers 'pl' (per-layer map) | 'age21' (age moved to intermediate) | 'single' (all at layer 22) | None
#               vec 'real' | 'rand' (random direction, norm of the summed per-layer vector) | 'rand_attr' (per-attribute norms) | None
#               sigma 0 | 0.07 | 'cal' ; vset 'A' (authors-format vectors) | 'B' (chat-format) | 'old' (data/demographic_vectors_layer22.pt)
ARMS = {"pe": dict(layers=None, vec=None, alpha=0, sigma=0)}
for tag, sg in (("s007", 0.07), ("cal", "cal")):
    ARMS[f"pl_pe_noise_{tag}"] = dict(layers="pl", vec=None, alpha=0, sigma=sg)
for a in (1, 2):
    for tag, sg in (("s007", 0.07), ("cal", "cal"), ("nonoise", 0)):
        ARMS[f"pl_a{a}_psii_{tag}"] = dict(layers="pl", vec="real", alpha=a, sigma=sg)
        ARMS[f"pl_a{a}_rand_{tag}"] = dict(layers="pl", vec="rand", alpha=a, sigma=sg)
ARMS["pl_a1_psii_calattr"] = dict(layers="pl", vec="real", alpha=1, sigma="calattr")
ARMS["pl_pe_noise_calattr"] = dict(layers="pl", vec=None, alpha=0, sigma="calattr")
ARMS["pl_a1_psii_cal_B"] = dict(layers="pl", vec="real", alpha=1, sigma="cal", vset="B")
ARMS["pl_a1_psii_cal_age21"] = dict(layers="age21", vec="real", alpha=1, sigma="cal")
ARMS["pl_a1_randattr_cal"] = dict(layers="pl", vec="rand_attr", alpha=1, sigma="cal")
ARMS["pe_value"] = dict(layers=None, vec=None, alpha=0, sigma=0, value=True)
ARMS["pl_a1_psii_cal_value"] = dict(layers="pl", vec="real", alpha=1, sigma="cal", value=True)
ARMS["pl_pe_noise_cal_value"] = dict(layers="pl", vec=None, alpha=0, sigma="cal", value=True)
for tag, sg in (("s007", 0.07), ("cal", "cal"), ("nonoise", 0)):
    ARMS[f"l22_psii_{tag}"] = dict(layers="single", vec="real", alpha=1, sigma=sg, vset="old")
ARMS["l22_pe_noise_cal"] = dict(layers="single", vec=None, alpha=0, sigma="cal")
ARMS["l22_pe_noise_s007"] = dict(layers="single", vec=None, alpha=0, sigma=0.07)
ARMS["l22_rand_cal"] = dict(layers="single", vec="rand", alpha=1, sigma="cal", vset="old")
ARMS["l22_rand_nonoise"] = dict(layers="single", vec="rand", alpha=1, sigma=0, vset="old")


def layer_assignment(kind):
    if kind == "age21":
        return {**LAYER_MAP, "age": MID}
    if kind == "single":
        return {at: SINGLE_LAYER + 1 for at in ATTRIBUTES}  # hs index 23 = output of layers[22]
    return LAYER_MAP


class MultiHook:
    """Forward hooks on decoder layers; edits[hs_index] = callable (hs, ctx) -> hs, applied to the output of layers[hs_index-1]."""

    def __init__(self, model):
        self.layers = model.model.layers; self.edits = {}; self.ctx = None; self.hs = []

    def _mk(self, i):
        def hook(mod, ins, out):
            e = self.edits.get(i)
            if e is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            for f in e:
                h = f(h, self.ctx)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    def __enter__(self):
        self.hs = [l.register_forward_hook(self._mk(i + 1)) for i, l in enumerate(self.layers)]; return self

    def __exit__(self, *a):
        for h in self.hs: h.remove()
        self.hs = []


class LastTokenEdit:
    """value vector: h[:, -1] += alpha * (v + sigma * eps) on every forward call (activation_steer.py `_hook_fn_ml`)."""

    def __init__(self, vec, alpha=VALUE_ALPHA, sigma=VALUE_SIGMA, seed=0):
        self.vec, self.alpha, self.sigma = vec, alpha, sigma; self.g = {}; self.seed = seed

    def __call__(self, hs, ctx):
        dev = hs.device
        if dev not in self.g: self.g[dev] = torch.Generator(device=dev).manual_seed(self.seed)
        B, T, H = hs.shape
        add = self.alpha * (self.vec.to(dev, torch.float32)[None] + self.sigma * torch.randn((B, H), device=dev, generator=self.g[dev]))
        hs = hs.clone(); hs[:, -1] += add.to(hs.dtype); return hs


def top_p_sample(logits, g, temp, k, p):
    """HF order: temperature -> top-k -> top-p -> multinomial."""
    v, idx = (logits.float() / temp).topk(k, -1)
    pr = torch.softmax(v, -1)
    cum = pr.cumsum(-1)
    keep = (cum - pr) < p
    pr = pr * keep
    pick = torch.multinomial(pr / pr.sum(-1, keepdim=True), 1, generator=g)
    return idx.gather(-1, pick)[:, 0]


NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_answer(text, k):
    m = NUM.search(text)
    if not m: return -1
    try: n = int(float(m.group()))
    except ValueError: return -1
    return n - 1 if 1 <= n <= k else -1


class PerLayer:
    def __init__(self, device):
        self.R = Runner("llama", device)
        self.old = self.R.vecs
        self.new = {}  # vset -> dict key -> (33,H)
        for vs, f in (("A", "authors"), ("B", "chat")):
            p = f"{PL}/demographic_vectors_perlayer_{f}.pt"
            if os.path.exists(p): self.new[vs] = torch.load(p)
        self.value_vec = torch.load(VALUE_FILE)[0].float()
        self.H = self.R.model.config.hidden_size
        self.mh = MultiHook(self.R.model)
        assert self.R.complete.sum() > 0
        self.sig = {m: json.load(open(f"{RES}/sigma_perlayer_{m}.json"))["sigma_best"] for m in ("attr", "layer") if os.path.exists(f"{RES}/sigma_perlayer_{m}.json")}
        self.sigma_single = json.load(open(f"{HERE}/results/followup/sigma_llama_r1.json"))["sigma_best"]

    def vec_for(self, vset, at, v, hs_idx):
        if vset == "old":
            return self.old[f"{at}/{v}"]  # layer-22 vector (hs[23]), the earlier recipe
        return self.new[vset][f"{at}/{v}"][hs_idx]

    def build_edits(self, arm, chunk, rng, noise_seed, sigma, mode):
        """-> dict hs_index -> list of callables"""
        edits = {}
        kind = arm["layers"]
        if kind is not None:
            amap = layer_assignment(kind)
            by_layer = {}
            for at, L in amap.items(): by_layer.setdefault(L, []).append(at)
            for L, ats in by_layer.items():
                n = len(ats)
                vec = None
                if arm["vec"] is not None:
                    per = torch.stack([torch.stack([self.vec_for(arm.get("vset", "A"), at, self.R.idents[r][at], L) for at in ats]) for r, _ in chunk]).float()  # (B,n,H)
                    if arm["vec"] == "real":
                        vec = per.sum(1)
                    elif arm["vec"] == "rand":
                        z = torch.randn(per.sum(1).shape, generator=rng); vec = z / z.norm(dim=-1, keepdim=True) * per.sum(1).norm(dim=-1, keepdim=True)
                    elif arm["vec"] == "rand_attr":
                        z = torch.randn(per.shape, generator=rng); vec = (z / z.norm(dim=-1, keepdim=True) * per.norm(dim=-1, keepdim=True)).sum(1)
                sg = sigma * (n ** 0.5 if mode == "attr" else 1.0)
                if vec is None and sg == 0: continue
                edits.setdefault(L, []).append(LinearEdit(vec, float(arm["alpha"]), sg, "all", "call", seed=noise_seed + 7919 * L))
        if arm.get("value"):
            edits.setdefault(32, []).append(LastTokenEdit(self.value_vec, seed=noise_seed + 31))
        return edits

    def sigma_of(self, arm):
        """-> (sigma, noise mode). 'attr': one draw per attribute vector (official code), a layer with n vectors gets sigma*sqrt(n);
        'layer': one draw per injection layer. Numeric sigma (0.07) = official semantics ('attr'); 'cal' = calibrated, one draw per layer
        (the convention of the single-layer runs); 'calattr' = calibrated with per-attribute draws (collapses the model)."""
        s = arm["sigma"]
        if arm["layers"] == "single":
            return float(self.sigma_single if s == "cal" else s), "layer"
        if s == "cal":
            assert "layer" in self.sig, "run `sens --noise layer` first"; return self.sig["layer"], "layer"
        if s == "calattr":
            assert "attr" in self.sig, "run `sens --noise attr` first"; return self.sig["attr"], "attr"
        return float(s), "attr"

    @torch.no_grad()
    def gen_pass(self, items, arm, sigma, noise_seed, gen_seed, bs, mode="attr"):
        """-> parsed idx (N,), texts, r1 logits (N,6) or None, option mass"""
        R = self.R; N = len(items)
        parsed = np.full(N, -1); texts = [""] * N
        lg1 = torch.full((N, 6), float("-inf")); mass = torch.zeros(N)
        rng = torch.Generator().manual_seed(noise_seed + 99)
        stop = torch.tensor(R.stop_ids, device=R.dev)
        with self.mh:
            for b in range(0, N, bs):
                chunk = items[b:b + bs]
                prompts = [R.prompt(r, qid, True, "r1") for r, qid in chunk]
                enc = R.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(R.dev)
                self.mh.edits = self.build_edits(arm, chunk, rng, noise_seed * 1000003 + b, sigma, mode)
                B = len(chunk)
                am = enc.attention_mask.clone(); pos = (am.cumsum(1) - 1).clamp(min=0)
                self.mh.ctx = dict(mask=am.float(), prefill=True)
                out = model_fwd(R.model, enc.input_ids, am, pos, None, True)
                past, logits = out.past_key_values, out.logits[:, -1]
                l1, ms = option_logits_from(logits, R.opt_ids)
                for i, (_, qid) in enumerate(chunk):
                    k = len(QBY[qid]["options"]); lg1[b + i, :k] = l1[i, :k]; mass[b + i] = ms[i]
                g = torch.Generator(device=logits.device).manual_seed(gen_seed * 7919 + b)
                finished = torch.zeros(B, dtype=torch.bool, device=logits.device); toks = []; keeps = []
                for step in range(MAXNEW):
                    nxt = top_p_sample(logits, g, TEMP, TOPK, TOPP)
                    is_stop = torch.isin(nxt, stop); keep = ~finished & ~is_stop
                    finished = finished | is_stop; toks.append(nxt); keeps.append(keep)
                    if step == MAXNEW - 1 or bool(finished.all()): break
                    am = torch.cat([am, keep.to(am.device, am.dtype)[:, None]], 1)
                    pos = (am.cumsum(1) - 1).clamp(min=0)[:, -1:]
                    self.mh.ctx = dict(mask=torch.ones(B, 1, device=R.dev), prefill=False)
                    out = model_fwd(R.model, nxt.to(R.dev)[:, None], am, pos, past, True)
                    past, logits = out.past_key_values, out.logits[:, -1]
                T = torch.stack(toks, 1).cpu(); K = torch.stack(keeps, 1).cpu()
                for i, (_, qid) in enumerate(chunk):
                    t = R.tok.decode(T[i][K[i]], skip_special_tokens=True).strip()
                    texts[b + i] = t; parsed[b + i] = parse_answer(t, len(QBY[qid]["options"]))
        self.mh.edits = {}
        return parsed, texts, lg1, mass


def model_fwd(model, ids, am, pos, past, cache):
    return model(input_ids=ids, attention_mask=am, position_ids=pos, past_key_values=past, use_cache=cache, logits_to_keep=1)


def run_arm(P, name, arm, seed, bs, n=100):
    R = P.R
    ids = R.sample_ids(seed, n); items = R.items_for(ids)
    sigma, mode = P.sigma_of(arm)
    t = time.time()
    noise_seed = 1000 * seed + 7
    parsed, texts, lg1, mass = P.gen_pass(items, arm, sigma, noise_seed, 1000 * seed + 21, bs, mode)
    attempts = np.ones(len(items), int)
    for att in range(1, MAXATT):  # the official code re-asks when the answer cannot be parsed
        bad = np.where(parsed < 0)[0]
        if len(bad) == 0: break
        sub = [items[i] for i in bad]
        p2, t2, _, _ = P.gen_pass(sub, arm, sigma, noise_seed + 100003 * att, 1000 * seed + 21 + 17 * att, bs, mode)
        for j, i in enumerate(bad):
            parsed[i] = p2[j]; texts[i] = t2[j]; attempts[i] += 1
    r1 = sample_answers(lg1, 1000 * seed + 13)
    d = pd.DataFrame({"method": name, "seed": seed, "resp": [r for r, _ in items], "qid": [q for _, q in items],
                      "ans_g": parsed, "ans_r1": r1, "human": [QBY[q]["options"].index(R.df.at[r, q]) for r, q in items],
                      "opt_mass": mass.numpy(), "attempts": attempts, "text": [json.dumps(t, ensure_ascii=True) for t in texts]})
    return d, lg1, time.time() - t, sigma


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["sens", "run"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--arms", nargs="*", default=[]); ap.add_argument("--list", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--probe", type=float, default=0.07); ap.add_argument("--noise", default="layer", choices=["attr", "layer"])
    a = ap.parse_args()
    if a.list:
        for k, v in ARMS.items(): print(k, v)
        return
    os.makedirs(f"{PL}/runs", exist_ok=True); os.makedirs(RES, exist_ok=True)
    P = PerLayer(a.device)
    if a.mode == "sens":
        arm = ARMS["pl_a1_psii_nonoise"]
        ids = P.R.sample_ids(0, a.n); items = P.R.items_for(ids)
        cal = dict(arm, sigma=a.probe)
        # same sampling seeds for the clean and the noisy pass; only the option logits at the prefill are used (R1-style, as in followup sens)
        _, _, clean, _ = P.gen_pass(items, arm, 0.0, 1, 5, a.bs, a.noise)
        _, _, noisy, _ = P.gen_pass(items, cal, a.probe, 1, 5, a.bs, a.noise)
        s = rank_mae(clean, noisy)
        res = {"config": f"per-layer PSII alpha=1 (vectors A), noise draws per {a.noise}", "probe_sigma": a.probe, "sensitivity": s,
               "sigma_best": max(0.0, 0.4 - s), "n_pairs": len(items)}
        json.dump(res, open(f"{RES}/sigma_perlayer_{a.noise}.json", "w"), indent=1); print(res); return
    for name in a.arms:  # arms outer: a finished arm has all its seeds
        for seed in a.seeds:
            arm = ARMS[name]; f = f"{PL}/runs/answers_{name}_seed{seed}.csv"
            if os.path.exists(f): print("skip", f); continue
            d, lg, secs, sg = run_arm(P, name, arm, seed, a.bs, a.n)
            np.save(f"{PL}/runs/logits_{name}_seed{seed}.npy", lg.numpy()); d.to_csv(f, index=False)
            print(name, seed, f"sigma={sg:.3f}", len(d), f"unparsed={float((d.ans_g < 0).mean()):.3f} retries={float((d.attempts > 1).mean()):.3f} mass={float(d.opt_mass.mean()):.3f} {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
