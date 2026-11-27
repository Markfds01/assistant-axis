"""One intervention interface + two readouts, shared by every method.

Hook: a forward hook on decoder layer L calls `edit(hs, ctx)` on the layer output (the residual stream after layer L).
  ctx = dict(mask=(B,T) 0/1 float tensor of real (non-pad) tokens of THIS forward call, prefill=bool)
Edits are plain callables hs -> hs, so a manifold edit (project / move h onto a curve) is just another class with the
same signature; nothing else changes. Methods are *configs* of the same edit:
  Direct / PE : no edit (they differ only in the prompt, a profile sentence)
  linear steering : h + alpha * sum_k d_k                    (LinearEdit(vec, alpha))
  PSII            : h + sum_k d_k + eps                       (LinearEdit(vec, 1.0, sigma))
  PE + noise      : h + eps                                   (LinearEdit(None, 0, sigma))
  PV              : h + coeff * sum_k d_k, only on the last prompt token and on every later (generated) token

Noise: `noise="call"` draws ONE Gaussian vector per sequence per forward call (as the official PSII code does: one draw
per call, broadcast over the tokens of that call, so with a KV cache the prompt gets one draw and every generated token
a fresh one). `noise="token"` is an independent draw per token (the first run).

Readouts (identical aggregation downstream):
  R1  one forward over the prompt, option-number logits at the last position.
  R3  think, then answer: sample <= max_new tokens of rationale (T=0.7, top_k=20, with KV cache; the edit is applied
      to every prompt token and to every generated token), then force a Spanish answer cue and read the option-number
      logits at its last position (edit applied there too, the cue is one extra forward call = one noise draw).
"""
import torch

SUFFIX = "\nRespuesta: "  # forced cue; the next token is then the option digit ("1".."6")


class LinearEdit:
    """h' = h + alpha * vec + sigma * eps on the selected positions.
    positions: "all"   -> every real token of every call
               "last"  -> only the last prompt token at prefill, then every token (PV: response-level)"""

    def __init__(self, vec=None, alpha=1.0, sigma=0.0, positions="all", noise="call", seed=0):
        self.vec, self.alpha, self.sigma, self.positions, self.noise, self.seed = vec, alpha, sigma, positions, noise, seed
        self._gens = {}

    def _gen(self, dev):
        if dev not in self._gens:
            self._gens[dev] = torch.Generator(device=dev).manual_seed(self.seed)
        return self._gens[dev]

    def __call__(self, hs, ctx):
        if self.vec is None and self.sigma == 0:
            return hs
        B, T, H = hs.shape
        add = torch.zeros(B, 1, H, device=hs.device, dtype=torch.float32)
        if self.vec is not None and self.alpha != 0:
            add = add + self.alpha * self.vec.to(hs.device, torch.float32)[:, None, :]
        if self.sigma > 0:
            shape = (B, 1, H) if self.noise == "call" else (B, T, H)
            add = add + torch.randn(shape, device=hs.device, dtype=torch.float32, generator=self._gen(hs.device)) * self.sigma
        m = ctx["mask"].to(hs.device)
        if self.positions == "last" and ctx["prefill"]:
            m = torch.zeros_like(m); m[:, -1] = 1
        return hs + (add * m[:, :, None]).to(hs.dtype)


class Hook:
    """Context manager registering the edit on model.model.layers[layer]; the runner sets .edit and .ctx before each call."""

    def __init__(self, model, layer):
        self.layer = model.model.layers[layer]; self.edit = None; self.ctx = None; self.h = None

    def _hook(self, mod, ins, out):
        if self.edit is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = self.edit(hs, self.ctx)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def __enter__(self): self.h = self.layer.register_forward_hook(self._hook); return self
    def __exit__(self, *a): self.h.remove()


def option_logits_from(logits, opt_ids):
    """(B,V) -> (B,len(opt_ids)) option-token logits, plus the probability mass the full softmax puts on them."""
    lg = logits.float()
    mass = torch.softmax(lg, -1)[:, opt_ids].sum(-1)
    return lg[:, opt_ids].cpu(), mass.cpu()


@torch.no_grad()
def readout_r1(model, hook, enc, opt_ids):
    hook.ctx = dict(mask=enc.attention_mask.float(), prefill=True)
    lg = model(**enc, logits_to_keep=1).logits[:, -1]
    return option_logits_from(lg, opt_ids)


def _sample_topk(logits, g, temp, k):
    """Gumbel-max over the top-k logits at temperature temp; the uniform stream is drawn for every row at every step, so
    two runs with the same seed and batch only diverge where the edit changes the argmax (used for sensitivity)."""
    v, idx = logits.float().topk(k, -1)
    u = torch.rand(v.shape, generator=g, device=v.device).clamp_min(1e-20)
    return idx.gather(-1, (v / temp - torch.log(-torch.log(u))).argmax(-1, keepdim=True))[:, 0]


@torch.no_grad()
def readout_r3(model, tok, hook, enc, opt_ids, suffix_ids, stop_ids, max_new=60, temp=0.7, topk=20, seed=0):
    """Returns (option logits (B,K) cpu, option mass (B,), rationale strings)."""
    dev = enc.input_ids.device
    B = enc.input_ids.shape[0]
    am = enc.attention_mask.clone()
    pos = (am.cumsum(1) - 1).clamp(min=0)
    hook.ctx = dict(mask=am.float(), prefill=True)
    out = model(input_ids=enc.input_ids, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
    past, logits = out.past_key_values, out.logits[:, -1]
    g = torch.Generator(device=logits.device).manual_seed(seed)
    stop = torch.tensor(stop_ids, device=logits.device)
    finished = torch.zeros(B, dtype=torch.bool, device=logits.device)
    toks, keeps = [], []
    for step in range(max_new):
        nxt = _sample_topk(logits, g, temp, topk)
        is_stop = torch.isin(nxt, stop)
        keep = ~finished & ~is_stop
        finished = finished | is_stop
        toks.append(nxt); keeps.append(keep)
        if bool(finished.all()):
            break
        am = torch.cat([am, keep.to(am.device, am.dtype)[:, None]], 1)
        pos = (am.cumsum(1) - 1).clamp(min=0)[:, -1:]
        hook.ctx = dict(mask=torch.ones(B, 1, device=dev), prefill=False)
        out = model(input_ids=nxt.to(dev)[:, None], attention_mask=am, position_ids=pos, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
    T = torch.stack(toks, 1).cpu(); K = torch.stack(keeps, 1).cpu()
    texts = [tok.decode(T[i][K[i]], skip_special_tokens=True).strip() for i in range(B)]
    # forced cue, read the option digit after it (same edit, applied to the cue tokens too)
    S = len(suffix_ids)
    sfx = torch.tensor(suffix_ids, device=dev)[None].expand(B, S)
    am = torch.cat([am, torch.ones(B, S, device=am.device, dtype=am.dtype)], 1)
    pos = (am.cumsum(1) - 1).clamp(min=0)[:, -S:]
    hook.ctx = dict(mask=torch.ones(B, S, device=dev), prefill=False)
    out = model(input_ids=sfx, attention_mask=am, position_ids=pos, past_key_values=past, use_cache=True, logits_to_keep=1)
    lg, mass = option_logits_from(out.logits[:, -1], opt_ids)
    return lg, mass, texts
