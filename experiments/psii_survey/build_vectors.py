"""Demographic vectors (paper 3.1.2): value mean - attribute mean of response-token-averaged hidden states.
Usage: python build_vectors.py [--device cuda:0|auto] [--max_new 150] [--bs 40] [--spec llama|qwen] [--limit Q]
For --spec qwen: device auto (2 GPUs), writes data/demographic_vectors_qwen_layer45.pt and all layers to OUT."""
import argparse, json, os, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max_new", type=int, default=150); ap.add_argument("--bs", type=int, default=40)
ap.add_argument("--limit", type=int, default=0, help="questions per attribute (0 = all 40)")
ap.add_argument("--spec", default="llama", choices=list(SPECS))
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
Q = json.load(open(f"{HERE}/data/dv_questions.json")); P = json.load(open(f"{HERE}/data/dv_personas.json"))
model, tok = load_model(a.device, a.spec)
torch.manual_seed(0)
nl = model.config.num_hidden_layers
LAYER = SPECS[a.spec]["layer"]
dev0 = model.get_input_embeddings().weight.device

items = []  # (attr, value, persona, question)
for at in ATTRIBUTES:
    qs = Q[at][:a.limit] if a.limit else Q[at]
    for v in VALUES[at]:
        for p in P[at][v]:
            for q in qs: items.append((at, v, p, q))
print(len(items), "prompts", flush=True)

sums = {}  # (attr,value) -> [sum (nl,H) float32, n]
log = []
for b in range(0, len(items), a.bs):
    batch = items[b:b + a.bs]
    prompts = [chat_prompt(tok, p, q) for _, _, p, q in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev0)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=TEMP, top_k=TOPK,
                             pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        resp = gen[:, plen:]
        # response token mask: tokens up to and including first eos/eot
        stops = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>" if a.spec == "llama" else "<|im_end|>")]
        eos = torch.isin(resp, torch.tensor(stops, device=resp.device))
        first = torch.where(eos.any(1), eos.float().argmax(1), torch.full((len(batch),), resp.shape[1] - 1, device=resp.device))
        rmask = (torch.arange(resp.shape[1], device=resp.device)[None] <= first[:, None]).float()
        full_mask = torch.cat([enc.attention_mask, rmask.long()], 1)
        out = model(input_ids=gen, attention_mask=full_mask.to(gen.device), output_hidden_states=True, logits_to_keep=1)
        mean = torch.stack([((h[:, plen:].float() * (rm := rmask.to(h.device))[:, :, None]).sum(1) / rm.sum(1)[:, None]).to(dev0)
                            for h in out.hidden_states[1:]], 0)  # (nl,B,H): response-token mean of each layer's output
        del out
    for i, (at, v, _, _) in enumerate(batch):
        s = sums.setdefault((at, v), [torch.zeros(nl, mean.shape[-1], device=mean.device), 0])
        s[0] += mean[:, i]; s[1] += 1
    if b == 0: log.append(tok.decode(resp[0], skip_special_tokens=True)[:300])
    if (b // a.bs) % 10 == 0: print(b, "/", len(items), flush=True)

vec = {}
for at in ATTRIBUTES:
    tot = sum(sums[(at, v)][0] for v in VALUES[at]); n = sum(sums[(at, v)][1] for v in VALUES[at])
    amean = tot / n
    for v in VALUES[at]:
        vec[f"{at}/{v}"] = (sums[(at, v)][0] / sums[(at, v)][1] - amean).cpu()
torch.save(vec, f"{OUT}/{SPECS[a.spec]['all_layers']}")
torch.save({k: x[LAYER].clone() for k, x in vec.items()}, f"{HERE}/data/{SPECS[a.spec]['vec_file']}")
norms = {k: round(float(x[LAYER].norm()), 3) for k, x in vec.items()}
json.dump({"layer": LAYER, "norms": norms, "sample_response": log}, open(f"{HERE}/data/vector_stats" + ("" if a.spec == "llama" else f"_{a.spec}") + ".json", "w"), indent=1)
print(norms)
