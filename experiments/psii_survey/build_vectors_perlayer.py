"""Per-layer demographic vectors for every layer of Llama-3.1-8B-Instruct, following the authors' recipe
(PSII `demographic_vectors/eval.py` + `generate_vec.py`) on our own persona/question dataset.

  python build_vectors_perlayer.py gen   --shard 0 --nshards 2 --device cuda:0   # generate responses + hidden-state sums
  python build_vectors_perlayer.py merge                                           # sums -> vectors (all layers)

Recipe (authors):  persona instruction = system role, question = user turn; response sampled with the chat template at
T=1.0, top_k=20, top_p=0.8 (their eval.py default temperature / generation_config); then ONE teacher-forced pass over the raw text
`persona + "\\n" + question + response` (no chat template, no special tokens, as `generate_vec.py`), hidden states of all
33 outputs (index 0 = embeddings), mean over the response tokens; vector(value) = mean over the value's samples - mean over
all samples of the attribute. No normalisation anywhere.
We additionally compute variant B: the same response teacher-forced inside the chat template (the format used at inference)
so the two can be compared. Index convention of the arrays below: hidden_states index h (0..32); the output of decoder
layer l is hidden_states[l + 1].
Differences from the authors: 5 hand-written personas x 40 hand-written questions per value (200 samples/value in the
authors' data too, but GPT-4o-written), max_new_tokens 400 instead of 1000 (responses are cut there), a 7-attribute set.
Synthetic persona responses and sums go to OUT/perlayer/ (not committed).
"""
import argparse, json, os, torch
from common import *

PL = f"{OUT}/perlayer"
GEN = dict(do_sample=True, temperature=1.0, top_k=20, top_p=0.8)


def teacher_forced_mean(model, tok, texts, prompt_lens, dev0, add_special=False):
    """texts: full strings, prompt_lens: #tokens of the prompt part. -> (33,B,H) float32 mean of hidden states over response tokens."""
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=add_special).to(dev0)
    out = model(**enc, output_hidden_states=True, logits_to_keep=1)
    T = enc.input_ids.shape[1]
    ar = torch.arange(T, device=dev0)[None]
    pl = torch.tensor(prompt_lens, device=dev0)[:, None]
    m = ((ar >= pl) & enc.attention_mask.bool()).float()
    res = torch.stack([((h.float() * m[:, :, None]).sum(1) / m.sum(1)[:, None]) for h in out.hidden_states], 0).cpu()
    tok.padding_side = "left"
    return res


def gen_mode(a):
    os.makedirs(PL, exist_ok=True)
    Q = json.load(open(f"{HERE}/data/dv_questions.json")); P = json.load(open(f"{HERE}/data/dv_personas.json"))
    model, tok = load_model(a.device)
    dev0 = model.get_input_embeddings().weight.device
    torch.manual_seed(1000 + a.shard)
    values = [(at, v) for at in ATTRIBUTES for v in VALUES[at]]
    values = [x for i, x in enumerate(values) if i % a.nshards == a.shard]
    items = [(at, v, p, q) for at, v in values for p in P[at][v] for q in (Q[at][:a.limit] if a.limit else Q[at])]
    print(f"shard {a.shard}: {len(values)} values, {len(items)} prompts", flush=True)
    sums = {x: {"A": [torch.zeros(33, model.config.hidden_size), 0], "B": [torch.zeros(33, model.config.hidden_size), 0]} for x in values}
    rf = open(f"{PL}/responses_shard{a.shard}.jsonl", "w")
    stops = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    for b in range(0, len(items), a.bs):
        batch = items[b:b + a.bs]
        chat = [chat_prompt(tok, p, q) for _, _, p, q in batch]
        tok.padding_side = "left"
        enc = tok(chat, return_tensors="pt", padding=True, add_special_tokens=False).to(dev0)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, eos_token_id=stops, **GEN)
        resp = [tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for g in gen]
        # variant A (authors): raw "persona\nquestion" + response, no template, no special tokens
        rawp = [(p + "\n" + q).strip() for _, _, p, q in batch]
        plA = [len(tok(x, add_special_tokens=False).input_ids) for x in rawp]
        # variant B: chat template prompt (with generation prompt) + response
        plB = [len(tok(x, add_special_tokens=False).input_ids) for x in chat]
        for sub in range(0, len(batch), a.tf_bs):
            sl = slice(sub, sub + a.tf_bs)
            with torch.no_grad():
                hA = teacher_forced_mean(model, tok, [x + r for x, r in zip(rawp[sl], resp[sl])], plA[sl], dev0)
                hB = teacher_forced_mean(model, tok, [x + r for x, r in zip(chat[sl], resp[sl])], plB[sl], dev0)
            for i, (at, v, _, _) in enumerate(batch[sl]):
                for k, h in (("A", hA), ("B", hB)):
                    s = sums[(at, v)][k]; s[0] += h[:, i]; s[1] += 1
        for (at, v, p, q), r in zip(batch, resp):
            rf.write(json.dumps({"attr": at, "value": v, "question": q, "response": r}, ensure_ascii=False) + "\n")
        rf.flush()
        print(b, "/", len(items), flush=True)
    torch.save(sums, f"{PL}/sums_shard{a.shard}.pt")


def merge_mode(a):
    sums = {}
    for f in sorted(os.listdir(PL)):
        if f.startswith("sums_shard") and f.endswith(".pt"):
            sums.update(torch.load(f"{PL}/{f}"))
    assert len(sums) == sum(len(VALUES[at]) for at in ATTRIBUTES), len(sums)
    out = {"A": {}, "B": {}}
    for k in out:
        for at in ATTRIBUTES:
            tot = sum(sums[(at, v)][k][0] * 1.0 for v in VALUES[at]); n = sum(sums[(at, v)][k][1] for v in VALUES[at])
            amean = tot / n
            for v in VALUES[at]:
                out[k][f"{at}/{v}"] = (sums[(at, v)][k][0] / sums[(at, v)][k][1] - amean)
    torch.save(out["A"], f"{PL}/demographic_vectors_perlayer_authors.pt")
    torch.save(out["B"], f"{PL}/demographic_vectors_perlayer_chat.pt")
    # compact export of the injection layers (hidden_states index 16 / 21 / 26) for the repo
    torch.save({k: {h: x[h].clone() for h in (16, 21, 26)} for k, x in out["A"].items()}, f"{HERE}/data/demographic_vectors_perlayer_h16_21_26.pt")
    ns = {f"{at}/{v}": sums[(at, v)]["A"][1] for at in ATTRIBUTES for v in VALUES[at]}
    # per-layer norm table + variant agreement (cosine A vs B, and hs[l] vs hs[l+1] i.e. authors' index vs matched index)
    stats = {"n_samples_per_value": ns, "norm_A_by_layer_mean": [], "norm_B_by_layer_mean": [], "cos_A_B_by_layer_mean": [],
             "cos_A_adjacent_by_layer_mean": []}
    keys = list(out["A"])
    for h in range(33):
        A = torch.stack([out["A"][k][h] for k in keys]); B = torch.stack([out["B"][k][h] for k in keys])
        stats["norm_A_by_layer_mean"].append(float(A.norm(dim=-1).mean())); stats["norm_B_by_layer_mean"].append(float(B.norm(dim=-1).mean()))
        stats["cos_A_B_by_layer_mean"].append(float(torch.cosine_similarity(A, B, dim=-1).mean()))
        if h < 32:
            A1 = torch.stack([out["A"][k][h + 1] for k in keys])
            stats["cos_A_adjacent_by_layer_mean"].append(float(torch.cosine_similarity(A, A1, dim=-1).mean()))
    json.dump(stats, open(f"{HERE}/data/vector_stats_perlayer.json", "w"), indent=1)
    print(json.dumps({k: [round(x, 3) for x in v] for k, v in stats.items() if k != "n_samples_per_value"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["gen", "merge"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--tf_bs", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=400); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    gen_mode(a) if a.mode == "gen" else merge_mode(a)
