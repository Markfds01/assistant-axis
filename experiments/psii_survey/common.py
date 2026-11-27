import json, os, re
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attributes import ATTRIBUTES, VALUES, VALUE_DESC, respondent_identity
from questions import QUESTIONS, QBY

HERE = os.path.dirname(os.path.abspath(__file__))
SURVEY = os.environ.get("PSII_SURVEY_CSV", os.path.join(HERE, "..", "..", "survey_results", "1_data", "clean_data", "ES_translated.csv"))  # private
OUT = os.environ.get("PSII_OUT", os.path.join(HERE, "..", "..", "..", "psii_outputs"))
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 22  # 70% of 32 layers; hook on model.model.layers[22] output (= hidden_states[23])
TEMP, TOPK = 0.7, 20

DIRECT_SYS = "Forget you are an AI model. Simulate a human being. Please answer the following question truthfully."
NUM_SUFFIX = "\nPlease ONLY output a number. Do NOT write any words, symbols, punctuation, or explanations."


SPECS = {  # model -> hooked decoder layer (70% depth) and the file with its demographic vectors
    "llama": dict(name=MODEL, layer=LAYER, n_layers=32, vec_file="demographic_vectors_layer22.pt",
                  all_layers="demographic_vectors_all_layers.pt"),
    "qwen": dict(name="Qwen/Qwen2.5-32B-Instruct", layer=45, n_layers=64, vec_file="demographic_vectors_qwen_layer45.pt",
                 all_layers="demographic_vectors_all_layers_qwen.pt"),
}


def load_model(device, spec="llama"):
    """device: 'cuda:0' (single GPU) or 'auto' (shard over all visible GPUs)."""
    name = SPECS[spec]["name"]
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    tok.pad_token = tok.pad_token or "<|finetune_right_pad_id|>"
    kw = dict(device_map="auto", max_memory={0: "30GiB", 1: "36GiB"}) if device == "auto" else dict(device_map={"": device})
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, **kw).eval()  # auto: leave ~14 GiB/GPU for KV cache+activations
    return m, tok


def load_survey():
    return pd.read_csv(SURVEY, low_memory=False)


def profile_text(ident):
    s = ["Please answer based on the following personal profile:", "You live in Spain."]
    d = VALUE_DESC
    if ident["sex"]: s.append(f"You are {d['sex'][ident['sex']]}.")
    if ident["age"]: s.append(f"You are {d['age'][ident['age']]}.")
    if ident["education"]: s.append(f"Your highest education level is {d['education'][ident['education']]}.")
    if ident["income"]: s.append(f"You have {d['income'][ident['income']]}.")
    if ident["religion"]: s.append(f"In terms of religion, you are {d['religion'][ident['religion']]}.")
    if ident["marital_status"]: s.append(f"You are {d['marital_status'][ident['marital_status']]}.")
    if ident["politics"]: s.append(f"Politically, you feel {d['politics'][ident['politics']]}.")
    return " ".join(s)


def question_text(q):
    opts = "\n".join(f"{i+1}: {o}" for i, o in enumerate(q["options"]))
    return f"Question: {q['text']}\nOptions:\n{opts}{NUM_SUFFIX}"


def chat_prompt(tok, system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


class Injector:
    """Forward hook on one decoder layer. mode='all': add vec (+per-token Gaussian noise) to every non-pad
    prompt token; mode='last': add vec only to the last position (PV, response-level)."""
    def __init__(self, model, layer=LAYER):
        self.layer = model.model.layers[layer]; self.h = None
        self.vec = None; self.sigma = 0.0; self.mode = "all"; self.mask = None; self.gen = None

    def _hook(self, mod, ins, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.vec is None and self.sigma == 0: return out
        add = torch.zeros_like(hs)
        if self.vec is not None:
            add = add + self.vec.to(hs.dtype)[:, None, :]
        if self.sigma > 0:
            add = add + (torch.randn(hs.shape, device=hs.device, dtype=torch.float32, generator=self.gen) * self.sigma).to(hs.dtype)
        if self.mode == "last":
            m = torch.zeros_like(self.mask); m[:, -1] = 1
        else:
            m = self.mask
        hs = hs + add * m[:, :, None].to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def __enter__(self): self.h = self.layer.register_forward_hook(self._hook); return self
    def __exit__(self, *a): self.h.remove()


def option_token_ids(tok, kmax=6):
    return [tok.encode(str(i + 1), add_special_tokens=False)[0] for i in range(kmax)]
