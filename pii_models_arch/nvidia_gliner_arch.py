"""NVIDIA GLiNER-PII (urchade GLiNER library): architecture + how JSON is produced."""
import json, time, torch
from gliner import GLiNER
from common import TEXT, param_count, banner

LABELS = ["person", "date of birth", "email", "phone number", "account number", "address"]
g = GLiNER.from_pretrained("nvidia/gliner-PII", map_location="cpu")
banner("config (selected)")
cfg = g.config
for k in ["model_name", "span_mode", "max_width", "max_len", "hidden_size", "dropout", "fine_tune", "subtoken_pooling", "ent_token", "sep_token", "class_token_index"]:
    print(f"{k:18s}", getattr(cfg, k, None))
banner("module tree")
core = g.model
for name, mod in core.named_children():
    print(f"{name:22s} {type(mod).__name__:28s} params={param_count(mod):,}")
    for n2, c2 in mod.named_children():
        if n2 in ("bert_layer", "token_rep_layer", "rnn", "span_rep_layer", "prompt_rep_layer", "projection", "scorer"):
            pass
print("total params:", f"{param_count(core):,}")

banner("raw model output (one forward pass)")
cap = {}
h = core.register_forward_hook(lambda mod, i, o: cap.setdefault("out", o))
t = time.time(); ents = g.predict_entities(TEXT, LABELS, threshold=0.5); dt = time.time() - t
h.remove()
out = cap["out"]
lg = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, (tuple, list)) else out)
print("output type:", type(out).__name__, "| logits shape:", tuple(lg.shape), "(batch, words, max_width, num_labels)")
print(f"predict_entities wall time on CPU: {dt:.2f}s")

banner("decoded -> JSON (json.dumps of predict_entities list)")
print(json.dumps(ents, indent=2, ensure_ascii=False))
