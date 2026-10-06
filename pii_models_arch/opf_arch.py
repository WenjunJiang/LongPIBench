"""OpenAI Privacy Filter: architecture + how JSON is produced."""
import json, time, torch
from opf import OPF
from common import TEXT, param_count, banner

opf = OPF(device="cpu")
rt = opf.get_runtime()
m = rt.model
banner("checkpoint config"); print(json.dumps({k: v for k, v in rt.model.config.__dict__.items()} if hasattr(m, "config") else {}, default=str)[:800])
banner("module tree (top level)")
for name, mod in m.named_children():
    print(f"{name:12s} {type(mod).__name__:20s} params={param_count(mod):,}")
print("block[0] children:", [(n, type(c).__name__) for n, c in m.block[0].named_children()])
print("total params:", f"{param_count(m):,}")
print("label space:", rt.label_info.__dict__ if hasattr(rt.label_info, "__dict__") else rt.label_info)

banner("raw model output (one forward pass)")
cap = {}
h = m.register_forward_hook(lambda mod, i, o: cap.setdefault("logits", o))
t = time.time(); res = opf.redact(TEXT); dt = time.time() - t
h.remove()
lg = cap["logits"]
print("input tokens:", len(rt.encoding.encode(TEXT)), "| logits shape:", tuple(lg.shape), "(batch, tokens, labels)")
print(f"redact() wall time on CPU: {dt:.2f}s")

banner("decoded -> JSON (opf.redact(...).to_json())")
print(res.to_json(indent=2))
