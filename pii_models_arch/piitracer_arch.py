"""Perplexity PII-Tracer: architecture + how JSON is produced."""
import json, time, torch
from dataclasses import asdict, is_dataclass
from transformers import AutoModel
from common import TEXT, param_count, banner

m = AutoModel.from_pretrained("perplexity-ai/PII-Tracer", trust_remote_code=True)
m.eval()
banner("config (selected)")
cfg = m.config.to_dict()
print({k: v for k, v in cfg.items() if k != "backbone"})
bb = cfg.get("backbone", {})
print("backbone:", {k: bb.get(k) for k in ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "is_causal"]})
banner("module tree")
for n, c in m.named_children():
    print(f"{n:18s} {type(c).__name__:22s} params={param_count(c):,}")
print("total params:", f"{param_count(m):,}")

banner("raw model output (one forward pass)")
cap = {}
h = m.register_forward_hook(lambda mod, i, o: cap.setdefault("out", o))
t = time.time(); spans, sens = m.predict(TEXT); dt = time.time() - t
h.remove()
out = cap["out"]
items = out.items() if isinstance(out, dict) else (enumerate(out) if isinstance(out, (tuple, list)) else [("out", out)])
for k, v in items:
    print(f"  {k}: {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}")
print(f"  predict wall time on CPU: {dt:.2f}s | sensitivity={sens}")

banner("decoded -> JSON (span objects serialized by caller)")
def to_d(s):
    return asdict(s) if is_dataclass(s) else {k: getattr(s, k) for k in ("label", "start", "end") if hasattr(s, k)}
print(json.dumps([{**to_d(s), "text": TEXT[s.start:s.end]} for s in spans], indent=2, ensure_ascii=False))
print("mask():", m.mask(TEXT))
