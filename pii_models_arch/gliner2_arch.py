"""Fastino GLiNER2-PII: architecture + how JSON is produced."""
import json, time, torch
from gliner2 import GLiNER2
from common import TEXT, param_count, banner

m = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")
m.eval()
banner("config"); print(m.config)
banner("module tree")
for n, c in m.named_children():
    print(f"{n:12s} {type(c).__name__:18s} params={param_count(c):,}")
print("total params:", f"{param_count(m):,}")

caps = {}
def hook(name):
    return lambda mod, i, o: caps.setdefault(name, []).append(
        tuple(o.shape) if torch.is_tensor(o) else type(o).__name__)
hs = [m.count_pred.register_forward_hook(hook("count_pred logits")),
      m.count_embed.register_forward_hook(hook("count_embed (instance x field queries)")),
      m.span_rep.register_forward_hook(hook("span_rep"))]

LABELS = ["full_name", "date_of_birth", "email", "phone_number", "account_number", "street_address"]
banner("1) extract_entities")
t = time.time(); ents = m.extract_entities(TEXT, LABELS, include_spans=True, include_confidence=True); dt = time.time() - t
for k, v in caps.items(): print(f"  hooked {k}: {v}")
print(f"  wall time on CPU: {dt:.2f}s")
print(json.dumps(ents, indent=2, ensure_ascii=False))

caps.clear()
banner("2) extract_json (structured record, schema given at inference)")
SCHEMA = {"contact": ["name::str", "email::str", "phone::str", "address::str"]}
t = time.time(); rec = m.extract_json(TEXT, SCHEMA); dt = time.time() - t
for k, v in caps.items(): print(f"  hooked {k}: {v}")
print(f"  wall time on CPU: {dt:.2f}s")
print(json.dumps(rec, indent=2, ensure_ascii=False))
for h in hs: h.remove()
