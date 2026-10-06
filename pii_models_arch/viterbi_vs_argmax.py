"""Compare OPF constrained Viterbi decoding with per-token argmax on the same logits."""
import torch, torch.nn.functional as F
from opf import OPF

TEXTS = [
    "Contact Dr. Maria del Carmen Rodríguez-Fernández at maria.rf@clinic-example.org or ext. 4471.",
    "Wire to IBAN DE89 3704 0044 0532 0130 00, ref inv-2026/0912, attn J. P. van der Berg, 14 Rue de la Paix 75002 Paris.",
    "my pw is Tr0ub4dor&3 and the backup key is AKIA4EXAMPLE7QXZ2LMN, born 03/07/91 in Busan",
    "Call 010 - 9876 - 5432 tomorrow, ask for Kim Min-jun (김민준), address 서울시 강남구 테헤란로 152",
]
vit = OPF(device="cpu")
arg = OPF(device="cpu"); arg.set_decode_mode("argmax")
rt = vit.get_runtime()
names = rt.label_info.span_class_names
tags = rt.label_info.token_boundary_tags
tok2span = rt.label_info.token_to_span_label

for text in TEXTS:
    v = vit.redact(text).to_dict()["detected_spans"]
    a = arg.redact(text).to_dict()["detected_spans"]
    print("\nTEXT:", text)
    print("  viterbi:", [(s["label"], s["text"]) for s in v])
    print("  argmax :", [(s["label"], s["text"]) for s in a])
    if v != a:
        ids = rt.encoding.encode(text)
        with torch.no_grad():
            lp = F.log_softmax(rt.model(torch.tensor([ids], dtype=torch.int32), attention_mask=torch.ones(1, len(ids), dtype=torch.bool)).float(), -1)[0]
        print("  per-token argmax tags (token -> tag):")
        row = []
        for i, t in enumerate(ids):
            k = int(lp[i].argmax())
            tag = "O" if k == 0 else f"{tags[k]}-{names[tok2span[k]]}"
            row.append(f"{rt.encoding.decode([t])!r}:{tag}")
        print("   ", " | ".join(row))
