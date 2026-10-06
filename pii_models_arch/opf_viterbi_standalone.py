"""Reuse OPF's constrained BIES Viterbi decoder without the OPF model, on a custom label set.

Run with the OPF venv: privacy_filter_test/privacy-filter/.venv/bin/python opf_viterbi_standalone.py
"""
import torch
from opf._core.sequence_labeling import build_label_info
from opf._core.decoding import ViterbiCRFDecoder
from opf._core.spans import labels_to_spans

# Any span types work: O + {B,I,E,S} x types.
names = ["O"] + [f"{p}-{t}" for t in ("injection", "health") for p in "BIES"]
label_info = build_label_info(names)
decoder = ViterbiCRFDecoder(label_info=label_info)  # 6 transition biases default to 0.0

# Toy per-token log-probs [seq_len, num_labels]; replace with your model's log_softmax output.
idx = {n: i for i, n in enumerate(names)}
lp = torch.full((6, len(names)), -6.0)
for t, (best, alt) in enumerate([("B-injection", None), ("I-injection", None), ("O", "I-injection"),
                                 ("I-injection", None), ("E-health", "E-injection"), ("O", None)]):
    lp[t, idx[best]] = -0.3
    if alt:
        lp[t, idx[alt]] = -1.2

print("argmax :", [names[i] for i in lp.argmax(1).tolist()])
path = decoder.decode(lp)
print("viterbi:", [names[i] for i in path])
print("spans (type, start_tok, end_tok):",
      [(label_info.span_class_names[s], a, b) for s, a, b in labels_to_spans(dict(enumerate(path)), label_info)])
