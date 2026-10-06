"""Feed a single per-token score p (e.g. injection probability) into OPF's Viterbi without retraining.

Emissions: B/I/E/S-<type> all get log p, O gets log(1-p). The 'enter span' bias then acts as a
per-span penalty, so short noisy blips are dropped and multiple spans are returned.
Run with privacy_filter_test/privacy-filter/.venv/bin/python.
"""
import json, math, torch
from opf._core.sequence_labeling import build_label_info
from opf._core.decoding import ViterbiCRFDecoder
from opf._core.spans import labels_to_spans

names = ["O"] + [f"{t}-injection" for t in "BIES"]
li = build_label_info(names)

def decode(p, enter_bias=0.0):
    p = torch.tensor(p).clamp(1e-6, 1 - 1e-6)
    em = torch.empty(len(p), len(names))
    em[:, 0] = torch.log1p(-p)
    em[:, 1:] = torch.log(p).unsqueeze(1)
    dec = ViterbiCRFDecoder(label_info=li, transition_bias_background_to_start=enter_bias,
                            transition_bias_end_to_start=enter_bias)
    path = dec.decode(em)
    return [(a, b) for _, a, b in labels_to_spans(dict(enumerate(path)), li)]

#      noise blip       span 1 with a dip            span 2
p = [0.1, 0.6, 0.1, 0.1, 0.8, 0.9, 0.4, 0.9, 0.85, 0.1, 0.1, 0.7, 0.75, 0.8, 0.1]
print("scores:", p)
print("threshold 0.5     :", [i for i, x in enumerate(p) if x > 0.5])
for b in [0.0, -1.0, -3.0]:
    print(f"viterbi bias {b:+.1f}:", decode(p, b), "(start, end_exclusive)")
