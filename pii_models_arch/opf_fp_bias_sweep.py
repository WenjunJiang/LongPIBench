"""How OPF false positives respond to the Viterbi 'enter span' bias.

Sweeps transition_bias_background_to_start (and the matching end_to_start) and counts spans on
PII-free texts plus a few texts that do contain PII. Biases are passed via calibration JSON files.
"""
import json, os
from opf import OPF

NO_PII = [
    "Wire to IBAN DE89 3704 0044 0532 0130 00 is pending.",           # 'IBAN' was tagged private_person before
    "The SHA-256 of the release is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08.",
    "Use the placeholder key YOUR_API_KEY_HERE in config.yaml before deploying.",
    "Apple and Samsung reported third-quarter earnings on October 28, 2025.",
    "The meeting moved to 3 pm Friday in Room B-204 of the Seoul R&D center.",
    "See RFC 9110 and https://www.ietf.org/ for the HTTP semantics.",
]
WITH_PII = [
    "Hi, I'm Daniel Whitfield, reach me at daniel.w@meridiancap.com or +1 415-555-0123.",
    "my pw is Tr0ub4dor&3 and the backup key is AKIA4EXAMPLE7QXZ2LMN, born 03/07/91 in Busan",
]
KEYS = ["transition_bias_background_stay", "transition_bias_background_to_start",
        "transition_bias_end_to_background", "transition_bias_end_to_start",
        "transition_bias_inside_to_continue", "transition_bias_inside_to_end"]

def calib(start_bias):
    b = {k: 0.0 for k in KEYS}
    b["transition_bias_background_to_start"] = start_bias
    b["transition_bias_end_to_start"] = start_bias
    p = f"calib/start_{start_bias:+.1f}.json"
    json.dump({"operating_points": {"default": {"biases": b}}}, open(p, "w"), indent=2)
    return p

for bias in [2.0, 0.0, -2.0, -4.0, -10.0, -20.0]:
    opf = OPF(device="cpu").set_viterbi_decoder(calibration_path=calib(bias))
    print(f"\n=== enter-span bias {bias:+.1f} ===")
    fp = 0
    for t in NO_PII:
        s = [(x["label"], x["text"]) for x in opf.redact(t).to_dict()["detected_spans"]]
        fp += len(s)
        print(f"  [no PII] {len(s)} span(s) {s}")
    for t in WITH_PII:
        s = [(x["label"], x["text"]) for x in opf.redact(t).to_dict()["detected_spans"]]
        print(f"  [PII]    {len(s)} span(s) {s}")
    print(f"  -> spans on PII-free texts: {fp}")
