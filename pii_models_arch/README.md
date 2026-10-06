# PII model architecture traces

Loads four open PII detectors on CPU, prints each one's module tree and the shape of its raw output on one forward pass, and shows how that output becomes JSON.

Test text (one English sentence, 6 PII items): see `common.py`.
Environment: Linux, 4 vCPU, 15 GB RAM, no GPU.

| Script | Model | Env | Output |
|---|---|---|---|
| `opf_arch.py` | OpenAI Privacy Filter (`~/.opf/privacy_filter`) | `privacy_filter_test/privacy-filter/.venv` | `opf_arch_output.txt` |
| `nvidia_gliner_arch.py` | `nvidia/gliner-PII` | `gliner_test/GLiNER/.venv` (gliner 0.2.29, transformers 5.16.1) | `nvidia_gliner_arch_output.txt` |
| `gliner2_arch.py` | `fastino/gliner2-privacy-filter-PII-multi` | `.venv-gliner2` (gliner2 2.0.0, transformers 4.57.6) | `gliner2_arch_output.txt` |
| `piitracer_arch.py` | `perplexity-ai/PII-Tracer` | `gliner_test/GLiNER/.venv` (needs transformers >= 5.2) | `piitracer_arch_output.txt` |

`.venv-gliner2` setup: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install "gliner2[local]" protobuf sentencepiece`
(without `protobuf`/`sentencepiece`, loading the GLiNER2-PII tokenizer fails under transformers 4.57.6).

## Measured

| | OpenAI PF | NVIDIA GLiNER-PII | GLiNER2-PII | PII-Tracer |
|---|---|---|---|---|
| Backbone | own 8-layer MoE (128 experts, 4 active), banded bidirectional attention (±128 tokens) | DeBERTa-v3-large | mDeBERTa-v3-base | Qwen3 (28 layers) run bidirectionally |
| Params (counted) | 1,399,486,832 | 445,463,040 | 307,098,645 | 596,088,870 |
| Head | Linear 640→33 (O + BIES×8) | span rep (max width 12) · label-prompt dot product | span rep (max width 8) · per-instance field queries from `count_embed`; `count_pred` picks 0–19 instances | Linear 1024→37 (O + BIES×9) + Linear 1024→1 sensitivity |
| Raw output on test text | `(1, 65, 33)` per-token label logits | `(1, 42, 12, 6)` word × width × label | `span_rep (1, 36, 8, 768)`, `count_pred (1, 20)`, queries `(1, n_fields, 768)` | `logits (1, 84, 37)`, `sensitivity_logits (1,)` |
| Decoder | constrained BIES Viterbi | threshold + greedy non-overlap | threshold per (instance, field) | constrained BIOES Viterbi |
| Found (of 6) | 6 | 6 (name split into two spans) | 6 | 6 |
| CPU wall time | 8.9 s | 4.1 s | 0.26 s | 8.7 s |

Timings include first-call overhead and are not phone latency.

In all four, the network outputs score tensors only. JSON comes from deterministic Python after decoding: `RedactionResult.to_dict()` in OPF, `predict_entities` dicts in GLiNER, `_extract_entities` / `_extract_structures` in GLiNER2, and `PredictedSpan` dataclasses in PII-Tracer that the caller serializes.
