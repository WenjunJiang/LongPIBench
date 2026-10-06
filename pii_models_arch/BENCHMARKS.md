# Reported benchmark results and threshold guidance (as published; not re-run here)

Sources: PII-TRACE paper (arXiv 2609.22200, Perplexity); GLiNER2-PII HF card (Fastino); nvidia/gliner-PII HF card;
OpenAI Privacy Filter model card PDF (cdn.openai.com/.../OpenAI-Privacy-Filter-Model-Card.pdf). Numbers from different
sources use different metrics and label mappings and are not comparable across tables.

## A. PII-TRACE test set (1,922 conversations), all four models, label-agnostic (paper Tables 3, 7, Fig. 3)
| Model | Char P | Char R | Char F1 | Span-overlap F1 | Consistent detection | FP0 (lower better) | Korean char F1 |
|---|---|---|---|---|---|---|---|
| OpenAI PF | 0.358 | 0.785 | 0.492 | 0.401 | 0.386 | 0.557 | 0.358 |
| NVIDIA GLiNER-PII (zero-shot) | 0.261 | 0.785 | 0.392 | 0.338 | 0.472 | 0.901 | 0.460 |
| GLiNER2-PII | 0.211 | 0.475 | 0.292 | 0.269 | 0.448 | 0.862 | 0.266 |
| PII-Tracer (single 4096 window) | 0.507 | 0.830 | 0.629 | 0.621 | 0.853 | 0.385 | 0.616 |
PII-Tracer with 50%-overlap sliding windows: P 0.493, R 0.965, F1 0.653 (Table 8). Benchmark built by PII-Tracer's authors from the same traffic as its training data.

## B. SPY (legal + medical), exact-match span F1, from the GLiNER2-PII card
| Model | Legal F1 | Medical F1 | Avg F1 |
|---|---|---|---|
| GLiNER2-PII | 0.473 | 0.480 | 0.477 |
| NVIDIA GLiNER-PII | 0.390 | 0.411 | 0.400 |
| urchade/gliner_multi_pii-v1 | 0.377 | 0.419 | 0.398 |
| OpenAI PF | 0.354 | 0.406 | 0.380 |
Other SPY numbers: PII-TRACE paper char F1 PII-Tracer 0.585 vs OPF 0.543; OPF card token F1 0.545 zero-shot, 0.962 after fine-tuning on 10% of SPY train.

## C. Other public benchmarks (each row from one source)
| Benchmark | Metric | Result | Source |
|---|---|---|---|
| ai4privacy | char F1 | PII-Tracer 0.950 vs OPF 0.907 | PII-TRACE paper |
| PII-Masking-300k (ai4privacy) | token F1 / span F1 | OPF 0.960 / 0.926 | OPF card |
| Nemotron-PII | char F1 | PII-Tracer 0.847 vs OPF 0.709 | PII-TRACE paper |
| Nemotron-PII | strict F1 (threshold 0.3) | NVIDIA GLiNER-PII 0.87 (its own training-data family) | NVIDIA card |
| Gretel PII masking | char F1 | PII-Tracer 0.952 vs OPF 0.895 | PII-TRACE paper |
| TAB (real court cases) | char F1 | PII-Tracer 0.594 vs OPF 0.350 | PII-TRACE paper |
| Argilla PII / AI4Privacy | strict F1 (threshold 0.3) | NVIDIA GLiNER-PII 0.70 / 0.64 | NVIDIA card |
| CredData (secrets) | token F1 / span F1 | OPF 0.842 / 0.617 | OPF card |
| Korean synthetic | P / R / F1 | OPF 0.903 / 0.887 / 0.895 | OPF card Table 7 |
| Korean, category clue before PII | P / R | OPF 0.651 / 0.821 | OPF card Table 8 |

## D. Threshold / operating-point controls
| Model | Knob | Default | Published guidance |
|---|---|---|---|
| OpenAI PF | 6 Viterbi transition biases (background stay, enter span, continue, close, end-to-start...) via `--viterbi-calibration-path` | all 0.0 (`viterbi_calibration.json`, "default") | Operating points chosen from span P/R/F1/F2 on held-out data; biases toward span entry/continuation raise recall |
| NVIDIA GLiNER-PII | sigmoid span-score `threshold`; `flat_ner` greedy non-overlap | 0.5 (library) | Card's evaluation used 0.3; usage example uses 0.5 |
| GLiNER2-PII | `threshold` per (instance, field) | 0.5 (`extract_entities`) | Raise per-label thresholds for `person`/`full_name`; dictionary filter for common false positives; calibrate on a small domain dev set |
| PII-Tracer | Viterbi `viterbi_b_bias` (enter B) and `viterbi_e_bias` (leave E); long-input policy trunc / chunk / slide | 0.0 / 0.0; single window | Choose decoding policy at deploy time: slide raises recall 0.830→0.965 at precision 0.507→0.493 |
