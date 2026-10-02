# GLiNER PII smoke test

Checks that [urchade/GLiNER](https://github.com/urchade/GLiNER) (commit `5803656`, Apache-2.0) installs from source, runs on CPU, and finds PII in one long sentence.

Environment: Linux, 4 vCPU, 15 GB RAM, no GPU, Python 3.11, torch 2.14.1+cpu, transformers 5.16.1, gliner 0.2.29.

Models tested (both Apache-2.0 on Hugging Face):
- `urchade/gliner_multi_pii-v1` (multilingual, mDeBERTa backbone)
- `knowledgator/gliner-pii-base-v1.0`

## Reproduce

```bash
cd gliner_test
git clone --depth 1 https://github.com/urchade/GLiNER.git
cd GLiNER
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install protobuf sentencepiece   # needed by the mDeBERTa tokenizer; not pulled in by pip install -e .
python ../pii_test.py
```

Without `protobuf` and `sentencepiece`, loading `gliner_multi_pii-v1` fails with `ValueError: tiktoken is required ...`.

## Results

Test input: one 727-character English sentence with 11 PII items (name, date of birth, address, phone, email, passport, bank account, bank name, credit card, medical condition, doctor's name), plus one Korean sentence. Labels are passed at inference time (zero-shot); threshold 0.4. Full output: [pii_test_output.txt](pii_test_output.txt).

| | gliner_multi_pii-v1 | gliner-pii-base-v1.0 |
|---|---|---|
| English long sentence | 11 / 11 found | 11 / 11 found |
| Health detail ("type 2 diabetes") | found, as `medical condition` | found, as `medical condition` |
| Korean sentence (name, phone, email, address, condition) | 5 / 5 found; spans include trailing particles (e.g. `010-1234-5678로`, `...123이며`) | 5 found; address cut to `123이며`, more particles attached |
| CPU time, English sentence | ~350 ms | ~250 ms |
| CPU time, Korean sentence | ~140 ms | ~130 ms |
| Model load | ~16 s (first run includes download) | ~14 s |

Notes:
- GLiNER splits words on whitespace, so Korean particles attached to a word end up inside the span. Korean output would need post-processing to strip particles.
- Unlike openai/privacy-filter (fixed 8 categories, no health category), GLiNER takes labels at inference time, so it tagged "type 2 diabetes" when given a `medical condition` label. This is an entity-level match, not detection of a whole sensitive sentence.
- Timings are a generic 4-vCPU cloud VM with no optimization; they are not phone latency.
