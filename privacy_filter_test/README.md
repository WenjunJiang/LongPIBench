# openai/privacy-filter smoke test

Checks that [openai/privacy-filter](https://github.com/openai/privacy-filter) (commit `f7f00ca`) installs and runs on CPU.

Environment: Linux, 4 vCPU, 15 GB RAM, no GPU, Python 3.11, torch 2.14.1+cpu.

## Reproduce

```bash
cd privacy_filter_test
git clone --depth 1 https://github.com/openai/privacy-filter.git
cd privacy-filter
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .

opf --device cpu "Alice was born on 1990-01-02."                    # downloads ~2.5 GB checkpoint to ~/.opf/privacy_filter
opf eval examples/data/sample_eval_five_examples.jsonl --device cpu   # bundled sample eval
python ../smoke_test.py                                               # English, Korean, secret, long document
```

The checkpoint auto-downloads only to the default path `~/.opf/privacy_filter`; setting `OPF_CHECKPOINT` to an empty directory fails instead of downloading.

## Results

| Check | Result |
|---|---|
| Install (`pip install -e .`) | OK |
| One-shot CLI redaction | OK: `Alice was born on <PRIVATE_DATE>.` ("Alice" alone was not tagged) |
| Bundled eval, 5 synthetic examples | OK: span F1 0.8955, detection F1 0.9846 ([eval_output.txt](eval_output.txt)) |
| English PII (name, phone, email, account, address) | All 5 found |
| Korean PII (name, phone, email, address) | All 4 found |
| API key | Tagged as `secret` |
| Health sentence ("diagnosed with depression") | Not tagged: only the date and the doctor's name; the model has no health category |
| ~7,700-character document | Found the 3 PII items buried in it |

Full output: [smoke_test_output.txt](smoke_test_output.txt).

## CPU speed

| Input | Time |
|---|---|
| Model load + warm-up | ~4 s |
| One short sentence | ~1–2 s |
| ~160-character paragraph | ~3–6 s |
| ~7,700-character document | ~86 s |

The bundled eval reports about 13 tokens/s for the model forward pass on this machine. This is the repo's reference PyTorch path on a generic 4-vCPU cloud VM, not an optimized on-device build, so it says nothing definitive about phone latency; it does show that an unoptimized CPU run is far too slow for interactive use on long files.
