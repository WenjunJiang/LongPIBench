# LongPIBench
Benchmark dPID with LongPIBench dataset

## Export LongPIBench to JSONL

`export_jsonl.py` converts [LongPIBench](https://github.com/liu00222/LongPIBench) examples into
JSONL for testing a prompt-injection detector. Each line holds `text` and `label` (0 = clean,
1 = injected), plus `suite`, `item_id`, `attack`, and `goal` metadata.

```bash
pip install git+https://github.com/liu00222/LongPIBench.git
longpibench download-data          # fetches the data from Hugging Face

# Email suite: 100 clean + 100 x 3 attacks x 3 goals injected = 1000 lines
python export_jsonl.py --suite email --output data/email.jsonl

# Balanced: clean + the published default condition only (authority_spoof, goal 0)
python export_jsonl.py --suite email --attacks authority_spoof --goals 0 --output data/email_default.jsonl

# All four suites
python export_jsonl.py --suite all --output data/all.jsonl
```

Other options: `--no-clean` (injected only), `--with-system-prompt` (prepend the task prompt to
`text`), `--data-dir` (use an existing dataset snapshot).
