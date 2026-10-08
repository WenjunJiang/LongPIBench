# PII tagger on mmBERT-small

Token-level PII detector trained on NVIDIA Nemotron-PII:

| Part | Source it follows |
|---|---|
| Labels: `O` + B/I/E/S × 8 types = 33 | OpenAI Privacy Filter v2 taxonomy |
| Decoding: constrained BIES Viterbi with 6 transition biases | OpenAI Privacy Filter (`opf/_core/decoding.py`); `tests/test_viterbi.py` checks identical paths against OPF |
| Loss: `1.5 × class-weighted CE(BIES) + 0.3 × BCE(sensitivity)` | PII-TRACE (arXiv 2609.22200, eq. 3) |
| Encoder | `jhu-clsp/mmBERT-small` (140M params, hidden 384, 8192 context) |
| Data | `nvidia/Nemotron-PII` (CC-BY-4.0), 55 source labels mapped to the 8 OPF types |

## Layout

```
conf/config.yaml                         all defaults: data, model, train, loss, decode, tune, final, predict
conf/label_map/nemotron_to_opf_v2.yaml   Nemotron -> OPF label mapping, merge rules, sensitive labels
conf/experiment/smoke.yaml               tiny CPU run used to check the pipeline
main.py                                  Hydra entry point (stage=preprocess|tune|final|calibrate|predict)
pii_mmbert/labels.py                     BIES label space, labels -> spans
pii_mmbert/viterbi.py                    constrained Viterbi (OPF semantics)
pii_mmbert/data.py                       label mapping, tokenization, windows; Hugging Face datasets cache
pii_mmbert/model.py                      encoder + tag head + sensitivity head, PII-TRACE loss
pii_mmbert/engine.py                     training loop, document-level evaluation
pii_mmbert/metrics.py                    span and token metrics
pii_mmbert/calibration.py                Platt scaling, F-beta thresholds, ECE
pii_mmbert/stages.py                     preprocess / Ray Tune / final / calibrate / predict
tests/                                   mapping, BIES, windows, Viterbi-vs-OPF tests
requirements.txt                         dependencies (pip install -r requirements.txt)
pytest.ini                               puts this directory on sys.path for the tests
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch            # pick the CUDA build for GPU machines
pip install -r requirements.txt
mkdir -p data_raw
for s in train test; do
  curl -L -o data_raw/$s.parquet \
    https://huggingface.co/datasets/nvidia/Nemotron-PII/resolve/main/data/$s-00000-of-00001.parquet
done
pytest -q
```

The package is not pip-installed. Run every command from this directory: `python main.py` imports
`pii_mmbert` from here, `pytest.ini` does the same for the tests, and `stage=tune` passes this directory to
the Ray Tune workers through `PYTHONPATH`.

## Running

Every setting lives in `conf/config.yaml`. Override any of them on the command line with Hydra syntax
(`key=value`); there are no argparse flags.

```bash
# 1. Tokenize and cache (also validates that every Nemotron label is covered by the mapping)
python main.py stage=preprocess

# 2. Hyperparameter search with Ray Tune (search space and ranges: tune.search_space in config.yaml)
python main.py stage=tune
python main.py stage=tune tune.num_samples=32 tune.resources_per_trial.gpu=1 train.epochs=4

# 3. Final model: defaults < tuned best params < command-line overrides
python main.py stage=final final.best_params_path=outputs/tune/best_params.json
python main.py stage=final final.best_params_path=outputs/tune/best_params.json train.epochs=5 train.lr=3e-5

# 4. Calibrate: Platt scaling + F-beta thresholds on the held-out calib split, written to calibration.json
python main.py stage=calibrate
python main.py stage=calibrate calibration.beta=2.0          # favour recall when choosing thresholds

# 5. Predict (applies calibration.json when present: calibrated scores, thresholded spans, sensitivity)
python main.py stage=predict predict.text="Call Kim Min-jun at 010-1234-5678"

# Any stage with the tiny CPU settings
python main.py +experiment=smoke stage=tune
```

Outputs:

- `outputs/tune/best_params.json`, `summary.json`, `trials.csv`, `ray_results/`
- `outputs/final_model/`: `encoder/` (Hugging Face format + tokenizer), `heads.pt`, `pii_tagger.json`
  (labels, decode biases, window settings), `resolved_config.yaml`, `metrics.json`, and after
  `stage=calibrate` `calibration.json` (Platt parameters, thresholds, calibration and test report)

## Design notes

- **Label mapping** is data, not code: edit `conf/label_map/nemotron_to_opf_v2.yaml`. Preprocessing fails on
  any source label that is not listed. Generic `date`/`time`, `company_name`, demographic attributes, and
  `swift_bic` are background, following OPF's definitions. City, state, county, country and postcode count
  only when they sit next to a street address or coordinate.
- **Merging**: adjacent spans of the same type separated by at most 3 whitespace/comma characters are merged,
  so first name + last name becomes one `private_person` span, as OPF predicts it.
- **Span offsets**: Nemotron's span `text` field differs from `text[start:end]` in case only (6,770 spans);
  offsets are used as-is.
- **Splits**: each Nemotron uid appears twice (us and intl locale). Nemotron's train file is split by a
  seeded hash of the uid into train / val (`data.val_fraction`, used by Ray Tune) / calib
  (`data.calib_fraction`), so the two variants never land on different sides. The calib split is never
  trained on, including in `stage=final` with `include_val_in_train=true`. Nemotron's test file is the test set.
- **Calibration** (`stage=calibrate`, `pii_mmbert/calibration.py`):
  - Raw span score = mean probability of the decoded labels over the span's tokens. Target = the span
    exactly matches a gold span (type and character offsets).
  - Platt scaling `p = sigmoid(a * logit(score) + b)`, fitted with Platt's smoothed targets, per span type
    when the type has at least `calibration.min_spans_per_type` calib spans, otherwise the global fit.
  - Threshold per type maximises F-beta on the calib split. Recall's denominator is all gold spans of that
    type, including ones the decoder never proposed, so a threshold can only trade precision for recall.
  - A type gets its own threshold only with at least `min_spans_per_type` calib spans and at least one
    correct one. With too few spans it uses the global threshold. With enough spans but none correct,
    `calibration.zero_positive_policy` decides: `global` (default; keeps recall for PII) or `reject`
    (drop the type). The rule used per type is recorded as `threshold_rule` in `calibration.json`.
  - The sensitivity head is calibrated the same way at document level (max window logit vs. whether the
    document contains a sensitive source label).
  - The report lists ECE before and after Platt on calib and test, and test P/R/F1 before and after thresholds.
- **Storage**: preprocessed documents and training windows are Hugging Face `datasets` written with
  `save_to_disk` under `cache/nemotron_<hash>/` (`train`, `val`, `test`, `windows_<split>_L<len>_S<stride>`).
  They are memory-mapped with `load_from_disk`, so training and evaluation read rows on demand, and Ray Tune
  trials receive dataset directories rather than copies of the data. `stats.json` is written last and marks
  the cache as complete; every directory is written to a `.tmp` path and renamed when done.
- **Long documents**: windows of `data.max_length` tokens with `data.stride` overlap. At evaluation, overlapping
  windows' log-probabilities are averaged per token and Viterbi runs once over the whole document (as OPF does).
- **Sensitivity head**: target = whether the window contains a span whose source label is in
  `sensitive_source_labels` (special-category data). Set `loss.lambda_sens=0` to train without it.
- **Metrics**: exact typed span P/R/F1 on character offsets (primary, `span_f1`), token-level PII-vs-O detection,
  per-type span F1, and the share of PII-free documents with any prediction.

## Verified in this repo (CPU, 4 vCPU, no GPU)

- `pytest`: label mapping, merging, BIES round trip, windowing; Viterbi paths identical to OPF's
  `ViterbiCRFDecoder` on 30 random inputs with random biases (run with the OPF package installed).
- `stage=preprocess` on all 200k rows with `data.num_proc=4`: 1.9 min, peak RSS 2.05 GB in the main
  process and at most 1.0 GB per worker, 0.93 GB on disk. (The earlier pickled-numpy cache took 7.5 min and
  7.4 GB.) Splits: train 89,980 rows / 44,990 uids, val 10,020 / 5,010, test 100,000 / 50,000; 17.3% of
  tokens are PII.
- `+experiment=smoke` (240 rows, 256-token windows): `stage=tune` ran 2 trials with ASHA stopping one after
  epoch 1; `stage=final` applied `best_params.json` and the command-line `train.batch_size=8` on top;
  `stage=predict` returned JSON spans. Smoke numbers only show the pipeline runs; they are not a quality
  result.
- After moving to `datasets`, the smoke `tune` → `final` → `predict` run was repeated and all stages passed.
  Ray Tune's random search is seeded from `seed` (`BasicVariantGenerator(random_state=seed)`); two Tuner
  runs with the same seed drew identical configurations.
- `stage=calibrate` on the smoke model (41 calib rows): span ECE 0.105 → 0.046 on calib and
  0.115 → 0.077 on test; thresholds raised test span precision 0.17 → 0.50 at unchanged recall. The smoke
  model is undertrained (34 steps), so only the mechanics are meaningful here.
- A full run needs a GPU: the smoke epoch took ~190 s for 328 windows of 256 tokens on 2 CPU threads.
