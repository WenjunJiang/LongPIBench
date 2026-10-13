# PII tagger on mmBERT-small

Token-level PII detector trained on NVIDIA Nemotron-PII:

| Part | Source it follows |
|---|---|
| Labels: `O` + B/I/E/S × 8 types = 33 | OpenAI Privacy Filter v2 taxonomy |
| Decoding: constrained BIES Viterbi with 6 transition biases | OpenAI Privacy Filter (`opf/_core/decoding.py`); `tests/test_viterbi.py` checks identical paths against OPF |
| Loss: token cross-entropy over the 33 labels, averaged over non-special tokens | OPF's fine-tuning code (`opf/_train/runner.py`, `_masked_token_loss_and_accuracy`) |
| Operating point: Viterbi biases, then per-type Platt scaling and F-beta thresholds, all chosen on a held-out calib split | `stage=calibrate` |
| Encoder | `jhu-clsp/mmBERT-small` (140M params, hidden 384, 8192 context) |
| Data | `nvidia/Nemotron-PII` (CC-BY-4.0), 55 source labels mapped to the 8 OPF types |

## Layout

```
conf/config.yaml                         all defaults: data, model, train, loss, decode, tune, final, predict
conf/label_map/nemotron_to_opf_v2.yaml   Nemotron -> OPF label mapping and merge rules
conf/experiment/smoke.yaml               tiny CPU run used to check the pipeline
main.py                                  Hydra entry point (stage=preprocess|tune|final|calibrate|predict)
pii_mmbert/labels.py                     BIES label space, labels -> spans
pii_mmbert/viterbi.py                    constrained Viterbi (OPF semantics), single and batched
pii_mmbert/data.py                       label mapping, tokenization, windows; Hugging Face datasets cache
pii_mmbert/model.py                      encoder + tag head, token cross-entropy
pii_mmbert/engine.py                     training loop, document log-probs, span extraction, evaluation
pii_mmbert/metrics.py                    span and token metrics
pii_mmbert/decode_search.py              Viterbi bias search on cached calib log-probs
pii_mmbert/calibration.py                Platt scaling, F-beta thresholds, ECE
pii_mmbert/stages.py                     preprocess / Ray Tune / final / calibrate / predict
tests/                                   mapping, BIES, windows, Viterbi (vs OPF, batched), bias search, calibration
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

# 4. Calibrate on the held-out calib split: Viterbi biases, then Platt + F-beta thresholds -> calibration.json
python main.py stage=calibrate
python main.py stage=calibrate calibration.beta=2.0                      # favour recall
python main.py stage=calibrate calibration.bias_search.enabled=false     # keep decode.biases

# 5. Predict (applies calibration.json when present: its biases, calibrated scores, thresholded spans)
python main.py stage=predict predict.text="Call Kim Min-jun at 010-1234-5678"

# Any stage with the tiny CPU settings
python main.py +experiment=smoke stage=tune
```

Outputs:

- `outputs/tune/best_params.json`, `summary.json`, `trials.csv`, `ray_results/`
- `outputs/final_model/`: `encoder/` (Hugging Face format + tokenizer), `heads.pt`, `pii_tagger.json`
  (labels, decode biases, window settings), `resolved_config.yaml`, `metrics.json`, and after
  `stage=calibrate` `calibration.json` (decode biases, Platt parameters, thresholds, search and test report)

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
- **Loss**: plain token cross-entropy, as in OPF's fine-tuning code. There is no class weighting by default
  (`loss.background_weight: 1.0`). Training with an "O" weight w approximately learns probabilities with
  O's mass scaled by w; after per-token normalisation (which does not change the Viterbi path) that is the
  same as adding log(w) to every transition into O. That is a decode setting, so it is chosen after training
  in `stage=calibrate` instead of being fixed into the weights. The equivalence is exact only for a model
  that fits the weighted target perfectly; `background_weight` stays available for an ablation.
- **Calibration** (`stage=calibrate`), all on the calib split, which no training stage uses:
  1. *Viterbi biases* (`pii_mmbert/decode_search.py`). The model runs once over the calib documents and
     their log-probs are kept in memory; each candidate only re-runs (batched) Viterbi. A candidate is scored
     by the best F-beta it reaches with one global threshold on raw span scores, because thresholds are
     applied afterwards: judging at threshold 0 would favour conservative biases and give up recall a
     threshold could keep. Random search over `calibration.bias_search.search_space`; trial 0 is always
     `decode.biases`, so the result never scores below it on calib. `background_stay` is not searched:
     every legal transition gets exactly one bias, so adding one constant to all six changes no path.
  2. *Platt and thresholds* (`pii_mmbert/calibration.py`), fitted on the spans decoded with the chosen biases.
     Raw span score = mean probability of the decoded labels over the span's tokens; target = the span
     exactly matches a gold span (type and character offsets). Platt scaling `p = sigmoid(a * logit(score) + b)`
     with Platt's smoothed targets, per span type when the type has at least `calibration.min_spans_per_type`
     calib spans, otherwise the global fit.
  - Threshold per type maximises F-beta. Recall's denominator is all gold spans of that type, including ones
    the decoder never proposed, so a threshold can only trade precision for recall; biases are the step that
    can raise recall.
  - A type gets its own threshold only with at least `min_spans_per_type` calib spans and at least one
    correct one. With too few spans it uses the global threshold. With enough spans but none correct,
    `calibration.zero_positive_policy` decides: `global` (default; keeps recall for PII) or `reject`
    (drop the type). The rule used per type is recorded as `threshold_rule` in `calibration.json`.
  - Biases, Platt parameters and thresholds are chosen on the same split. They are a few dozen parameters
    on ~5,000 documents, and the test set is not touched, so reported test numbers stay unbiased.
  - `calibration.json` stores the chosen biases; `stage=predict` decodes with them (falling back to the
    model's `pii_tagger.json` biases without a calibration file), so prediction uses the decoder the
    thresholds were fitted for.
  - The report has the search (baseline, best, top 5), ECE before and after Platt on calib and test, and test
    metrics three ways: `test_before` (config biases, no thresholds), `test_biases_only`, and `test_after`.
- **Storage**: preprocessed documents and training windows are Hugging Face `datasets` written with
  `save_to_disk` under `cache/nemotron_<hash>/` (`train`, `val`, `calib`, `test`,
  `windows_<split>_L<len>_S<stride>`).
  They are memory-mapped with `load_from_disk`, so training and evaluation read rows on demand, and Ray Tune
  trials receive dataset directories rather than copies of the data. `stats.json` is written last and marks
  the cache as complete; every directory is written to a `.tmp` path and renamed when done.
- **Long documents**: windows of `data.max_length` tokens with `data.stride` overlap. At evaluation, overlapping
  windows' log-probabilities are averaged per token and Viterbi runs once over the whole document. OPF uses the
  same per-token aggregation and single Viterbi pass, but its windows do not overlap (`example_to_windows` uses
  stride = window size, with `n_ctx` 128,000 by default on GPU and 4,096 on CPU). OPF can afford that because each
  token only attends to 128 tokens on either side. mmBERT has a global-attention layer every third layer, so the
  window length changes what each token sees. With mmBERT's tokenizer, 98.9% of Nemotron training documents fit
  in one 1024-token window (p50 205, p99 1034 tokens).
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
