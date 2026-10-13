"""Pipeline stages: preprocess, tune (Ray Tune), final (train + test), calibrate, predict."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .calibration import Calibrator, expected_calibration_error, fit_calibrator
from .data import gold_spans, load_or_build, load_windows, subset, windows_dir
from .decode_search import decode_cache, search_biases
from .engine import (doc_log_probs, evaluate, pick_device, predict_document, score_prediction, special_ids,
                     spans_from_path, train)
from .metrics import SpanScorer
from .model import load_model, save_model
from .viterbi import ViterbiDecoder

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- helpers

def build_search_space(entries) -> dict:
    """Turn the `tune.search_space` list from the config into Ray Tune sampling objects."""
    from ray import tune
    space = {}
    for e in entries:
        kind = e["type"]
        if kind == "loguniform":
            space[e["param"]] = tune.loguniform(e["lower"], e["upper"])
        elif kind == "uniform":
            space[e["param"]] = tune.uniform(e["lower"], e["upper"])
        elif kind == "choice":
            space[e["param"]] = tune.choice(list(e["values"]))
        elif kind == "randint":
            space[e["param"]] = tune.randint(e["lower"], e["upper"])
        else:
            raise ValueError(f"unsupported search space type {kind!r} for {e['param']}")
    return space


def apply_params(cfg: DictConfig, params: dict) -> DictConfig:
    """Return a copy of cfg with dotted-path params applied (keys must already exist)."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in params.items():
        if OmegaConf.select(cfg, key, default="__missing__") == "__missing__":
            raise KeyError(f"parameter {key!r} does not exist in the config")
        OmegaConf.update(cfg, key, value, merge=False)
    return cfg


def trial_resources(tune_cfg, cluster_resources: dict) -> dict:
    """Resource request of one trial: cpu and gpu from tune.resources_per_trial, plus the custom Ray
    resources in tune.resources_per_trial.custom exactly as configured (e.g. {"accelerator_type:RTX": 1})."""
    cfg_res = tune_cfg.resources_per_trial
    res = {"cpu": float(cfg_res.get("cpu", 1)), "gpu": float(cfg_res.get("gpu", 0))}
    for name, amount in (cfg_res.get("custom") or {}).items():
        res[str(name)] = float(amount)
        if name not in cluster_resources:
            log.warning("custom resource %r is not in this Ray cluster (%s); trials will stay PENDING",
                        name, sorted(cluster_resources))
    return res


def _clean(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if v is not None}


# ----------------------------------------------------------------------------- stages

def run_preprocess(cfg, resolve_path) -> dict:
    data = load_or_build(cfg, resolve_path)
    log.info("data stats:\n%s", json.dumps(data["stats"], indent=2))
    return data["stats"]


def _tune_trial(params: dict, base_cfg: dict, windows_dir: str, val_dir: str):
    """One Ray Tune trial. Receives dataset directories only and memory-maps them, so trials share
    the page cache instead of each holding a copy of the data."""
    from datasets import load_from_disk
    from ray import tune
    from .labels import LabelSpace
    cfg = apply_params(OmegaConf.create(base_cfg), params)
    space = LabelSpace(tuple(cfg.label_map.target_types))
    train(cfg, load_from_disk(windows_dir), space, eval_docs=load_from_disk(val_dir), eval_prefix="val",
          report=lambda m: tune.report(_clean(m)))


def run_tune(cfg, resolve_path) -> dict:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search import BasicVariantGenerator

    data = load_or_build(cfg, resolve_path)
    bos, eos, _ = special_ids(cfg.model.name)
    load_windows(cfg, data, ["train"], bos, eos)        # build once in the driver; trials memory-map it
    out_dir = Path(resolve_path(cfg.paths.output_dir)) / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg.tune.scheduler.type != "asha":
        raise ValueError(f"unsupported scheduler {cfg.tune.scheduler.type!r}")
    scheduler = ASHAScheduler(time_attr="epoch", max_t=int(cfg.train.epochs),
                              grace_period=int(cfg.tune.scheduler.grace_period),
                              reduction_factor=int(cfg.tune.scheduler.reduction_factor))
    # Trial workers import pii_mmbert from the project directory (the package is not pip-installed).
    project_dir = str(Path(__file__).resolve().parents[1])
    pythonpath = os.pathsep.join(p for p in (project_dir, os.environ.get("PYTHONPATH")) if p)
    ray.init(ignore_reinit_error=True, include_dashboard=False,
             runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
    resources = trial_resources(cfg.tune, ray.cluster_resources())
    log.info("resources per trial: %s, max concurrent trials: %s", resources, cfg.tune.max_concurrent_trials)
    trainable = tune.with_resources(
        tune.with_parameters(_tune_trial, base_cfg=base_cfg,
                             windows_dir=str(windows_dir(cfg, data, "train")),
                             val_dir=str(data["root"] / "val")),
        resources,
    )
    tuner = tune.Tuner(
        trainable,
        param_space=build_search_space(OmegaConf.to_container(cfg.tune.search_space, resolve=True)),
        tune_config=tune.TuneConfig(metric=cfg.tune.metric, mode=cfg.tune.mode,
                                    num_samples=int(cfg.tune.num_samples), scheduler=scheduler,
                                    search_alg=BasicVariantGenerator(random_state=int(cfg.seed)),
                                    max_concurrent_trials=cfg.tune.max_concurrent_trials),
        run_config=tune.RunConfig(name=cfg.tune.name, storage_path=str(out_dir / "ray_results")),
    )
    results = tuner.fit()
    best = results.get_best_result(metric=cfg.tune.metric, mode=cfg.tune.mode, scope="all")
    best_params = dict(best.config)
    summary = {"metric": cfg.tune.metric, "mode": cfg.tune.mode,
               "best_value": best.metrics.get(cfg.tune.metric), "best_params": best_params}
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    results.get_dataframe().to_csv(out_dir / "trials.csv", index=False)
    log.info("best %s = %s with %s", cfg.tune.metric, summary["best_value"], best_params)
    log.info("wrote %s", out_dir / "best_params.json")
    return summary


def resolve_final_cfg(cfg: DictConfig, cli_overrides: list[str], resolve_path) -> DictConfig:
    """Config defaults < tuned best params < command-line overrides."""
    if cfg.final.best_params_path:
        best = json.loads(Path(resolve_path(cfg.final.best_params_path)).read_text())
        cfg = apply_params(cfg, best)
        log.info("applied tuned params from %s: %s", cfg.final.best_params_path, best)
        if cli_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(
                [o for o in cli_overrides if "=" in o and not o.startswith(("+", "~", "hydra"))]))
            log.info("re-applied command-line overrides on top: %s", cli_overrides)
    return cfg


def run_final(cfg: DictConfig, cli_overrides: list[str], resolve_path) -> dict:
    cfg = resolve_final_cfg(cfg, cli_overrides, resolve_path)
    data = load_or_build(cfg, resolve_path)
    bos, eos, pad = special_ids(cfg.model.name)
    splits = ["train", "val"] if cfg.final.include_val_in_train else ["train"]
    windows = load_windows(cfg, data, splits, bos, eos)
    eval_docs = None if cfg.final.include_val_in_train else data["val"]
    model, metrics = train(cfg, windows, data["space"], eval_docs=eval_docs, eval_prefix="val")

    out_dir = Path(resolve_path(cfg.paths.output_dir)) / cfg.final.output_name
    if cfg.final.eval_on_test:
        device = pick_device(cfg.train.device)
        test = evaluate(model, data["test"], data["space"], cfg, device, bos, eos, pad)
        metrics.update({f"test_{k}": v for k, v in test.items()})
    from transformers import AutoTokenizer
    save_model(model, AutoTokenizer.from_pretrained(cfg.model.name), data["space"], out_dir,
               meta={"base_model": cfg.model.name,
                     "decode_biases": OmegaConf.to_container(cfg.decode.biases),
                     "max_length": cfg.data.max_length, "stride": cfg.data.stride})
    (out_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2))
    log.info("final metrics: %s", json.dumps(_clean(metrics), indent=2))
    log.info("saved model to %s", out_dir)
    return metrics


def _log_prob_cache(model, docs, space, cfg, device, bos, eos, pad, max_docs) -> list[dict]:
    """Run the model once per document and keep what decoding and scoring need."""
    docs = subset(docs, max_docs, int(cfg.seed))
    return [{"text": row["text"], "offset_start": row["offset_start"], "offset_end": row["offset_end"],
             "log_probs": doc_log_probs(model, row["input_ids"], space, cfg, device, bos, eos, pad),
             "gold": set(gold_spans(row))}
            for row in docs]


def run_calibrate(cfg, resolve_path) -> dict:
    """On the calib split of a trained model: choose Viterbi biases, then fit Platt calibration and
    F-beta thresholds on the spans decoded with those biases. Both go into calibration.json."""
    model_dir = Path(resolve_path(cfg.calibration.model_dir))
    device = pick_device(cfg.train.device)
    model, space, meta = load_model(model_dir, device)
    run_cfg = OmegaConf.merge(cfg, {"data": {"max_length": meta["max_length"], "stride": meta["stride"]}})
    data = load_or_build(cfg, resolve_path)
    bos, eos, pad = special_ids(str(model_dir / "encoder"))
    trim = bool(cfg.decode.trim_whitespace)
    beta = float(cfg.calibration.beta)
    base_biases = OmegaConf.to_container(cfg.decode.biases)

    # 1. Viterbi biases, scored with a global threshold in the loop.
    cache = _log_prob_cache(model, data["calib"], space, run_cfg, device, bos, eos, pad,
                            cfg.calibration.max_docs)
    if cfg.calibration.bias_search.enabled:
        biases, search_report = search_biases(cache, space, base_biases, cfg.calibration.bias_search, beta,
                                              trim, int(cfg.seed))
    else:
        biases, search_report = base_biases, {"enabled": False}
    log.info("decode biases: %s", biases)

    # 2. Platt scaling and thresholds on the spans those biases produce.
    span_records, gold_per_type = [], {t: 0 for t in space.span_types}
    for pred, c in zip(decode_cache(cache, space, biases, trim), cache):
        for t, _, _ in c["gold"]:
            gold_per_type[t] += 1
        for sp, raw in zip(pred.spans, pred.scores):
            span_records.append((sp[0], raw, int(sp in c["gold"])))
    cal = fit_calibrator(span_records, gold_per_type, cfg.calibration, decode_biases=biases)
    tp = [t for _, _, t in span_records]
    cal.report.update({
        "calib_docs": len(cache),
        "bias_search": search_report,
        "calib_span_ece_raw": expected_calibration_error([r for _, r, _ in span_records], tp),
        "calib_span_ece_platt": expected_calibration_error([cal.span_prob(t, r) for t, r, _ in span_records], tp),
    })
    del cache

    # 3. Test: config biases (before) -> chosen biases -> chosen biases + Platt/thresholds (after).
    if cfg.calibration.eval_on_test:
        test = subset(data["test"], cfg.eval.max_docs, int(cfg.seed))
        base_dec, chosen_dec = ViterbiDecoder(space, base_biases), ViterbiDecoder(space, biases)
        before, with_biases, after = (SpanScorer(space.span_types) for _ in range(3))
        t_records = []
        for row in test:
            lp = doc_log_probs(model, row["input_ids"], space, run_cfg, device, bos, eos, pad)
            args = (lp, row["text"], row["offset_start"], row["offset_end"], space, trim)
            p_base = spans_from_path(base_dec.decode(lp), *args)
            p_new = p_base if biases == base_biases else spans_from_path(chosen_dec.decode(lp), *args)
            score_prediction(before, row, p_base)
            score_prediction(with_biases, row, p_new)
            score_prediction(after, row, p_new, calibrator=cal)
            gold = set(gold_spans(row))
            t_records += [(sp[0], raw, int(sp in gold)) for sp, raw in zip(p_new.spans, p_new.scores)]
        t_tp = [t for _, _, t in t_records]
        cal.report.update({
            "test_before": _clean(before.result()),
            "test_biases_only": _clean(with_biases.result()),
            "test_after": _clean(after.result()),
            "test_span_ece_raw": expected_calibration_error([r for _, r, _ in t_records], t_tp),
            "test_span_ece_platt": expected_calibration_error([cal.span_prob(t, r) for t, r, _ in t_records], t_tp),
        })

    out = model_dir / cfg.calibration.output_file
    cal.save(out)
    log.info("calibration report:\n%s", json.dumps(cal.report, indent=2))
    log.info("wrote %s", out)
    return cal.report


def run_predict(cfg, resolve_path) -> list[dict]:
    from transformers import AutoTokenizer
    model_dir = Path(resolve_path(cfg.predict.model_dir))
    device = pick_device(cfg.train.device)
    model, space, meta = load_model(model_dir, device)
    tok = AutoTokenizer.from_pretrained(model_dir / "encoder")
    text = cfg.predict.text
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [a for a, _ in enc["offset_mapping"]]
    ends = [b for _, b in enc["offset_mapping"]]
    bos, eos, pad = special_ids(str(model_dir / "encoder"))
    run_cfg = OmegaConf.merge(cfg, {"data": {"max_length": meta["max_length"], "stride": meta["stride"]}})
    # The decoder must match the one calibration was fitted with: biases come from calibration.json
    # when present, otherwise from the model's own metadata (decode.biases at training time).
    cal_path = model_dir / cfg.calibration.output_file
    cal = Calibrator.load(cal_path) if cal_path.exists() else None
    biases = cal.decode_biases if cal is not None and cal.decode_biases else meta["decode_biases"]
    pred = predict_document(model, text, enc["input_ids"], starts, ends, space, ViterbiDecoder(space, biases),
                            run_cfg, device, bos, eos, pad)
    result = {"text": text, "calibrated": cal is not None}
    if cal is not None:
        spans = [(pred.spans[i], p) for i, p in cal.keep(pred.spans, pred.scores)]
    else:
        spans = list(zip(pred.spans, pred.scores))
    result["detected_spans"] = [{"label": t, "start": s, "end": e, "text": text[s:e], "score": round(p, 4)}
                                for (t, s, e), p in spans]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["detected_spans"]
