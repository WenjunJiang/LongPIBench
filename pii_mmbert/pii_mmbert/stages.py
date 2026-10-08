"""Pipeline stages: preprocess, tune (Ray Tune), final (train + test), predict."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .calibration import Calibrator, expected_calibration_error, fit_calibrator
from .data import gold_spans, load_or_build, load_windows, windows_dir
from .engine import evaluate, iter_predictions, pick_device, predict_document, special_ids, train
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
    res = dict(cfg.tune.resources_per_trial)
    trainable = tune.with_resources(
        tune.with_parameters(_tune_trial, base_cfg=base_cfg,
                             windows_dir=str(windows_dir(cfg, data, "train")),
                             val_dir=str(data["root"] / "val")),
        {"cpu": res.get("cpu", 1), "gpu": res.get("gpu", 0)},
    )
    # Trial workers import pii_mmbert from the project directory (the package is not pip-installed).
    project_dir = str(Path(__file__).resolve().parents[1])
    pythonpath = os.pathsep.join(p for p in (project_dir, os.environ.get("PYTHONPATH")) if p)
    ray.init(ignore_reinit_error=True, include_dashboard=False,
             runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
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


def _collect_calibration_records(model, docs, space, cfg, device, bos, eos, pad, max_docs):
    """Span records [(type, raw_score, is_tp)], gold counts per type, sensitivity records [(logit, y)]."""
    span_records, sens_records = [], []
    gold_per_type = {t: 0 for t in space.span_types}
    for row, pred in iter_predictions(model, docs, space, cfg, device, bos, eos, pad, max_docs):
        gold = set(gold_spans(row))
        for t, _, _ in gold:
            gold_per_type[t] += 1
        for sp, raw in zip(pred.spans, pred.scores):
            span_records.append((sp[0], raw, int(sp in gold)))
        if pred.sens_logit != float("-inf"):
            sens_records.append((pred.sens_logit, int(len(row["sens_start"]) > 0)))
    return span_records, gold_per_type, sens_records


def run_calibrate(cfg, resolve_path) -> dict:
    """Fit Platt calibration and F-beta thresholds on the calib split of a trained model."""
    model_dir = Path(resolve_path(cfg.calibration.model_dir))
    device = pick_device(cfg.train.device)
    model, space, meta = load_model(model_dir, device)
    run_cfg = OmegaConf.merge(cfg, {"data": {"max_length": meta["max_length"], "stride": meta["stride"]}})
    data = load_or_build(cfg, resolve_path)
    bos, eos, pad = special_ids(str(model_dir / "encoder"))

    span_records, gold_per_type, sens_records = _collect_calibration_records(
        model, data["calib"], space, run_cfg, device, bos, eos, pad, cfg.calibration.max_docs)
    cal = fit_calibrator(span_records, gold_per_type, sens_records, cfg.calibration)
    raw = [r for _, r, _ in span_records]
    tp = [t for _, _, t in span_records]
    cal.report["calib_span_ece_raw"] = expected_calibration_error(raw, tp)
    cal.report["calib_span_ece_platt"] = expected_calibration_error(
        [cal.span_prob(t, r) for t, r, _ in span_records], tp)
    cal.report["calib_docs"] = len(data["calib"]) if cfg.calibration.max_docs is None else min(
        int(cfg.calibration.max_docs), len(data["calib"]))

    if cfg.calibration.eval_on_test:
        test_raw = evaluate(model, data["test"], space, run_cfg, device, bos, eos, pad)
        test_cal = evaluate(model, data["test"], space, run_cfg, device, bos, eos, pad, calibrator=cal)
        t_records, _, t_sens = _collect_calibration_records(model, data["test"], space, run_cfg, device,
                                                            bos, eos, pad, cfg.eval.max_docs)
        t_tp = [t for _, _, t in t_records]
        cal.report["test_before"] = _clean(test_raw)
        cal.report["test_after"] = _clean(test_cal)
        cal.report["test_span_ece_raw"] = expected_calibration_error([r for _, r, _ in t_records], t_tp)
        cal.report["test_span_ece_platt"] = expected_calibration_error(
            [cal.span_prob(t, r) for t, r, _ in t_records], t_tp)
        if cal.sens_platt is not None and t_sens:
            y = [lab for _, lab in t_sens]
            probs = [cal.sens_platt(s) for s, _ in t_sens]
            pred = [p >= cal.sens_threshold for p in probs]
            tp_s = sum(1 for p, l in zip(pred, y) if p and l)
            cal.report["test_sensitivity"] = {
                "precision": tp_s / max(1, sum(pred)), "recall": tp_s / max(1, sum(y)),
                "ece": expected_calibration_error(probs, y)}

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
    decoder = ViterbiDecoder(space, dict(cfg.decode.biases))
    pred = predict_document(model, text, enc["input_ids"], starts, ends, space, decoder, run_cfg,
                            device, bos, eos, pad)
    cal_path = model_dir / cfg.calibration.output_file
    result = {"text": text, "calibrated": cal_path.exists()}
    if cal_path.exists():
        cal = Calibrator.load(cal_path)
        kept = cal.keep(pred.spans, pred.scores)
        spans = [(pred.spans[i], p) for i, p in kept]
        sens = cal.sensitivity(pred.sens_logit)
        if sens is not None:
            result["sensitivity"] = {"probability": sens[0], "sensitive": bool(sens[1])}
    else:
        spans = list(zip(pred.spans, pred.scores))
    result["detected_spans"] = [{"label": t, "start": s, "end": e, "text": text[s:e], "score": round(p, 4)}
                                for (t, s, e), p in spans]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["detected_spans"]
