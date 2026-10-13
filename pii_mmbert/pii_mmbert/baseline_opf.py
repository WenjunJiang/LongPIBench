"""OpenAI Privacy Filter as a baseline on the same test documents, gold spans and metrics.

OPF (https://github.com/openai/privacy-filter) is an optional dependency, needed only for this stage.
It predicts the same eight span types, so its character spans are scored directly against our mapped
gold spans; for token metrics they are projected onto mmBERT tokens with the rule used for the gold labels.

OPF runs with its own defaults (constrained Viterbi with its shipped operating point, whitespace trimming,
context window from the checkpoint). Predictions are appended to a JSONL file as they are produced, so an
interrupted run resumes where it stopped.

OPF is slow on CPU (~16 s per test document on 4 vCPUs); run this stage on a GPU.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .data import gold_spans, load_or_build, subset, token_bies_labels
from .metrics import SpanScorer

log = logging.getLogger(__name__)


def _read_done(pred_path: Path) -> dict[int, dict]:
    done = {}
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["row"]] = rec
    return done


def run_opf_baseline(cfg, resolve_path) -> dict:
    try:
        from opf import OPF
    except ImportError as exc:
        raise ImportError("stage=baseline_opf needs OPF: pip install git+https://github.com/openai/privacy-filter"
                          ) from exc
    bcfg = cfg.baseline.opf
    data = load_or_build(cfg, resolve_path)
    space = data["space"]
    docs = subset(data["test"], cfg.eval.max_docs, int(cfg.seed))
    device = bcfg.device if bcfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        log.warning("OPF on CPU takes ~16 s per document; use a GPU for the full test set")

    out_dir = Path(resolve_path(cfg.paths.output_dir)) / bcfg.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_info = {"docs": len(docs), "eval_max_docs": cfg.eval.max_docs, "seed": int(cfg.seed),
                "data_root": str(data["root"].name), "device": device,
                "checkpoint": bcfg.checkpoint, "context_window_length": bcfg.context_window_length}
    info_path, pred_path = out_dir / "run.json", out_dir / "predictions.jsonl"
    if info_path.exists() and pred_path.exists():
        previous = json.loads(info_path.read_text())
        if previous != run_info:
            raise ValueError(f"{out_dir} holds predictions from a different run ({previous}); "
                             "delete it or set baseline.opf.output_name")
    info_path.write_text(json.dumps(run_info, indent=2))
    done = _read_done(pred_path)
    log.info("OPF baseline on %d test documents (%d already predicted), device=%s", len(docs), len(done),
             device)

    kwargs = {"device": device}
    if bcfg.checkpoint:
        kwargs["model"] = str(resolve_path(bcfg.checkpoint))
    if bcfg.context_window_length:
        kwargs["context_window_length"] = int(bcfg.context_window_length)
    redactor = OPF(**kwargs)

    scorer = SpanScorer(space.span_types)
    seconds = [r["seconds"] for r in done.values()]
    t_start = time.time()
    with pred_path.open("a") as fout:
        for i, row in enumerate(docs):
            rec = done.get(i)
            if rec is not None and rec["uid"] != row["uid"]:
                raise ValueError(f"{pred_path} row {i} is uid {rec['uid']}, expected {row['uid']}")
            if rec is None:
                t0 = time.time()
                result = redactor.redact(row["text"]).to_dict()
                rec = {"row": i, "uid": row["uid"], "seconds": time.time() - t0,
                       "spans": [[s["label"], int(s["start"]), int(s["end"])] for s in result["detected_spans"]]}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                seconds.append(rec["seconds"])
                new = len(seconds) - len(done)
                if new % 100 == 0:
                    rate = (time.time() - t_start) / new
                    log.info("%d/%d documents, %.2f s/doc, ~%.0f min left", i + 1, len(docs), rate,
                             rate * (len(docs) - i - 1) / 60)
            pred = [(lab, s, e) for lab, s, e in rec["spans"]]
            unknown = {lab for lab, _, _ in pred} - set(space.span_types)
            if unknown:
                raise ValueError(f"OPF predicted labels outside the target types: {sorted(unknown)}")
            offsets = np.stack([np.asarray(row["offset_start"]), np.asarray(row["offset_end"])], axis=1)
            scorer.add(gold_spans(row), pred, row["token_labels"], token_bies_labels(offsets, pred, space))

    metrics = {k: v for k, v in scorer.result().items() if v is not None}
    metrics["seconds_per_doc"] = float(np.mean(seconds)) if seconds else None
    metrics["run"] = run_info
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("OPF baseline metrics:\n%s", json.dumps(metrics, indent=2))
    log.info("wrote %s", out_dir / "metrics.json")
    return metrics
