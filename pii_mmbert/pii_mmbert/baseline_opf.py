"""OpenAI Privacy Filter as a baseline on the same test documents, gold spans and metrics.

OPF (https://github.com/openai/privacy-filter) is an optional dependency, needed only for this stage.
It predicts the same eight span types, so its character spans are scored directly against our mapped
gold spans; for token metrics they are projected onto mmBERT tokens with the rule used for the gold labels.

OPF runs with its own defaults (constrained Viterbi with its shipped operating point, whitespace trimming,
context window from the checkpoint). Predictions are appended to a JSONL file as they are produced, so an
interrupted run resumes where it stopped.

On CPU, OPF's mixture-of-experts layer copies the weights of each token's experts per token, which makes
it ~20x slower than necessary. With baseline.opf.cpu_grouped_moe=true the layer is replaced by one that
applies each expert once to all tokens routed to it (the same arithmetic, grouped like OPF's GPU kernel).
The residual stream is bf16, so the different summation order changes probabilities slightly (mean
|dp| ~1e-5 on 12 test documents, no argmax change); compare on a sample with cpu_grouped_moe=false.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import gold_spans, load_or_build, subset, token_bies_labels
from .metrics import SpanScorer

log = logging.getLogger(__name__)


def _grouped_moe_forward(self, x):
    """Drop-in for opf._model.model.MLPBlock.forward (torch path): same routing, weights and SwiGLU,
    but each expert is applied once to all of its tokens instead of gathering weights per token."""
    from opf._model.model import swiglu
    batch_shape = x.shape[:-1]
    t = self.norm(x).reshape(-1, x.shape[-1])
    g = F.linear(t.float(), self.gate.weight.float(), self.gate.bias.float())
    experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
    weights = torch.softmax(experts.values, dim=1) / self.experts_per_token
    idx = experts.indices
    n, k = idx.shape
    tf = t.float()
    out = torch.zeros((n, k, t.shape[1]), dtype=torch.float32, device=t.device)
    flat = idx.reshape(-1)
    for e in torch.unique(flat).tolist():
        pos = (flat == e).nonzero().squeeze(1)
        tok, slot = pos // k, pos % k
        h = tf[tok] @ self.mlp1_weight[e].float() + self.mlp1_bias[e].float()
        h = swiglu(h, limit=self.swiglu_limit, packed=self.packed_geglu)
        out[tok, slot] = h @ self.mlp2_weight[e].float() + self.mlp2_bias[e].float()
    out = torch.einsum("bec,be->bc", out, weights.to(out.dtype)) * self.experts_per_token
    return x + out.to(x.dtype).reshape(*batch_shape, -1)


@contextlib.contextmanager
def _grouped_moe(enabled: bool):
    from opf._model import model as opf_model
    original = opf_model.MLPBlock.forward
    if enabled:
        opf_model.MLPBlock.forward = _grouped_moe_forward
    try:
        yield
    finally:
        opf_model.MLPBlock.forward = original


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
    grouped = bool(bcfg.cpu_grouped_moe) and device == "cpu"

    out_dir = Path(resolve_path(cfg.paths.output_dir)) / bcfg.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_info = {"docs": len(docs), "eval_max_docs": cfg.eval.max_docs, "seed": int(cfg.seed),
                "data_root": str(data["root"].name), "device": device, "cpu_grouped_moe": grouped,
                "checkpoint": bcfg.checkpoint, "context_window_length": bcfg.context_window_length}
    info_path, pred_path = out_dir / "run.json", out_dir / "predictions.jsonl"
    if info_path.exists() and pred_path.exists():
        previous = json.loads(info_path.read_text())
        if previous != run_info:
            raise ValueError(f"{out_dir} holds predictions from a different run ({previous}); "
                             "delete it or set baseline.opf.output_name")
    info_path.write_text(json.dumps(run_info, indent=2))
    done = _read_done(pred_path)
    log.info("OPF baseline on %d test documents (%d already predicted), device=%s, grouped MoE=%s",
             len(docs), len(done), device, grouped)

    kwargs = {"device": device}
    if bcfg.checkpoint:
        kwargs["model"] = str(resolve_path(bcfg.checkpoint))
    if bcfg.context_window_length:
        kwargs["context_window_length"] = int(bcfg.context_window_length)
    redactor = OPF(**kwargs)

    scorer = SpanScorer(space.span_types)
    seconds = [r["seconds"] for r in done.values()]
    t_start = time.time()
    with _grouped_moe(grouped), pred_path.open("a") as fout:
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
