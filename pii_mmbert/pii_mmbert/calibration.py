"""Platt calibration of span and sensitivity scores, and F-beta threshold selection.

Span calibration
    Input score: logit of the span's raw confidence (mean probability of the decoded labels).
    Target: 1 if the predicted span exactly matches a gold span (type and character offsets), else 0.
    p = sigmoid(a * score + b), fitted per span type when the type has enough calibration spans,
    otherwise with the parameters fitted on all spans. Fitting uses Platt's (1999) smoothed targets
    (N+ + 1) / (N+ + 2) and 1 / (N- + 2).

Threshold
    Chosen on the calibration set to maximise F-beta, where recall's denominator is the number of gold
    spans (including gold spans the decoder never proposed), so the threshold cannot overstate recall.

Sensitivity
    Input score: the document's max window sensitivity logit. Target: the document contains a span
    whose source label is in sensitive_source_labels. Same Platt fit and F-beta threshold.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


@dataclass
class Platt:
    a: float = 1.0
    b: float = 0.0

    def __call__(self, score: float) -> float:
        z = self.a * score + self.b
        if math.isnan(z):
            return 0.0
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)


def fit_platt(scores: list[float], labels: list[int]) -> Platt:
    """Fit sigmoid(a * s + b) to binary labels with Platt's smoothed targets (LBFGS on log loss)."""
    if not scores:
        return Platt()
    s = torch.tensor(scores, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    target = torch.where(y > 0.5, torch.tensor((n_pos + 1) / (n_pos + 2), dtype=torch.float64),
                         torch.tensor(1 / (n_neg + 2), dtype=torch.float64))
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a * s + b, target)
        loss.backward()
        return loss

    opt.step(closure)
    return Platt(float(a.detach()), float(b.detach()))


def best_threshold(probs: list[float], is_tp: list[int], n_gold: int, beta: float) -> dict:
    """Threshold on calibrated probabilities maximising F-beta.

    Spans with prob >= threshold are kept. recall = kept true positives / n_gold.
    """
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f_beta": 0.0, "kept": 0}
    b2 = beta * beta
    tp = 0
    for k, i in enumerate(order, start=1):
        tp += is_tp[i]
        if k < len(order) and probs[order[k]] == probs[i]:
            continue  # only cut between distinct probabilities
        prec = tp / k
        rec = tp / n_gold if n_gold else 0.0
        f = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0
        if f > best["f_beta"]:
            best = {"threshold": probs[i], "precision": prec, "recall": rec, "f_beta": f, "kept": k}
    return best


def expected_calibration_error(probs: list[float], labels: list[int], bins: int = 10) -> float:
    if not probs:
        return 0.0
    total, err = len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
    return err


@dataclass
class Calibrator:
    span_platt: dict = field(default_factory=dict)        # type -> Platt (or "__all__")
    span_threshold: dict = field(default_factory=dict)    # type -> threshold (or "__all__")
    sens_platt: Platt | None = None
    sens_threshold: float | None = None
    report: dict = field(default_factory=dict)

    def span_prob(self, span_type: str, raw_score: float) -> float:
        platt = self.span_platt.get(span_type, self.span_platt.get("__all__", Platt()))
        return platt(logit(raw_score))

    def keep(self, spans, raw_scores) -> list[tuple[int, float]]:
        """(index, calibrated prob) of spans at or above their type's threshold."""
        out = []
        for i, ((typ, _, _), raw) in enumerate(zip(spans, raw_scores)):
            p = self.span_prob(typ, raw)
            thr = self.span_threshold.get(typ, self.span_threshold.get("__all__", 0.0))
            if p >= thr:
                out.append((i, p))
        return out

    def sensitivity(self, sens_logit: float) -> tuple[float, bool] | None:
        if self.sens_platt is None:
            return None
        if not math.isfinite(sens_logit):
            return 0.0, False
        p = self.sens_platt(sens_logit)
        return p, p >= (self.sens_threshold or 0.5)

    def save(self, path: Path) -> None:
        payload = {
            "span_platt": {k: asdict(v) for k, v in self.span_platt.items()},
            "span_threshold": self.span_threshold,
            "sens_platt": asdict(self.sens_platt) if self.sens_platt else None,
            "sens_threshold": self.sens_threshold,
            "report": self.report,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        d = json.loads(path.read_text())
        return cls(span_platt={k: Platt(**v) for k, v in d["span_platt"].items()},
                   span_threshold=d["span_threshold"],
                   sens_platt=Platt(**d["sens_platt"]) if d["sens_platt"] else None,
                   sens_threshold=d["sens_threshold"], report=d.get("report", {}))


def fit_calibrator(span_records, gold_per_type: dict, sens_records, cfg_cal) -> Calibrator:
    """span_records: [(type, raw_score, is_tp)]; gold_per_type: {type: n_gold};
    sens_records: [(sens_logit, label)] or empty."""
    beta, min_n = float(cfg_cal.beta), int(cfg_cal.min_spans_per_type)
    cal = Calibrator()
    all_scores = [logit(r) for _, r, _ in span_records]
    all_tp = [int(t) for _, _, t in span_records]
    cal.span_platt["__all__"] = fit_platt(all_scores, all_tp)
    all_probs = [cal.span_platt["__all__"](s) for s in all_scores]
    overall = best_threshold(all_probs, all_tp, sum(gold_per_type.values()), beta)
    cal.span_threshold["__all__"] = overall["threshold"]
    cal.report["spans"] = {"__all__": {"n": len(all_tp), "positives": sum(all_tp), **overall}}

    if cfg_cal.per_type:
        for typ, n_gold in gold_per_type.items():
            idx = [i for i, (t, _, _) in enumerate(span_records) if t == typ]
            entry = {"n": len(idx), "positives": sum(all_tp[i] for i in idx)}
            if len(idx) >= min_n and 0 < entry["positives"] < len(idx):
                cal.span_platt[typ] = fit_platt([all_scores[i] for i in idx], [all_tp[i] for i in idx])
                entry["platt"] = "per_type"
            else:
                entry["platt"] = "global"
            probs = [cal.span_prob(typ, span_records[i][1]) for i in idx]
            best = best_threshold(probs, [all_tp[i] for i in idx], n_gold, beta)
            cal.span_threshold[typ] = best["threshold"]
            cal.report["spans"][typ] = {**entry, **best}

    if sens_records and cfg_cal.sensitivity:
        scores = [s for s, _ in sens_records]
        labels = [int(y) for _, y in sens_records]
        cal.sens_platt = fit_platt(scores, labels)
        probs = [cal.sens_platt(s) for s in scores]
        best = best_threshold(probs, labels, sum(labels), beta)
        cal.sens_threshold = best["threshold"]
        cal.report["sensitivity"] = {"n": len(labels), "positives": sum(labels), **best,
                                     "ece": expected_calibration_error(probs, labels)}
    return cal
