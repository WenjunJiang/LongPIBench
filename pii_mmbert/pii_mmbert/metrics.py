"""Span-level and token-level metrics on decoded predictions."""

from __future__ import annotations

from collections import Counter


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _overlaps(a, b) -> bool:
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


class SpanScorer:
    """Accumulates typed span matches and token-level detection (PII vs O).

    Exact match (primary): same type and identical character offsets.
    Overlap match: same type and at least one shared character. Precision counts predicted spans that
    overlap a gold span, recall counts gold spans overlapped by a prediction. It separates boundary
    conventions (e.g. whether "+1" belongs to a phone number) from spans that were missed or invented.
    """

    def __init__(self, span_types):
        self.span_types = list(span_types)
        self.tp, self.fp, self.fn = Counter(), Counter(), Counter()
        self.tok = Counter()
        self.ov = Counter()
        self.docs = 0
        self.docs_without_gold = 0
        self.docs_without_gold_flagged = 0

    def add(self, gold_spans, pred_spans, gold_token_labels, pred_token_labels):
        self.docs += 1
        gold, pred = set(gold_spans), set(pred_spans)
        for t, *_ in gold & pred:
            self.tp[t] += 1
        for t, *_ in pred - gold:
            self.fp[t] += 1
        for t, *_ in gold - pred:
            self.fn[t] += 1
        self.ov["pred"] += len(pred)
        self.ov["gold"] += len(gold)
        self.ov["pred_hit"] += sum(any(_overlaps(p, g) for g in gold) for p in pred)
        self.ov["gold_hit"] += sum(any(_overlaps(g, p) for p in pred) for g in gold)
        for g, p in zip(gold_token_labels, pred_token_labels):
            if g and p:
                self.tok["tp"] += 1
            elif p:
                self.tok["fp"] += 1
            elif g:
                self.tok["fn"] += 1
        if not gold_spans:
            self.docs_without_gold += 1
            self.docs_without_gold_flagged += bool(pred_spans)

    def result(self) -> dict:
        micro = _prf(sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values()))
        tok = _prf(self.tok["tp"], self.tok["fp"], self.tok["fn"])
        op = self.ov["pred_hit"] / self.ov["pred"] if self.ov["pred"] else 0.0
        orr = self.ov["gold_hit"] / self.ov["gold"] if self.ov["gold"] else 0.0
        out = {
            "span_precision": micro["precision"], "span_recall": micro["recall"], "span_f1": micro["f1"],
            "span_overlap_precision": op, "span_overlap_recall": orr,
            "span_overlap_f1": 2 * op * orr / (op + orr) if op + orr else 0.0,
            "token_precision": tok["precision"], "token_recall": tok["recall"], "token_f1": tok["f1"],
            "docs": self.docs,
            "fp_rate_on_docs_without_pii": (self.docs_without_gold_flagged / self.docs_without_gold
                                            if self.docs_without_gold else None),
        }
        for t in self.span_types:
            prf = _prf(self.tp[t], self.fp[t], self.fn[t])
            out[f"span_f1/{t}"] = prf["f1"]
            out[f"span_precision/{t}"] = prf["precision"]
            out[f"span_recall/{t}"] = prf["recall"]
            out[f"span_gold/{t}"] = self.tp[t] + self.fn[t]
            out[f"span_pred/{t}"] = self.tp[t] + self.fp[t]
        return out
