"""Span-level and token-level metrics on decoded predictions."""

from __future__ import annotations

from collections import Counter


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


class SpanScorer:
    """Accumulates exact typed span matches and token-level detection (PII vs O)."""

    def __init__(self, span_types):
        self.span_types = list(span_types)
        self.tp, self.fp, self.fn = Counter(), Counter(), Counter()
        self.tok = Counter()
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
        out = {
            "span_precision": micro["precision"], "span_recall": micro["recall"], "span_f1": micro["f1"],
            "token_precision": tok["precision"], "token_recall": tok["recall"], "token_f1": tok["f1"],
            "docs": self.docs,
            "fp_rate_on_docs_without_pii": (self.docs_without_gold_flagged / self.docs_without_gold
                                            if self.docs_without_gold else None),
        }
        for t in self.span_types:
            out[f"span_f1/{t}"] = _prf(self.tp[t], self.fp[t], self.fn[t])["f1"]
        return out
