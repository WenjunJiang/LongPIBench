import math

from pii_mmbert.metrics import SpanScorer


def test_exact_and_overlap_match():
    sc = SpanScorer(["private_phone", "private_date"])
    gold = [("private_phone", 10, 22), ("private_date", 30, 40)]
    pred = [("private_phone", 7, 22), ("private_date", 50, 55)]   # phone boundary differs, date invented
    sc.add(gold, pred, [0] * 5, [0] * 5)
    r = sc.result()
    assert r["span_f1"] == 0.0
    assert r["span_overlap_precision"] == 0.5 and r["span_overlap_recall"] == 0.5
    assert math.isclose(r["span_overlap_f1"], 0.5)
    assert r["span_gold/private_date"] == 1 and r["span_pred/private_date"] == 1
