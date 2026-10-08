import math
import random

from omegaconf import OmegaConf

from pii_mmbert.calibration import (Calibrator, Platt, best_threshold, expected_calibration_error,
                                    fit_calibrator, fit_platt, logit)


def test_fit_platt_recovers_known_sigmoid():
    rng = random.Random(0)
    true = Platt(a=2.0, b=-1.0)
    scores = [rng.uniform(-4, 4) for _ in range(20000)]
    labels = [int(rng.random() < true(s)) for s in scores]
    fit = fit_platt(scores, labels)
    assert abs(fit.a - 2.0) < 0.15 and abs(fit.b + 1.0) < 0.15


def test_best_threshold_counts_missed_gold_in_recall():
    probs = [0.9, 0.8, 0.4, 0.2]
    is_tp = [1, 1, 0, 1]
    best = best_threshold(probs, is_tp, n_gold=5, beta=1.0)   # 5 gold, decoder found 3 of them
    # keep top-2: P=1, R=2/5 -> F=0.571; keep all 4: P=3/4, R=3/5 -> F=0.667 (best)
    assert best["kept"] == 4 and math.isclose(best["recall"], 0.6) and best["threshold"] == 0.2
    assert best_threshold(probs, is_tp, n_gold=5, beta=0.5)["kept"] == 2   # precision-weighted


def test_threshold_only_cuts_between_distinct_probs():
    best = best_threshold([0.5, 0.5, 0.1], [1, 0, 0], n_gold=1, beta=1.0)
    assert best["kept"] == 2   # cannot keep just one of the tied 0.5 spans


def test_ece_perfect_and_bad():
    assert expected_calibration_error([0.0, 1.0], [0, 1]) == 0.0
    assert math.isclose(expected_calibration_error([0.9, 0.9], [0, 0]), 0.9)


def test_calibrator_per_type_fallback_and_roundtrip(tmp_path):
    rng = random.Random(1)
    records = []
    for _ in range(400):                     # plenty of person spans, mixed correctness
        raw = rng.uniform(0.3, 0.99)
        records.append(("private_person", raw, int(rng.random() < raw)))
    for _ in range(5):                       # too few secret spans -> global Platt
        records.append(("secret", 0.9, 1))
    cfg = OmegaConf.create({"beta": 1.0, "min_spans_per_type": 50, "per_type": True, "sensitivity": True})
    sens = [(rng.uniform(-3, 3), 0) for _ in range(50)] + [(rng.uniform(0, 5), 1) for _ in range(50)]
    cal = fit_calibrator(records, {"private_person": 420, "secret": 5}, sens, cfg)
    assert "private_person" in cal.span_platt and "secret" not in cal.span_platt
    assert cal.report["spans"]["secret"]["platt"] == "global"
    assert cal.sens_platt is not None and 0.0 <= cal.sens_threshold <= 1.0
    path = tmp_path / "calibration.json"
    cal.save(path)
    again = Calibrator.load(path)
    spans = [("private_person", 0, 4), ("secret", 5, 9)]
    assert again.keep(spans, [0.95, 0.9]) == cal.keep(spans, [0.95, 0.9])
    assert again.sensitivity(float("-inf")) == (0.0, False)
    assert math.isclose(again.span_prob("secret", 0.9), cal.span_platt["__all__"](logit(0.9)))
