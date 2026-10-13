"""Choosing Viterbi transition biases on the calib split (first step of stage=calibrate).

The model runs once over the calib documents; their per-token log-probs are kept in memory and every
candidate bias setting only re-runs Viterbi (batched). A candidate is scored by the best F-beta it
reaches with one global threshold on raw span scores. Thresholds are applied after decoding, so a
setting that proposes more spans should be judged after its low-confidence false positives are cut,
not at threshold 0; judging at threshold 0 would favour conservative biases and give up recall that
a threshold could have kept.

All six biases are added to legal transitions only, and every legal transition gets exactly one of
them, so adding the same constant to all six changes no path. The default search space therefore
leaves transition_bias_background_stay at its decode.biases value (5 free parameters).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .calibration import best_threshold
from .engine import DocPrediction, spans_from_path
from .labels import LabelSpace
from .viterbi import BIAS_KEYS, ViterbiDecoder

log = logging.getLogger(__name__)


def sample_params(entries, rng: np.random.Generator) -> dict:
    """One random draw from a search-space list ({param, type, lower/upper or values})."""
    out = {}
    for e in entries:
        kind = e["type"]
        if kind == "uniform":
            out[e["param"]] = float(rng.uniform(e["lower"], e["upper"]))
        elif kind == "loguniform":
            out[e["param"]] = float(math.exp(rng.uniform(math.log(e["lower"]), math.log(e["upper"]))))
        elif kind == "choice":
            values = list(e["values"])
            out[e["param"]] = values[int(rng.integers(len(values)))]
        elif kind == "randint":
            out[e["param"]] = int(rng.integers(e["lower"], e["upper"]))
        else:
            raise ValueError(f"unsupported search space type {kind!r} for {e['param']}")
    return out


def decode_cache(cache: list[dict], space: LabelSpace, biases: dict, trim_whitespace: bool
                 ) -> list[DocPrediction]:
    """Decode cached documents ({text, offset_start, offset_end, log_probs, gold}) with given biases."""
    paths = ViterbiDecoder(space, biases).decode_batch([c["log_probs"] for c in cache])
    return [spans_from_path(path, c["log_probs"], c["text"], c["offset_start"], c["offset_end"], space,
                            trim_whitespace)
            for path, c in zip(paths, cache)]


def score_biases(cache: list[dict], space: LabelSpace, biases: dict, beta: float, trim_whitespace: bool
                 ) -> dict:
    """Best F-beta over one global threshold, plus the recall ceiling (all proposed spans kept)."""
    scores, is_tp = [], []
    for pred, c in zip(decode_cache(cache, space, biases, trim_whitespace), cache):
        for sp, raw in zip(pred.spans, pred.scores):
            scores.append(raw)
            is_tp.append(int(sp in c["gold"]))
    n_gold = sum(len(c["gold"]) for c in cache)
    best = best_threshold(scores, is_tp, n_gold, beta)
    return {"f_beta": best["f_beta"], "precision": best["precision"], "recall": best["recall"],
            "raw_threshold": best["threshold"], "spans_proposed": len(scores),
            "recall_ceiling": sum(is_tp) / n_gold if n_gold else 0.0}


def search_biases(cache: list[dict], space: LabelSpace, base_biases: dict, cfg_search, beta: float,
                  trim_whitespace: bool, seed: int) -> tuple[dict, dict]:
    """Random search around base_biases; the base setting is always trial 0.

    The best candidate replaces the base only if it beats it by at least cfg_search.min_improvement
    (F-beta). Taking the maximum over many candidates on a finite split favours lucky ones; without this
    margin, a 41-document smoke split picked biases of +-2 that scored 0.003 higher on calib and lower
    on test. Returns (biases, report)."""
    entries = list(cfg_search.search_space)
    unknown = {e["param"] for e in entries} - set(BIAS_KEYS)
    if unknown:
        raise ValueError(f"calibration.bias_search.search_space has unknown bias keys: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    candidates = [dict(base_biases)]
    candidates += [dict(base_biases, **sample_params(entries, rng)) for _ in range(int(cfg_search.num_samples))]
    trials = []
    for i, biases in enumerate(candidates):
        res = score_biases(cache, space, biases, beta, trim_whitespace)
        trials.append({"trial": i, "biases": biases, **res})
        if i == 0 or (i + 1) % 10 == 0:
            log.info("bias search %d/%d: best F-beta so far %.4f", i + 1, len(candidates),
                     max(t["f_beta"] for t in trials))
    best = max(trials, key=lambda t: (t["f_beta"], -t["trial"]))   # ties keep the earlier trial
    min_gain = float(cfg_search.min_improvement)
    accepted = best["f_beta"] - trials[0]["f_beta"] >= min_gain and best["trial"] != 0
    chosen = best if accepted else trials[0]
    ranked = sorted(trials, key=lambda t: (-t["f_beta"], t["trial"]))
    report = {"enabled": True, "beta": beta, "trials": len(trials), "min_improvement": min_gain,
              "accepted": accepted, "baseline": trials[0], "best": best, "chosen_trial": chosen["trial"],
              "top": ranked[:5]}
    if not accepted:
        log.info("best candidate gains %.4f F-beta over decode.biases (< %.4f): keeping decode.biases",
                 best["f_beta"] - trials[0]["f_beta"], min_gain)
    return dict(chosen["biases"]), report
