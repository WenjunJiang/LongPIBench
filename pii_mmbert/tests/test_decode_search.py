import numpy as np
import torch
from omegaconf import OmegaConf

from pii_mmbert.decode_search import decode_cache, sample_params, search_biases
from pii_mmbert.labels import LabelSpace


def test_sample_params_types_and_ranges():
    rng = np.random.default_rng(0)
    entries = [{"param": "u", "type": "uniform", "lower": -1.0, "upper": 2.0},
               {"param": "l", "type": "loguniform", "lower": 1e-3, "upper": 1e-1},
               {"param": "c", "type": "choice", "values": [4, 8]},
               {"param": "r", "type": "randint", "lower": 0, "upper": 3}]
    for _ in range(50):
        p = sample_params(entries, rng)
        assert -1.0 <= p["u"] <= 2.0 and 1e-3 <= p["l"] <= 1e-1 and p["c"] in (4, 8) and p["r"] in (0, 1, 2)


def _cache(space):
    # "Hi Bob ok": the model slightly prefers O over S-private_person on "Bob" (0.55 vs 0.45).
    text = "Hi Bob ok"
    lp = torch.full((3, space.num_labels), 1e-4).log()
    lp[0, 0] = lp[2, 0] = np.log(0.99)
    lp[1, 0], lp[1, space.id("S", "private_person")] = np.log(0.55), np.log(0.45)
    return [{"text": text, "offset_start": [0, 3, 7], "offset_end": [2, 6, 9], "log_probs": lp,
             "gold": {("private_person", 3, 6)}}]


def test_search_finds_start_bias_that_recovers_missed_span():
    space = LabelSpace(("private_person",))
    cache = _cache(space)
    assert decode_cache(cache, space, {}, True)[0].spans == []
    cfg = OmegaConf.create({"num_samples": 3, "min_improvement": 0.005, "search_space": [
        {"param": "transition_bias_background_to_start", "type": "uniform", "lower": 0.5, "upper": 2.0}]})
    biases, report = search_biases(cache, space, {}, cfg, beta=1.0, trim_whitespace=True, seed=0)
    assert report["baseline"]["f_beta"] == 0.0 and report["best"]["f_beta"] == 1.0
    assert biases["transition_bias_background_to_start"] >= 0.5
    assert decode_cache(cache, space, biases, True)[0].spans == [("private_person", 3, 6)]


def test_search_keeps_baseline_when_nothing_beats_it():
    space = LabelSpace(("private_person",))
    cache = _cache(space)
    cfg = OmegaConf.create({"num_samples": 3, "min_improvement": 0.005, "search_space": [
        {"param": "transition_bias_background_to_start", "type": "uniform", "lower": -3.0, "upper": -1.0}]})
    biases, report = search_biases(cache, space, {"transition_bias_background_to_start": 1.0}, cfg,
                                   beta=1.0, trim_whitespace=True, seed=0)
    assert biases == {"transition_bias_background_to_start": 1.0} and report["best"]["trial"] == 0


def test_small_gain_keeps_baseline():
    space = LabelSpace(("private_person",))
    cache = _cache(space)
    cfg = OmegaConf.create({"num_samples": 3, "min_improvement": 1.5, "search_space": [
        {"param": "transition_bias_background_to_start", "type": "uniform", "lower": 0.5, "upper": 2.0}]})
    biases, report = search_biases(cache, space, {}, cfg, beta=1.0, trim_whitespace=True, seed=0)
    assert report["best"]["f_beta"] == 1.0 and not report["accepted"] and biases == {}
