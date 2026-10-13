import pytest
from omegaconf import OmegaConf

from pii_mmbert.stages import resolve_accelerator_type, trial_resources

CLUSTER = {"CPU": 32.0, "GPU": 4.0, "accelerator_type:GeForce-RTX-4090": 1.0}


def _cfg(acc):
    return OmegaConf.create({"resources_per_trial": {"cpu": 4, "gpu": 0.25}, "accelerator_type": acc})


def test_substring_resolves_to_ray_name_with_small_share():
    res = trial_resources(_cfg("RTX"), CLUSTER)
    assert res == {"cpu": 4.0, "gpu": 0.25, "accelerator_type:GeForce-RTX-4090": 0.01}


def test_no_accelerator_constraint():
    assert trial_resources(_cfg(None), {"CPU": 4.0}) == {"cpu": 4.0, "gpu": 0.25}


def test_unknown_or_ambiguous_type_fails_fast():
    with pytest.raises(ValueError, match="no GPU detected"):
        resolve_accelerator_type("RTX", {"CPU": 4.0})
    two = dict(CLUSTER, **{"accelerator_type:RTX-A6000": 1.0})
    with pytest.raises(ValueError, match="set the exact name"):
        resolve_accelerator_type("RTX", two)
    assert resolve_accelerator_type("RTX-A6000", two) == "RTX-A6000"
