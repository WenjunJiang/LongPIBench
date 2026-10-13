from omegaconf import OmegaConf

from pii_mmbert.stages import trial_resources


def _cfg(custom):
    return OmegaConf.create({"resources_per_trial": {"cpu": 4, "gpu": 0.25, "custom": custom}})


def test_custom_resources_are_passed_as_configured():
    res = trial_resources(_cfg({"accelerator_type:RTX": 1}), {"CPU": 32.0, "GPU": 4.0, "accelerator_type:RTX": 4.0})
    assert res == {"cpu": 4.0, "gpu": 0.25, "accelerator_type:RTX": 1.0}


def test_no_custom_resources():
    assert trial_resources(_cfg(None), {"CPU": 4.0}) == {"cpu": 4.0, "gpu": 0.25}


def test_missing_custom_resource_only_warns(caplog):
    res = trial_resources(_cfg({"accelerator_type:RTX": 1}), {"CPU": 4.0})
    assert res["accelerator_type:RTX"] == 1.0
    assert "trials will stay PENDING" in caplog.text
