"""Entry point. All settings come from conf/config.yaml and can be overridden Hydra-style:

    python main.py stage=preprocess
    python main.py stage=tune tune.num_samples=32 tune.resources_per_trial.gpu=1
    python main.py stage=final final.best_params_path=outputs/tune/best_params.json train.epochs=5
    python main.py stage=calibrate                           # Viterbi biases, Platt + thresholds (calib split)
    python main.py stage=test                                # test-split metrics, before / after calibration
    python main.py stage=predict predict.text="Call Kim Min-jun at 010-1234-5678"
    python main.py stage=baseline_opf eval.max_docs=5000      # OpenAI Privacy Filter on the same test docs
    python main.py +experiment=smoke stage=tune          # small CPU run
"""

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from pii_mmbert import baseline_opf, stages

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    stage = cfg.stage
    if stage == "preprocess":
        stages.run_preprocess(cfg, to_absolute_path)
    elif stage == "tune":
        stages.run_tune(cfg, to_absolute_path)
    elif stage == "final":
        stages.run_final(cfg, list(HydraConfig.get().overrides.task), to_absolute_path)
    elif stage == "calibrate":
        stages.run_calibrate(cfg, to_absolute_path)
    elif stage == "test":
        stages.run_test(cfg, to_absolute_path)
    elif stage == "predict":
        stages.run_predict(cfg, to_absolute_path)
    elif stage == "baseline_opf":
        baseline_opf.run_opf_baseline(cfg, to_absolute_path)
    else:
        raise ValueError(f"unknown stage {stage!r}; expected preprocess | tune | final | calibrate | test | predict"
                         " | baseline_opf")


if __name__ == "__main__":
    main()
