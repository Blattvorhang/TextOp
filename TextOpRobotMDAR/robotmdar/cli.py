#!/usr/bin/env python3
"""
RobotMDAR CLI - Simple entry point using Hydra framework
"""

import os

os.environ.setdefault("HYDRA_FULL_ERROR", "1")
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from robotmdar.dtype.debug import pdb_decorator
from robotmdar.utils.dof_contract import configure_dof_contract


@hydra.main(
    config_path="config",
    config_name="base",
    version_base="1.1",
)
def main(cfg: DictConfig):

    # Get the task from config
    task = cfg.task
    if "data" in cfg and "dof_dim" in cfg.data:
        configure_dof_contract(cfg)

    print("-=-" * 30)
    print(OmegaConf.to_yaml(cfg))
    print("=-=" * 30)

    task_modules = {
        "train-dar": "train.train_dar",
        "train-mvae": "train.train_mvae",
        "vis-mvae": "eval.vis_mvae",
        "vis-dar": "eval.vis_dar",
        "loop-dar": "eval.loop_dar",
        "planner-dar": "planner.planner_dar",
        "freq-dar": "eval.freq_dar",
        "export-dar": "export.export_dar_onnx",
        "noise-opt": "opt.noise_opt",
        "refresh-goal-stats": "dataloader.refresh_goal_stats",
    }

    if task in task_modules:
        module = __import__('robotmdar.' + task_modules[task],
                            fromlist=['main'])
        run = module.main
    else:
        print(f"Unknown task: {task}")
        print("Available tasks: " + ", ".join(task_modules.keys()))
        sys.exit(1)

    (run)(cfg)


if __name__ == "__main__":
    main()
