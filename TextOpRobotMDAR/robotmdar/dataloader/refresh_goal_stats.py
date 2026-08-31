"""Refresh goal_stats.pkl including the goal-clamp quantile tables.

Recomputes the frozen goal statistics from the training dataset and writes
them to ``target_dir/goal_stats.pkl`` — typically the checkpoint directory —
so the planner's data-driven goal clamp can read its envelope from the same
distribution the model was trained on.

Usage:
    robotmdar --config-name=refresh_goal_stats \\
        data.datadir=./dataset/BONES-SEED-29dof-FULL-50fps \\
        target_dir=./logs/pretrained/0827_goal_reaching \\
        skeleton.asset.assetRoot=./description/robots/g1/
"""

from pathlib import Path

import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from robotmdar.dtype import seed
from robotmdar.utils.dof_contract import configure_dof_contract


def main(cfg: DictConfig) -> None:
    configure_dof_contract(cfg)
    seed.set(cfg.seed)
    target_dir = Path(to_absolute_path(str(cfg.target_dir)))
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset = instantiate(cfg.data.train)
    stats = dataset._compute_goal_stats()
    out_path = target_dir / "goal_stats.pkl"
    torch.save(stats, out_path)
    print(f"Saved goal stats (incl. goal-clamp quantiles) to {out_path}")
