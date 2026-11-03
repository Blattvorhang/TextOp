"""Recompute goal normalization and diagnostic distributions in goal_stats.pkl.

Uses raw sampled goals with the dataset's statistics sampling path. Runtime
clamp limits remain manually configured in planner_dar.yaml.
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
    print(f"Saved goal statistics and diagnostic distributions to {out_path}")
    reference = stats['goal_clamp']['reference_distribution']
    print(f"Pooled reference goals: N={reference['n_samples']}/{reference['n_total']}, "
          f"height range={reference['height_range']} m; {reference['motion_filter']}")
