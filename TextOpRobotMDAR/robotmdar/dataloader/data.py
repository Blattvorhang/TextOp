"""
Final Clean High-Performance Data Loader for Robot Motion Primitive Dataset

This is a clean, well-structured implementation with:
1. Motion-first generation ensuring primitive continuity
2. Small, focused functions for better readability
3. 100% interface compatibility with original
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import joblib
import yaml
from typing import Any, Tuple, Dict, List, Optional
import sys
import os
import time
import random
from omegaconf import DictConfig
from loguru import logger

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
from robotmdar.skeleton.robot import RobotSkeleton
from robotmdar.utils.goal import (
    GOAL_CLAMP_QUANTILE_LEVELS,
    GoalEncoding,
    GoalType,
    SPLIT_GOAL_SCHEMA,
    SPLIT_HORIZONTAL_SLICE,
    SPLIT_JOINT_SLICE,
    SPLIT_ORIENTATION_SLICE,
    SPLIT_VERTICAL_SLICE,
    SPLIT_VELOCITY_SLICE,
    build_ego_split_goal,
    SPLIT_GOAL_DIM,
    quaternion_yaw,
    validate_goal_stats,
)
import robotmdar.dtype.motion as motion_dtype
from robotmdar.dtype.motion import (
    AbsolutePose,
    G1_23DOF_FROM_29DOF_INDICES,
    MotionDict,
    MotionKeys,
)
import json


_GOAL_STATS_POSITION_CLIP = 3.0
_GOAL_STATS_VELOCITY_CLIP = 5.0

_HISTORY_JOINT_AUG_AMPS = (
    ("shoulder", 0.30),
    ("elbow", 0.10),
    ("hip", 0.05),
    ("knee", 0.05),
    ("ankle", 0.05),
    ("waist", 0.20),
    ("wrist", 0.05),
)
_HISTORY_ROOT_AUG_AMPS = {"x": 0.20, "y": 0.20, "z": 0.05, "h": 0.01}


def _abs_p50_p99(values: torch.Tensor) -> tuple[float, float]:
    values = values.detach().abs().reshape(-1).float()
    if values.numel() == 0:
        return 0.0, 0.0
    quantiles = torch.quantile(
        values,
        torch.tensor([0.5, 0.99], device=values.device, dtype=values.dtype),
    )
    return float(quantiles[0]), float(quantiles[1])


def _is_goal_stats_writer() -> bool:
    """True only for the process allowed to create/replace the goal-stats cache.

    Under DDP the process group is initialized before datasets are built, so
    rank 0 is the single writer. torchrun also sets RANK per worker, which
    covers the no-process-group case; plain single-process runs default to 0.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")) == 0


def _is_torchrun() -> bool:
    return "RANK" in os.environ


def _save_goal_stats_cache(goal_stats, goal_stats_path: Path) -> None:
    """Persist the cache atomically; only the writer rank saves.

    torch.save is not atomic, so concurrent ranks must never write the same
    pkl: a single writer serializes to a temp file and renames it into place.
    """
    if not _is_goal_stats_writer():
        return
    tmp_path = goal_stats_path.with_name(goal_stats_path.name + ".tmp")
    try:
        torch.save(goal_stats, tmp_path)
        tmp_path.replace(goal_stats_path)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write goal stats cache {goal_stats_path}: the dataset "
            "directory is not writable. Make it writable (chmod) or place a "
            "precomputed goal_stats.pkl into the dataset directory."
        ) from exc
    logger.info(f" Saved goal stats to {goal_stats_path}")


def _wait_for_goal_stats_cache(goal_stats_path: Path,
                               timeout_s: float = 900.0) -> None:
    """Wait for the writer rank's train dataset to publish the cache file.

    The atomic rename in _save_goal_stats_cache guarantees that a visible
    file is always a complete one.
    """
    deadline = time.monotonic() + timeout_s
    while not goal_stats_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f}s waiting for "
                f"{goal_stats_path}; the train dataset on the writer rank "
                "never produced it (dataset directory not writable?)"
            )
        time.sleep(2.0)


def _save_text_embedding_cache(text_embeddings: Dict[str, torch.Tensor],
                               text_embedding_path: Path) -> None:
    """Persist CLIP text embeddings atomically from the writer rank."""
    if not _is_goal_stats_writer():
        return
    tmp_path = text_embedding_path.with_name(text_embedding_path.name + ".tmp")
    try:
        torch.save(text_embeddings, tmp_path)
        tmp_path.replace(text_embedding_path)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write text embedding cache {text_embedding_path}: the "
            "dataset directory is not writable. Disable "
            "data.load_text_embeddings or place a precomputed cache there."
        ) from exc
    logger.info(f" Saved text embeddings to {text_embedding_path}")


def _text_embedding_cache_complete(cache: Dict[str, torch.Tensor],
                                   needed_texts: set[str],
                                   clip_dim: int) -> bool:
    if '' not in cache:
        return False
    for text in needed_texts:
        embedding = cache.get(text)
        if embedding is None or int(embedding.shape[-1]) != int(clip_dim):
            return False
    return True


def _wait_for_complete_text_embedding_cache(text_embedding_path: Path,
                                            needed_texts: set[str],
                                            clip_dim: int,
                                            timeout_s: float = 900.0
                                            ) -> Dict[str, torch.Tensor]:
    deadline = time.monotonic() + timeout_s
    while True:
        if text_embedding_path.exists():
            cache = torch.load(text_embedding_path, map_location="cpu")
            if _text_embedding_cache_complete(cache, needed_texts, clip_dim):
                return cache
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f}s waiting for a complete "
                f"text embedding cache at {text_embedding_path}."
            )
        time.sleep(2.0)


class SkeletonPrimitiveDataset(data.IterableDataset):
    """
    Clean, high-performance SkeletonPrimitiveDataset with motion-first generation.
    
    Key features:
    - Motion-first generation ensuring primitive continuity
    - Small, focused functions for better readability
    - 100% interface compatibility with original
    """

    # ===============================================================
    # Load & build dataset

    def __init__(
        self,
        robot_cfg: DictConfig,
        batch_size: int,
        nfeats: int,
        history_len: int,
        future_len: int,
        num_primitive: int,
        datadir: str,
        action_statistics_path: str,
        goal_offset: int = 0,
        goal_offset_range: Optional[Tuple[int, int]] = None,
        goal_type: GoalType | str = GoalType.ROOT,
        goal_encoding: GoalEncoding | str | None = None,
        goal_per_primitive: bool = False,
        goal_timestep_mode: str = "relative",
        goal_include_log_d_hor: bool = True,
        time_to_arrival_mode: Optional[str] = None,
        weighted_sample: bool = False,
        frame_weight: bool = False,
        use_weighted_meanstd: bool = False,
        split: str = 'train',
        device: str = 'cuda',
        dof_dim: Optional[int] = None,
        feature_version: Optional[int] = None,
        normalization_path: Optional[str] = None,
        load_goal_stats: bool = True,
        std_floor: float = 0.0,
        **kwargs: Any
    ):
        super().__init__()
        if feature_version is not None:
            motion_dtype.set_feature_version(int(feature_version))
        # Store parameters
        self.batch_size = batch_size
        self.history_len = history_len
        self.future_len = future_len
        self.num_primitive = num_primitive

        self.nfeats = int(nfeats)
        self.dof_dim = (
            motion_dtype.infer_feature_dof_dim(self.nfeats)
            if dof_dim is None else int(dof_dim)
        )
        expected_nfeats = motion_dtype.motion_feature_dim_for_dof(self.dof_dim)
        if self.nfeats != expected_nfeats:
            raise ValueError(
                f"dof_dim={self.dof_dim} requires nfeats={expected_nfeats}, "
                f"got {self.nfeats}"
            )

        self.segment_len = self.history_len + self.future_len * self.num_primitive + 1
        self.context_len = self.history_len + self.future_len

        self.goal_type = GoalType.parse(goal_type)
        self.goal_encoding = GoalEncoding.parse(
            goal_encoding if goal_encoding is not None else GoalEncoding.LEGACY40
        )
        if self.goal_type is GoalType.JOINT_STATE and self.dof_dim != 29:
            raise ValueError(
                "goal_type='joint_state' requires dof_dim=29, got "
                f"{self.dof_dim}"
            )
        self.goal_per_primitive = goal_per_primitive
        self.goal_include_log_d_hor = bool(goal_include_log_d_hor)
        if time_to_arrival_mode is not None:
            goal_timestep_mode = time_to_arrival_mode
        self.goal_timestep_mode = str(goal_timestep_mode).lower()
        self.time_to_arrival_mode = self.goal_timestep_mode
        if self.goal_timestep_mode not in ("relative", "zero"):
            raise ValueError(
                "time_to_arrival_mode/goal_timestep_mode must be "
                "'relative' or 'zero', got "
                f"{goal_timestep_mode!r}"
            )
        self.goal_offset = int(goal_offset)
        if goal_offset_range is None:
            self.goal_offset_range = (self.goal_offset, self.goal_offset)
        else:
            if len(goal_offset_range) != 2:
                raise ValueError(
                    "goal_offset_range must contain inclusive [min, max] bounds"
                )
            self.goal_offset_range = tuple(int(value) for value in goal_offset_range)
        self.min_goal_offset, self.max_goal_offset = self.goal_offset_range
        if self.min_goal_offset > self.max_goal_offset:
            raise ValueError(
                f"Invalid goal_offset_range={self.goal_offset_range}: min > max"
            )
        min_valid_offset = 1 - self.future_len
        if self.min_goal_offset < min_valid_offset:
            raise ValueError(
                f"goal offsets must be >= {min_valid_offset} so the goal is "
                f"after the reference frame, got {self.goal_offset_range}"
            )

        self.required_length = self.segment_len + max(0, self.max_goal_offset)
        self.weighted_sample = weighted_sample
        self.frame_weight = frame_weight
        self.action_statistics_path = action_statistics_path
        self.use_weighted_meanstd = use_weighted_meanstd

        self.datadir = Path(datadir)
        self.load_goal_stats = bool(load_goal_stats)
        self.goal_stats_path = self.datadir / 'goal_stats.pkl'
        self.goal_stats = None
        self.normalization_path = (
            Path(normalization_path) if normalization_path else None
        )
        self.std_floor = float(std_floor)
        self.split = split
        self.device = "cpu"  # Keep embeddings on CPU initially
        self.sample_cache_size = max(0, int(kwargs.get('sample_cache_size', 8)))
        self.preload_ram = bool(kwargs.get('preload_ram', False))
        self.preload_num_workers = max(
            1, int(kwargs.get('preload_num_workers', 8)))
        self.preload_show_progress = bool(
            kwargs.get('preload_show_progress', _is_goal_stats_writer()))
        self.load_scene = bool(kwargs.get('load_scene', True))
        self.load_text_embeddings = bool(
            kwargs.get('load_text_embeddings', True))
        self.clip_version = str(kwargs.get('clip_version', 'ViT-B/32'))
        self.clip_dim = int(kwargs.get('clip_dim', 512))
        self._stats_device_cache = {}
        # Planner-side DR is disabled by default: augmentation_enabled must be
        # explicitly set to true (LDM configs only — VAE training stays clean).
        # Once enabled, augmentation activates from augmentation_start_step
        # global optimizer steps onward; the training loop updates the current
        # step through set_training_step.
        self.augmentation_start_step = int(kwargs.get('augmentation_start_step', 0))
        self.augmentation_prob = float(kwargs.get('augmentation_prob', 0.5))
        self.augmentation_enabled = bool(kwargs.get('augmentation_enabled', False))
        self.training_step = 0
        if not 0.0 <= self.augmentation_prob <= 1.0:
            raise ValueError('augmentation_prob must be in [0, 1]')
        self._sample_cache = OrderedDict()

        # DDP rank and world_size, set externally by training script
        self.rank = 0
        self.world_size = 1

        # Load and prepare data
        self._load_data()

        # Initialize skeleton and normalization
        self.skeleton = RobotSkeleton(device=self.device, cfg=robot_cfg)

        if self.weighted_sample and self.use_weighted_meanstd:
            self._load_weighted_meanstd()
        else:
            self._load_meanstd()

        self._load_goal_stats()

    def _load_data(self) -> None:
        """Load and prepare data efficiently"""
        logger.info(f" Loading {self.split} data...")
        self._load_statistics()

        # Load data files
        if self.split == 'none':
            return
        splits = ['train', 'val'] if self.split == 'all' else [self.split]
        all_data = []
        for split in splits:
            datapkl = self.datadir / f'{split}.pkl'
            assert datapkl.exists(), f"Data file {datapkl} does not exist"
            all_data.extend(joblib.load(datapkl))

        # Fix length labels and filter against the active training window. New
        # datasets use lightweight manifest records; old monolithic files remain
        # supported for compatibility.
        self.valid_indices = []
        required_length = self.required_length
        for i, item in enumerate(all_data):
            if 'motion' in item:
                item['length'] = int(item['motion']['motion_len'])
            elif '_data_path' in item:
                item['length'] = int(item['length'])
            else:
                raise ValueError(
                    f"Dataset item {i} has neither 'motion' nor '_data_path'"
                )
            if item['length'] >= required_length:
                self.valid_indices.append(i)

        self.raw_data = all_data

        if not self.valid_indices:
            longest = max((item['length'] for item in all_data), default=0)
            raise ValueError(
                "No motion sequences satisfy the active training window: "
                f"required_length={required_length} "
                f"(history_len={self.history_len}, future_len={self.future_len}, "
                f"num_primitive={self.num_primitive}, "
                f"goal_offset_range={self.goal_offset_range}), "
                f"loaded={len(all_data)}, longest={longest}, split={self.split!r}"
            )

        if self.weighted_sample:
            self._cal_sample_weight()

        if not getattr(self, 'load_scene', True):
            for item in all_data:
                item.pop('scene', None)

        if getattr(self, 'preload_ram', False):
            self._preload_manifest_samples()

        logger.info(f" Found {len(self.valid_indices)} valid samples out of {len(self.raw_data)}")

        if self.load_text_embeddings:
            self._load_text_embeddings()
        else:
            self.text_embeddings_dict = {
                '': torch.zeros(self.clip_dim, dtype=torch.float32)
            }

    def _strip_scene_if_needed(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if getattr(self, 'load_scene', True) or 'scene' not in sample:
            return sample
        sample = dict(sample)
        sample.pop('scene', None)
        return sample

    def _load_manifest_sample(self, sample_path: Path) -> Dict[str, Any]:
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Manifest sample does not exist: {sample_path}"
            )
        stored = joblib.load(sample_path)
        if not isinstance(stored, dict) or 'motion' not in stored:
            raise ValueError(f"Invalid manifest sample: {sample_path}")
        return self._strip_scene_if_needed(stored)

    def _preload_manifest_samples(self) -> None:
        """Hydrate manifest-backed samples once so training does no joblib I/O."""
        sample_paths = []
        seen = set()
        for sample_idx in self.valid_indices:
            relpath = self.raw_data[sample_idx].get('_data_path')
            if relpath is None:
                continue
            sample_path = self.datadir / relpath
            cache_key = str(sample_path)
            if cache_key in seen:
                continue
            seen.add(cache_key)
            sample_paths.append(sample_path)

        if not sample_paths:
            return

        num_workers = min(self.preload_num_workers, len(sample_paths))
        logger.info(
            " Preloading {} {} samples into RAM with {} threads"
            " (load_scene={})...",
            len(sample_paths),
            self.split,
            num_workers,
            self.load_scene,
        )
        preloaded = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            loaded_iter = executor.map(self._load_manifest_sample, sample_paths)
            iterator = tqdm(
                zip(sample_paths, loaded_iter),
                total=len(sample_paths),
                desc=f"preload {self.split}",
                disable=not self.preload_show_progress,
            )
            for sample_path, stored in iterator:
                preloaded[str(sample_path)] = stored
        self._preloaded_samples = preloaded
        self._sample_cache.clear()
        logger.info(
            " Preloaded {} {} samples into RAM",
            len(self._preloaded_samples),
            self.split,
        )

    def _hydrate_sample(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Load a manifest-backed sample, with a small per-worker LRU cache."""
        relpath = record.get('_data_path')
        if relpath is None:
            return record

        sample_path = self.datadir / relpath
        cache_key = str(sample_path)
        preloaded = getattr(self, '_preloaded_samples', None)
        if preloaded is not None:
            try:
                stored = preloaded[cache_key]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Manifest sample was not preloaded: {sample_path}"
                ) from exc
            sample = dict(stored)
            sample.update({key: value for key, value in record.items()
                           if key != '_data_path'})
            return sample

        cache = getattr(self, '_sample_cache', None)
        if cache is None:
            cache = self._sample_cache = OrderedDict()

        if cache_key in cache:
            stored = cache.pop(cache_key)
            cache[cache_key] = stored
        else:
            stored = self._load_manifest_sample(sample_path)
            cache[cache_key] = stored
            cache_size = getattr(self, 'sample_cache_size', 8)
            while len(cache) > cache_size:
                cache.popitem(last=False)

        # Sampling weights are computed on manifest records after packing.
        sample = dict(stored)
        sample.update({key: value for key, value in record.items()
                       if key != '_data_path'})
        return sample

    def _cal_sample_weight(self):

        logger.info(f" ====================Use Weighted Sample====================")

        with open(self.action_statistics_path, 'r') as f:
            action_statistics = json.load(f)

        # ── Recovery boost multiplier ──
        # Flat-lying recovery sequences (stand_up_lying*, faint_stand_up_lying*)
        # are rare (~0.4% of data). Multiply their sampling weight so the model
        # sees them more often without changing the global category distribution.
        RECOVERY_WEIGHT_MULTIPLIER = 5.0

        for data in self.raw_data:
            seq_weight = 0
            for seg in data['frame_ann']:
                seg_act_cat = seg[3]
                act_weights = 0
                for act_cat in seg_act_cat:
                    # breakpoint()
                    if act_cat not in action_statistics:
                        continue
                    else:
                        act_weights += action_statistics[act_cat]['weight']
                seq_weight += (seg[1] - seg[0]) * act_weights

            if data.get('_recovery_boost'):
                seq_weight *= RECOVERY_WEIGHT_MULTIPLIER

            data['weight'] = seq_weight
            num_frames = data['length']

            frame_weights = []
            for frame_idx in range(0, num_frames - self.segment_len + 1):
                start_t = frame_idx / self.fps
                end_t = (frame_idx + self.segment_len - 1) / self.fps
                frame_weight = 0
                for seg in data['frame_ann']:
                    overlap_len = self._get_overlap([seg[0], seg[1]], [start_t, end_t])
                    if overlap_len > 0:
                        act_weights = 0
                        for act_cat in seg[3]:
                            if act_cat not in action_statistics:
                                continue
                            else:
                                act_weights += action_statistics[act_cat]['weight']
                        # act_weights = sum([action_statistics[act_cat]['weight'] for act_cat in seg[3]])
                        frame_weight += overlap_len * act_weights
                frame_weights.append(frame_weight)
            data['frame_weights'] = frame_weights

        valid_data = [self.raw_data[i] for i in self.valid_indices]
        babel_sum = sum(data['weight'] for data in valid_data)
        print('babel sum: ', babel_sum)
        samp_percent = 0.0
        print('samp percent: ', samp_percent)
        if babel_sum > 0:
            for data in valid_data:
                data['weight'] = data['weight'] / babel_sum * (1 - samp_percent)

        seq_weights = np.asarray([data['weight'] for data in valid_data], dtype=np.float64)
        weight_sum = seq_weights.sum()
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            logger.warning(
                "Valid motion weights are empty or non-positive; using uniform sampling"
            )
            seq_weights = np.full(len(valid_data), 1.0 / len(valid_data))
        else:
            seq_weights = seq_weights / weight_sum
        self.seq_weights = seq_weights

        # self._statistic_sample_weight()
        # breakpoint()

    def _statistic_sample_weight(self):
        import re
        act_weight = {}
        # for i, data in enumerate(self.raw_data):
        #     for seg in data['frame_ann']:
        #         seg_ann = seg[2]
        #         seg_ann = re.sub(r'[^\w\s]', ' ', seg_ann.lower())
        #         act_weight[seg_ann] = act_weight.get(seg_ann,
        #                                              0) + self.seq_weights[i]

        # sorted_act_weight = sorted(act_weight.items(),
        #                            key=lambda item: item[1],
        #                            reverse=True)

        # # 2. 写入文件
        # with open('ann_sample_weight_statistics.txt', 'w',
        #           encoding='utf-8') as f:
        #     for seg in sorted_act_weight:
        #         # 将key和value转换为字符串并用分隔符连接
        #         line = f"{seg[0]}\t{seg[1]}"
        #         f.write(line + '\n')

        # print(f"总条目数: {len(sorted_act_weight)}")

        for i, data in enumerate(self.raw_data):
            act_weight[i] = data['weight']

        with open('data_sample_weight_statistics_norm.txt', 'w', encoding='utf-8') as f:
            for seg in act_weight.items():
                # 将key和value转换为字符串并用分隔符连接
                line = f"{seg[0]}\t{seg[1]}"
                f.write(line + '\n')

    def _load_statistics(self) -> None:
        """Load motion statistics"""
        statistics_yaml = self.datadir / 'statistics.yaml'
        with open(statistics_yaml, 'r') as f:
            self.statistics = yaml.safe_load(f)
        self.fps = self.statistics['fps']
        source_nfeats = int(self.statistics.get('nfeats', self.nfeats))
        self.source_feature_version = int(
            self.statistics.get('feature_version', 3)
        )
        self.source_dof_dim = int(
            self.statistics.get('dof_dim', (source_nfeats - 11) // 2)
        )

    def _select_model_dof(self, dof):
        """Adapt stored G1 joints to the selected model-facing contract."""
        source_dim = int(dof.shape[-1])
        if source_dim == self.dof_dim:
            return dof
        if source_dim == 29 and self.dof_dim == 23:
            return dof[..., list(G1_23DOF_FROM_29DOF_INDICES)]
        raise ValueError(
            f"Cannot adapt stored {source_dim}-DoF motion to "
            f"dof_dim={self.dof_dim}"
        )

    def set_training_step(self, step: int) -> None:
        """Update the step gate used by train-time history augmentation."""
        self.training_step = int(step)

    def _load_joint_limits(self) -> Optional[torch.Tensor]:
        """Lazily parse per-joint limits [dof_dim, 2] from the G1 MJCF.

        Returns None when limits are unavailable, in which case the
        augmentation falls back to the coarse ±π clamp.
        """
        import xml.etree.ElementTree as ETree
        try:
            mjcf = Path(self.skeleton.fk.mjcf_file)
            if not mjcf.exists():
                return None
            tree = ETree.parse(mjcf)
            joint_nodes = tree.getroot().findall('.//joint')
            ranges = {}
            for node in joint_nodes:
                name = node.attrib.get('name')
                rng = node.attrib.get('range')
                if name is None or rng is None:
                    continue
                parts = [float(v) for v in rng.split()]
                if len(parts) == 2:
                    ranges[name] = parts
            limits = []
            for name in self.skeleton.fk.dof_joint_names:
                limits.append(ranges.get(name, [-np.pi, np.pi]))
            return torch.as_tensor(limits, dtype=torch.float32)
        except Exception:
            return None

    def _joint_limit_tensor(self) -> Optional[torch.Tensor]:
        if not hasattr(self, '_cached_joint_limits'):
            self._cached_joint_limits = self._load_joint_limits()
        return self._cached_joint_limits

    @staticmethod
    def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ax, ay, az, aw = a.unbind(-1)
        bx, by, bz, bw = b.unbind(-1)
        return torch.stack((
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ), dim=-1)

    @staticmethod
    def _smoothstep(u: torch.Tensor) -> torch.Tensor:
        return 3.0 * u * u - 2.0 * u * u * u

    @staticmethod
    def _history_aug_weight_schedule(
        history_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        u = torch.linspace(0.0, 1.0, history_len + 1,
                           device=device, dtype=dtype)
        return 1.0 - SkeletonPrimitiveDataset._smoothstep(u)

    def _history_aug_dof_names(self) -> List[str]:
        names = list(self.skeleton.fk.dof_joint_names)
        if len(names) == self.dof_dim:
            return names
        if len(names) == 29 and self.dof_dim == 23:
            return [names[i] for i in G1_23DOF_FROM_29DOF_INDICES]
        raise ValueError(
            f"Cannot map {len(names)} skeleton joint names to "
            f"dof_dim={self.dof_dim}"
        )

    def _history_aug_amp_vector(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = []
        for name in self._history_aug_dof_names():
            for prefix, amplitude in _HISTORY_JOINT_AUG_AMPS:
                if prefix in name:
                    values.append(amplitude)
                    break
            else:
                raise ValueError(
                    f"No history augmentation amplitude for joint {name!r}"
                )
        return torch.as_tensor(values, device=device, dtype=dtype)

    def _history_aug_joint_limits(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        limits = self._joint_limit_tensor()
        if limits is None:
            return torch.as_tensor(
                [[-np.pi, np.pi]] * self.dof_dim,
                dtype=dtype,
                device=device,
            )
        limits = limits.to(device=device, dtype=dtype)
        if limits.shape[0] == self.dof_dim:
            return limits
        if limits.shape[0] == 29 and self.dof_dim == 23:
            return limits[list(G1_23DOF_FROM_29DOF_INDICES)]
        raise ValueError(
            f"Cannot map joint-limit tensor with shape {tuple(limits.shape)} "
            f"to dof_dim={self.dof_dim}"
        )

    def _augment_raw_motion(self, motion: Dict[str, torch.Tensor],
                            generator: Optional[torch.Generator]) -> bool:
        """Perturb raw history states before feature extraction."""
        augmentation_enabled = bool(getattr(self, 'augmentation_enabled', False))
        training_step = int(getattr(self, 'training_step', 0))
        augmentation_start_step = int(
            getattr(self, 'augmentation_start_step', 0))
        if (not augmentation_enabled or getattr(self, 'split', '') != 'train'
                or training_step < augmentation_start_step):
            return False
        if torch.rand((), generator=generator).item() >= self.augmentation_prob:
            return False

        history_count = min(self.history_len, motion['dof'].shape[0] - 1)
        if history_count <= 0:
            return False
        device, dtype = motion['dof'].device, motion['dof'].dtype
        w = self._history_aug_weight_schedule(
            history_count, device, dtype)[:history_count]

        amps = self._history_aug_amp_vector(device, dtype)
        limits = self._history_aug_joint_limits(device, dtype)
        x = motion['dof'][:history_count]
        active = w > 0.0
        lo = ((limits[:, 0] - x[active]) / w[active, None]).amax(dim=0)
        hi = ((limits[:, 1] - x[active]) / w[active, None]).amin(dim=0)
        lower, upper = torch.maximum(lo, -amps), torch.minimum(hi, amps)
        q = lower + (upper - lower) * torch.rand(
            self.dof_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        q = torch.where(lower <= upper, q, torch.zeros_like(q))
        motion['dof'][:history_count] += w[:, None] * q

        rx = (
            torch.rand((), generator=generator, device=device, dtype=dtype)
            * 2.0 - 1.0
        ) * _HISTORY_ROOT_AUG_AMPS["x"]
        ry = (
            torch.rand((), generator=generator, device=device, dtype=dtype)
            * 2.0 - 1.0
        ) * _HISTORY_ROOT_AUG_AMPS["y"]
        rz = (
            torch.rand((), generator=generator, device=device, dtype=dtype)
            * 2.0 - 1.0
        ) * _HISTORY_ROOT_AUG_AMPS["z"]
        half = 0.5 * w
        zeros = torch.zeros_like(half)
        q_x = torch.stack((
            torch.sin(half * rx), zeros, zeros, torch.cos(half * rx)
        ), dim=-1)
        q_y = torch.stack((
            zeros, torch.sin(half * ry), zeros, torch.cos(half * ry)
        ), dim=-1)
        q_z = torch.stack((
            zeros, zeros, torch.sin(half * rz), torch.cos(half * rz)
        ), dim=-1)
        q_off = self._quat_mul_xyzw(q_x, self._quat_mul_xyzw(q_y, q_z))
        motion['root_rot'][:history_count] = self._quat_mul_xyzw(
            motion['root_rot'][:history_count],
            q_off,
        )

        dh = (
            torch.rand((), generator=generator, device=device, dtype=dtype)
            * 2.0 - 1.0
        ) * _HISTORY_ROOT_AUG_AMPS["h"]
        motion['root_trans_offset'][:history_count, 2] += w * dh
        return True

    def _load_text_embeddings(self) -> None:
        """Load or compute CLIP embeddings for frame_ann text labels."""
        text_embedding_path = self.datadir / f'{self.split}_text_embed.pkl'
        needed_texts = self._collect_text_labels(self.raw_data)
        if text_embedding_path.exists():
            logger.info(" Loading cached text embeddings...")
            self.text_embeddings_dict = torch.load(
                text_embedding_path, map_location="cpu")
            if _text_embedding_cache_complete(
                    self.text_embeddings_dict, needed_texts, self.clip_dim):
                if '' not in self.text_embeddings_dict:
                    self.text_embeddings_dict[''] = torch.zeros(
                        self.clip_dim, dtype=torch.float32)
                return
            logger.info(" Text embedding cache is stale; rebuilding...")
            if not _is_goal_stats_writer() and _is_torchrun():
                self.text_embeddings_dict = (
                    _wait_for_complete_text_embedding_cache(
                        text_embedding_path, needed_texts, self.clip_dim))
                return
        else:
            if not _is_goal_stats_writer() and _is_torchrun():
                logger.info(" Waiting for cached text embeddings...")
                self.text_embeddings_dict = (
                    _wait_for_complete_text_embedding_cache(
                        text_embedding_path, needed_texts, self.clip_dim))
                if '' not in self.text_embeddings_dict:
                    self.text_embeddings_dict[''] = torch.zeros(
                        self.clip_dim, dtype=torch.float32)
                return
            logger.info(" Computing text embeddings...")
            from robotmdar.model.clip import load_and_freeze_clip
            clip_model = load_and_freeze_clip(
                clip_version=self.clip_version,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.text_embeddings_dict = self._compute_text_embeddings(
                self.raw_data, clip_model, clip_dim=self.clip_dim)
            _save_text_embedding_cache(
                self.text_embeddings_dict, text_embedding_path)
        if '' not in self.text_embeddings_dict:
            self.text_embeddings_dict[''] = torch.zeros(
                self.clip_dim, dtype=torch.float32)

    @staticmethod
    def _collect_text_labels(raw_data: List[Dict[str, Any]]) -> set[str]:
        all_texts = set()
        for item in raw_data:
            for ann in item.get('frame_ann', []):
                all_texts.add(str(ann[2]))
        return all_texts

    @staticmethod
    def _compute_text_embeddings(raw_data: List[Dict[str, Any]],
                                 clip_model: nn.Module,
                                 batch_size: int = 64,
                                 clip_dim: int = 512
                                 ) -> Dict[str, torch.Tensor]:
        """Compute text embeddings for unique frame_ann labels."""
        uni_texts = sorted(
            SkeletonPrimitiveDataset._collect_text_labels(raw_data))
        if not uni_texts:
            return {'': torch.zeros(clip_dim, dtype=torch.float32)}

        embeddings_list = []
        from robotmdar.model.clip import encode_text
        for i in range(0, len(uni_texts), batch_size):
            batch_texts = uni_texts[i:i + batch_size]
            batch_embeddings = encode_text(clip_model, batch_texts)
            embeddings_list.append(batch_embeddings.detach().cpu().float())

        text_embeddings = torch.cat(embeddings_list, dim=0)
        text_embeddings_dict = dict(zip(uni_texts, text_embeddings))
        text_embeddings_dict[''] = torch.zeros_like(text_embeddings[0])
        return text_embeddings_dict

    def _load_meanstd(self) -> None:
        """Load or compute mean/std for normalization"""
        meanstd_cache_path = (
            self.normalization_path
            if self.normalization_path is not None
            else self._default_meanstd_path(weighted=False)
        )
        if meanstd_cache_path.exists():
            logger.info(f" Loading cached mean/std from {meanstd_cache_path}...")
            meanstd = torch.load(meanstd_cache_path, map_location="cpu")
            if not self._meanstd_cache_is_current(meanstd):
                logger.warning(
                    " Cached mean/std at {} is stale for FeatureVersion {} "
                    "(arrival-state v6 caches are stored with metadata); "
                    "recomputing",
                    meanstd_cache_path, motion_dtype.FeatureVersion,
                )
                meanstd = self._compute_meanstd()
                torch.save(
                    self._meanstd_cache_payload(meanstd),
                    meanstd_cache_path,
                )
        else:
            logger.info(f" Computing mean/std..")
            assert self.split == 'train', "Compute mean and std from 'train' set"

            # zjk: DART meanstd cal method
            meanstd = self._compute_meanstd()
            # meanstd = self._compute_meanstd_V2()

            torch.save(
                self._meanstd_cache_payload(meanstd),
                meanstd_cache_path,
            )
            logger.info(f" Saved mean/std to {meanstd_cache_path}")

        self._set_meanstd(self._meanstd_cache_tuple(meanstd), meanstd_cache_path)

    def _load_weighted_meanstd(self) -> None:
        """Load or compute mean/std for normalization"""
        meanstd_cache_path = (
            self.normalization_path
            if self.normalization_path is not None
            else self._default_meanstd_path(weighted=True)
        )
        if meanstd_cache_path.exists():
            logger.info(f" Loading cached mean/std from {meanstd_cache_path}...")
            meanstd = torch.load(meanstd_cache_path, map_location="cpu")
            if not self._meanstd_cache_is_current(meanstd):
                logger.warning(
                    " Cached weighted mean/std at {} is stale for "
                    "FeatureVersion {}; recomputing",
                    meanstd_cache_path, motion_dtype.FeatureVersion,
                )
                meanstd = self._compute_meanstd()
                torch.save(
                    self._meanstd_cache_payload(meanstd),
                    meanstd_cache_path,
                )
        else:
            logger.info(f" Computing mean/std..")
            assert self.split == 'train', "Compute mean and std from 'train' set"

            # zjk: DART meanstd cal method
            meanstd = self._compute_meanstd()
            # meanstd = self._compute_meanstd_V2()

            torch.save(
                self._meanstd_cache_payload(meanstd),
                meanstd_cache_path,
            )
            logger.info(f" Saved mean/std to {meanstd_cache_path}")

        self._set_meanstd(self._meanstd_cache_tuple(meanstd), meanstd_cache_path)

    def _default_meanstd_path(self, weighted: bool) -> Path:
        stem = 'weighted_meanstd' if weighted else 'meanstd'
        if motion_dtype.FeatureVersion == 3:
            return self.datadir / f'{stem}.pkl'
        return self.datadir / (
            f'{stem}_v{motion_dtype.FeatureVersion}_dof{self.dof_dim}.pkl'
        )

    def _meanstd_cache_payload(self, meanstd):
        if motion_dtype.FeatureVersion == 6:
            mean, std = meanstd
            return {
                'mean': mean,
                'std': std,
                'feature_version': 6,
                'feature_alignment': 'arrival',
            }
        return meanstd

    def _meanstd_cache_is_current(self, meanstd) -> bool:
        if motion_dtype.FeatureVersion == 6:
            return (
                isinstance(meanstd, dict)
                and int(meanstd.get('feature_version', -1)) == 6
                and meanstd.get('feature_alignment') == 'arrival'
                and 'mean' in meanstd
                and 'std' in meanstd
            )
        return isinstance(meanstd, tuple) and len(meanstd) == 2

    def _meanstd_cache_tuple(self, meanstd):
        if isinstance(meanstd, dict):
            return meanstd['mean'], meanstd['std']
        return meanstd

    def _set_meanstd(self, meanstd, source_path: Path) -> None:
        mean, std = meanstd
        if mean.shape[-1] != self.nfeats or std.shape[-1] != self.nfeats:
            raise ValueError(
                f"Mean/std at {source_path} has shape "
                f"mean={tuple(mean.shape)}, std={tuple(std.shape)}; "
                f"expected last dim {self.nfeats} for "
                f"FeatureVersion {motion_dtype.FeatureVersion}"
            )
        if self.std_floor > 0.0:
            std = torch.clamp(std, min=self.std_floor)
        self.mean, self.std = mean, std
        self._stats_device_cache.clear()

    def _goal_stats_meta(self) -> Dict[str, Any]:
        return {
            'goal_offset_range': list(self.goal_offset_range),
            'goal_per_primitive': bool(self.goal_per_primitive),
            'future_len': int(self.future_len),
            'fps': float(self.fps),
            'goal_timestep_mode': self.goal_timestep_mode,
            'encodings': [
                GoalEncoding.SINGLE.value,
                GoalEncoding.SPLIT.value,
            ],
            'dataset_path': str(self.datadir),
            'goal_type': self.goal_type.value,
            'goal_dim': SPLIT_GOAL_DIM,
            'goal_schema': SPLIT_GOAL_SCHEMA,
            'goal_include_log_d_hor': bool(self.goal_include_log_d_hor),
            'feature_version': motion_dtype.FeatureVersion,
            'dof_dim': int(self.dof_dim),
        }

    def _goal_stats_from_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        pos_terms = []
        d_hor_terms = []
        urgency_terms = []
        velocity_terms = []
        orientation_terms = []
        pose_terms = []
        # Raw per-window clamp distribution: the ego goal XY radius at the
        # window start (the goal as issued) and the implied speed r/T.
        clamp_dist_terms = []
        clamp_speed_terms = []

        for primitive in batch_data:
            goal = build_ego_split_goal(
                world_goal_pos=primitive['world_goal_pos'],
                world_goal_rot=primitive['world_goal_rot'],
                world_goal_dof=primitive['world_goal_dof'],
                world_root_velocity=primitive['world_goal_vel'],
                reference_pos=primitive['gt_ref_pos'],
                reference_rot=primitive['gt_ref_rot'],
                time_to_arrival_seconds=primitive['time_to_arrival'],
                fps=float(self.fps),
                goal_include_log_d_hor=self.goal_include_log_d_hor,
            )
            hor = goal[:, SPLIT_HORIZONTAL_SLICE]
            vert = goal[:, SPLIT_VERTICAL_SLICE]
            orientation = goal[:, SPLIT_ORIENTATION_SLICE]
            pose = goal[:, SPLIT_JOINT_SLICE]
            velocity = goal[:, SPLIT_VELOCITY_SLICE]

            pos_terms.append(torch.cat((hor[:, 0:4], vert[:, 0:2]), dim=-1))
            d_hor_terms.append(hor[:, 3:4])
            urgency_terms.append(torch.cat((hor[:, 5:9], vert[:, 5:6]), dim=-1))
            orientation_terms.append(torch.cat((vert[:, 2:5], orientation), dim=-1))
            pose_terms.append(pose)
            velocity_terms.append(velocity)

            goal_radius = hor[..., 3:4]
            time_budget = torch.as_tensor(
                primitive['time_to_arrival'],
                device=goal_radius.device,
                dtype=goal_radius.dtype,
            ).reshape(goal_radius.shape).clamp_min(1.0 / float(self.fps))
            clamp_dist_terms.append(goal_radius)
            clamp_speed_terms.append(goal_radius / time_budget)

        if not pos_terms:
            raise ValueError("Unable to compute goal statistics from empty batch")

        pos = torch.cat(pos_terms, dim=0)
        d_hor = torch.cat(d_hor_terms, dim=0)
        s_d = torch.quantile(
            d_hor.detach().reshape(-1).float().clamp_min(0.0),
            0.5,
        ).clamp(min=1e-6)
        if self.goal_include_log_d_hor:
            log_terms = torch.log1p(
                d_hor / s_d.to(device=d_hor.device, dtype=d_hor.dtype)
            )
        else:
            log_terms = torch.zeros_like(d_hor)
        urgency = torch.cat(urgency_terms, dim=0)
        orientation = torch.cat(orientation_terms, dim=0)
        pose = torch.cat(pose_terms, dim=0)
        velocity = torch.cat(velocity_terms, dim=0)

        pos = pos.clone().clamp(
            -_GOAL_STATS_POSITION_CLIP, _GOAL_STATS_POSITION_CLIP
        )
        urgency = urgency.clone().clamp(
            -_GOAL_STATS_VELOCITY_CLIP, _GOAL_STATS_VELOCITY_CLIP
        )
        velocity = velocity.clone().clamp(
            -_GOAL_STATS_VELOCITY_CLIP, _GOAL_STATS_VELOCITY_CLIP
        )

        s_p = 1.0 / torch.clamp(pos.reshape(-1).std(unbiased=False), min=1e-6)
        if self.goal_include_log_d_hor:
            s_l = 1.0 / torch.clamp(
                log_terms.reshape(-1).std(unbiased=False), min=1e-6)
        else:
            s_l = log_terms.new_tensor(1.0)
        s_v = 1.0 / torch.clamp(
            torch.cat((urgency.reshape(-1), velocity.reshape(-1)), dim=0).std(unbiased=False),
            min=1e-6,
        )
        s_o = 1.0 / torch.clamp(
            orientation.std(dim=0, unbiased=False), min=1e-6
        )

        q_start = 13 if motion_dtype.FeatureVersion == 6 else 11
        q_mean = self.mean[q_start:q_start + 29].detach().cpu().clone()
        q_std = self.std[q_start:q_start + 29].detach().cpu().clone()

        pos_scaled = pos * s_p
        log_scaled = log_terms * s_l
        urgency_scaled = urgency * s_v
        velocity_scaled = velocity * s_v
        orientation_scaled = orientation * s_o
        pose_scaled = (pose - q_mean.to(pose.device, pose.dtype)) / q_std.to(
            pose.device, pose.dtype).clamp_min(1e-6)
        pos_p50, pos_p99 = _abs_p50_p99(pos_scaled)
        log_p50, log_p99 = _abs_p50_p99(log_scaled)
        urgency_p50, urgency_p99 = _abs_p50_p99(urgency_scaled)
        velocity_p50, velocity_p99 = _abs_p50_p99(velocity_scaled)
        orientation_p50, orientation_p99 = _abs_p50_p99(orientation_scaled)
        pose_p50, pose_p99 = _abs_p50_p99(pose_scaled)
        s_o_list = [round(float(value), 4) for value in s_o.tolist()]
        logger.info(
            "Goal stats (split55) scales: s_p={:.4f} s_l={:.4f} "
            "s_v={:.4f} s_d={:.4f} s_o={}",
            float(s_p), float(s_l), float(s_v), float(s_d), s_o_list,
        )
        logger.info(
            "Goal stats (split55) scaled |p50/p99|: pos={:.3f}/{:.3f} "
            "log={:.3f}/{:.3f} urg={:.3f}/{:.3f} vel={:.3f}/{:.3f} "
            "ori={:.3f}/{:.3f} pose={:.3f}/{:.3f}",
            pos_p50, pos_p99, log_p50, log_p99, urgency_p50, urgency_p99,
            velocity_p50, velocity_p99, orientation_p50, orientation_p99,
            pose_p50, pose_p99,
        )
        clamp_dist = torch.cat(clamp_dist_terms, dim=0)
        clamp_speed = torch.cat(clamp_speed_terms, dim=0)
        clamp_levels = torch.as_tensor(GOAL_CLAMP_QUANTILE_LEVELS)
        clamp_dist_quantiles = torch.quantile(clamp_dist, clamp_levels)
        clamp_speed_quantiles = torch.quantile(clamp_speed, clamp_levels)
        clamp_levels_percent = clamp_levels * 100.0
        logger.info(
            "Goal stats (split55) clamp distribution over {} windows: "
            "dist quantiles {} m / speed quantiles {} m/s at percentiles {}",
            int(clamp_dist.numel()),
            [round(float(v), 3) for v in clamp_dist_quantiles],
            [round(float(v), 3) for v in clamp_speed_quantiles],
            [round(float(v), 1) for v in clamp_levels_percent],
        )
        return {
            's_p': torch.as_tensor(float(s_p)),
            's_l': torch.as_tensor(float(s_l)),
            's_v': torch.as_tensor(float(s_v)),
            's_d': torch.as_tensor(float(s_d)),
            's_o': s_o.detach().cpu(),
            'q_mean': q_mean,
            'q_std': q_std,
            'goal_clamp': {
                'quantile_levels': [float(v) for v in clamp_levels_percent],
                'dist_quantiles': [float(v) for v in clamp_dist_quantiles],
                'speed_quantiles': [float(v) for v in clamp_speed_quantiles],
                'n_samples': int(clamp_dist.numel()),
            },
            'meta': self._goal_stats_meta(),
        }

    def _compute_goal_stats(self) -> Dict[str, Any]:
        if self.goal_type is not GoalType.JOINT_STATE:
            raise ValueError("goal statistics are only defined for joint_state goals")
        if self.goal_encoding is GoalEncoding.LEGACY40:
            raise ValueError("goal statistics are not required for legacy40 goals")

        saved_mean = self.mean
        saved_std = self.std
        try:
            N = 10000 // self.batch_size + 1
            samples = []
            for i in tqdm(range(N)):
                batch_data = self._generate_batch_optimized(
                    generator=torch.Generator().manual_seed(i)
                )
                samples.extend(batch_data)
            stats = self._goal_stats_from_batch(samples)
        finally:
            self.mean = saved_mean
            self.std = saved_std
            self._stats_device_cache.clear()
        return stats

    def _load_goal_stats(self) -> None:
        if not self.load_goal_stats:
            self.goal_stats = None
            return
        if self.goal_type is not GoalType.JOINT_STATE:
            self.goal_stats = None
            return
        if self.goal_encoding is GoalEncoding.LEGACY40:
            self.goal_stats = None
            return

        goal_stats_path = self.goal_stats_path
        if goal_stats_path.exists():
            logger.info(f" Loading cached goal stats from {goal_stats_path}...")
            goal_stats = torch.load(goal_stats_path, map_location="cpu")
        elif self.split != 'train':
            if not _is_torchrun():
                raise FileNotFoundError(
                    f"Missing goal stats cache {goal_stats_path} for "
                    f"split={self.split!r}"
                )
            # Multi-rank job: the writer rank's train dataset is producing
            # the cache right now; poll until the atomic rename publishes it.
            _wait_for_goal_stats_cache(goal_stats_path)
            goal_stats = torch.load(goal_stats_path, map_location="cpu")
        else:
            logger.info(" Computing goal stats...")
            goal_stats = self._compute_goal_stats()
            _save_goal_stats_cache(goal_stats, goal_stats_path)

        try:
            validate_goal_stats(
                goal_stats,
                goal_encoding=self.goal_encoding,
                goal_offset_range=self.goal_offset_range,
                goal_per_primitive=self.goal_per_primitive,
                future_len=self.future_len,
                fps=float(self.fps),
                goal_timestep_mode=self.goal_timestep_mode,
                datadir=str(self.datadir),
                goal_include_log_d_hor=self.goal_include_log_d_hor,
            )
        except ValueError:
            if self.split != 'train':
                raise
            logger.warning(
                "Cached goal stats at {} do not match the active train "
                "config; recomputing",
                goal_stats_path,
            )
            goal_stats = self._compute_goal_stats()
            _save_goal_stats_cache(goal_stats, goal_stats_path)
        if self.split == 'train' and 'goal_clamp' not in goal_stats:
            logger.warning(
                "Cached goal stats at {} predate the goal-clamp "
                "quantile tables; recomputing",
                goal_stats_path,
            )
            goal_stats = self._compute_goal_stats()
            _save_goal_stats_cache(goal_stats, goal_stats_path)
        self.goal_stats = goal_stats

    def _compute_meanstd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std efficiently"""
        motion_sum = torch.zeros(self.nfeats)
        motion_square_sum = torch.zeros(self.nfeats)
        count = 0

        # Sample a subset for statistics
        N = 10000 // self.batch_size + 1
        # fake a mean and std, so that we can call _generate_batch_optimized
        self.mean = torch.zeros(self.nfeats)
        self.std = torch.ones(self.nfeats)
        self._stats_device_cache.clear()
        # for i in range(N):
        for i in tqdm(range(N)):
            batch_data = self._generate_batch_optimized(generator=torch.Generator().manual_seed(i))

            for primitive_idx in range(self.num_primitive):
                motion_features = batch_data[primitive_idx]['motion']
                motion_sum += motion_features.sum(dim=(0, 1))
                motion_square_sum += motion_features.square().sum(dim=(0, 1))
                count += motion_features.shape[0] * motion_features.shape[1]

        mean = motion_sum / count
        std = (motion_square_sum / count - mean.square()).sqrt()
        return mean, std

    def _compute_meanstd_V2(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_mp_data = []
        for record in self.raw_data:
            seq_data = self._hydrate_sample(record)
            motion_data = seq_data['motion']
            num_frames = motion_data['root_trans_offset'].shape[0]
            primitive_data_list = []
            for start_frame in range(0, num_frames - self.context_len, self.future_len):
                end_frame = start_frame + self.context_len
                primitive_data_list.append(self._extract_single_primitive(seq_data, start_frame, end_frame,
                                            goal_frame=start_frame + self.context_len)['motion'])

            primitive_dict = {}
            for key in MotionKeys:
                primitive_dict[key] = torch.cat([data[key] for data in primitive_data_list], dim=0)

            batch_start_idx = 0
            while batch_start_idx < len(primitive_dict['root_trans_offset']):
                batch_end_idx = min(batch_start_idx + self.batch_size, len(primitive_dict['root_trans_offset']))
                # breakpoint()
                batch_primitive_dict = {key: primitive_dict[key][batch_start_idx:batch_end_idx] for key in MotionKeys}
                motion_tensor = motion_dtype.motion_dict_to_feature(batch_primitive_dict, self.skeleton)[0]
                all_mp_data.append(motion_tensor)
                batch_start_idx = batch_end_idx

        all_mp_data = torch.cat(all_mp_data, dim=0)
        tensor_mean = all_mp_data.mean(dim=[0, 1], keepdim=True)
        tensor_std = all_mp_data.std(dim=[0, 1], keepdim=True)
        return tensor_mean, tensor_std

    # ================================================================
    # Data reconstruction

    def normalize(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize features"""
        mean, std = self._stats_for(feat)
        return (feat - mean) / std

    def denormalize(self, feat: torch.Tensor) -> torch.Tensor:
        """Denormalize features"""
        mean, std = self._stats_for(feat)
        return feat * std + mean

    def _stats_for(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(feat.device), self.mean.dtype, self.std.dtype)
        cached = self._stats_device_cache.get(key)
        if cached is None:
            cached = (self.mean.to(feat.device), self.std.to(feat.device))
            self._stats_device_cache[key] = cached
        return cached

    def reconstruct_motion(
        self,
        motion_feature: torch.Tensor,
        abs_pose: Optional[AbsolutePose] = None,
        need_denormalize: bool = True,
        ret_fk: bool = True,
        ret_fk_full: bool = False,
        sliding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct motion from features"""
        if need_denormalize:
            motion_feature = self.denormalize(motion_feature)

        motion_dict = motion_dtype.motion_feature_to_dict(
            motion_feature, abs_pose, self.skeleton
        )

        if sliding_mask is not None:
            if sliding_mask.shape != motion_dict['contact_mask'].shape:
                raise ValueError(
                    "sliding_mask shape must match contact_mask: "
                    f"{sliding_mask.shape} != {motion_dict['contact_mask'].shape}"
                )
            motion_dict['sliding_mask'] = sliding_mask.to(
                device=motion_feature.device, dtype=torch.float32)

        if ret_fk:
            return self.skeleton.forward_kinematics(
                motion_dict, return_full=ret_fk_full, fps=self.fps
            )
        else:
            return motion_dict

    # ================================================================
    # Sampling from dataset

    def _get_overlap(self, seg1, seg2):
        overlap_len = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
        return overlap_len

    def _primitive_action_label(self, sample: Dict[str, Any],
                                prim_start: int, prim_end: int) -> str:
        """Best-overlap BABEL verb for the primitive window; 'unknown' fallback.

        frame_ann entries are (start_s, end_s, verb, [desc, ...]) in seconds.
        The window is [prim_start, prim_end) frames; units match via fps.
        """
        frame_ann = sample.get('frame_ann')
        if not frame_ann:
            return 'unknown'
        start_t = prim_start / float(self.fps)
        end_t = (prim_end - 1) / float(self.fps)
        best_ann = None
        best_overlap = 0.0
        for ann in frame_ann:
            overlap = self._get_overlap([ann[0], ann[1]], [start_t, end_t])
            if overlap > best_overlap:
                best_overlap = overlap
                best_ann = ann
        return str(best_ann[2]) if best_ann is not None else 'unknown'

    def have_overlap(self, seg1, seg2):
        if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
            return False
        else:
            return True

    def _world_goal_keypoints(self, raw_motion: Dict[str, Any],
                              goal_frame: int) -> torch.Tensor:
        goal_type = GoalType.parse(self.goal_type)
        goal_motion = {
            'dof': self._select_model_dof(torch.as_tensor(
                raw_motion['dof'][goal_frame:goal_frame + 1],
                dtype=torch.float32)),
            'root_trans_offset': torch.as_tensor(
                raw_motion['root_trans_offset'][goal_frame:goal_frame + 1],
                dtype=torch.float32),
            'root_rot': torch.as_tensor(
                raw_motion['root_rot'][goal_frame:goal_frame + 1],
                dtype=torch.float32),
        }
        goal_fk = self.skeleton.forward_kinematics(
            goal_motion, fps=self.fps
        )
        keypoint_ids = (
            self.skeleton.goal_limb_keypoint_id
            if goal_type is GoalType.BODY_EXT
            else self.skeleton.goal_keypoint_id
        )
        return goal_fk['global_translation_extend'][0, 0, keypoint_ids]

    def _world_goal_velocity(self, raw_motion: Dict[str, Any],
                             goal_frame: int) -> torch.Tensor:
        root_position = raw_motion['root_trans_offset']
        if goal_frame <= 0 or goal_frame >= len(root_position):
            raise IndexError(
                f"Goal frame {goal_frame} has no backward-difference frame in "
                f"motion of length {len(root_position)}"
            )
        return (
            torch.as_tensor(root_position[goal_frame], dtype=torch.float32)
            - torch.as_tensor(root_position[goal_frame - 1], dtype=torch.float32)
        ) * float(self.fps)

    def _time_to_arrival_seconds(self, reference_frame: int,
                                 goal_frame: int) -> torch.Tensor:
        """Time from reference frame to goal, in SECONDS.

        Downstream (train/eval/planner) converts this to a frame index via
        round(seconds * fps) before feeding it to the arrival-time PE.
        """
        if self.goal_timestep_mode == "zero":
            return torch.zeros(1, dtype=torch.float32)
        return torch.tensor(
            [max(0.0, (goal_frame - reference_frame) / float(self.fps))],
            dtype=torch.float32,
        )

    def _extract_single_primitive(
        self, sample: Dict[str, Any], prim_start: int, prim_end: int,
        goal_frame: int, world_goal_keypoints: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        """Extract a single primitive from motion data, plus goal+scene fields.

        Returns a dict with:
          - motion: raw MotionDict for the primitive window
          - world_goal_pos / world_goal_yaw: goal in world coordinates
          - history_start_pos / history_start_rot: primitive start absolute pose
          - gt_ref_pos / gt_ref_rot: last-history-frame absolute pose (egocentric reference)
          - scene: per-sequence occupancy grid dict
        """
        goal_type = GoalType.parse(self.goal_type)
        motion_data = {}
        for k in MotionKeys:
            if k in sample['motion']:
                value = torch.tensor(
                    sample['motion'][k][prim_start:prim_end],
                    dtype=torch.float32,
                )
                if k == 'dof':
                    value = self._select_model_dof(value)
                motion_data[k] = value

        raw_sliding_mask = sample['motion'].get('sliding_mask')
        if raw_sliding_mask is None:
            sliding_mask = torch.zeros_like(motion_data['contact_mask'])
        else:
            sliding_mask = torch.as_tensor(
                raw_sliding_mask[prim_start:prim_end], dtype=torch.float32)

        reference_frame = (
            prim_start + self.history_len
            if motion_dtype.FeatureVersion == 6
            else prim_start + self.history_len - 1
        )
        raw_motion = sample['motion']
        goal_rot = torch.as_tensor(raw_motion['root_rot'][goal_frame])

        action_label = self._primitive_action_label(
            sample, prim_start, prim_end)
        text_embedding = self.text_embeddings_dict.get(
            action_label, self.text_embeddings_dict[''])

        primitive = {
            'motion': motion_data,
            'sliding_mask': sliding_mask,
            'world_goal_pos': torch.as_tensor(raw_motion['root_trans_offset'][goal_frame], dtype=torch.float32),
            'world_goal_yaw': quaternion_yaw(goal_rot.float()),
            'history_start_pos': torch.as_tensor(raw_motion['root_trans_offset'][prim_start], dtype=torch.float32),
            'history_start_rot': torch.as_tensor(raw_motion['root_rot'][prim_start], dtype=torch.float32),
            'gt_ref_pos': torch.as_tensor(raw_motion['root_trans_offset'][reference_frame], dtype=torch.float32),
            'gt_ref_rot': torch.as_tensor(raw_motion['root_rot'][reference_frame], dtype=torch.float32),
            'scene': sample.get('scene', {}),
            'is_recovery': bool(sample.get('_recovery_boost', False)),
            'action_label': action_label,
            'text_embedding': text_embedding.float(),
        }
        primitive['_clean_history_delta_q'] = (
            motion_data['dof'][self.history_len]
            - motion_data['dof'][self.history_len - 1]
        ).clone()
        if goal_type.uses_keypoints:
            if world_goal_keypoints is None:
                world_goal_keypoints = self._world_goal_keypoints(
                    raw_motion, goal_frame)
            primitive['world_goal_keypoints'] = world_goal_keypoints
        if goal_type is GoalType.JOINT_STATE:
            primitive['world_goal_rot'] = goal_rot.float()
            primitive['world_goal_dof'] = self._select_model_dof(
                torch.as_tensor(
                    raw_motion['dof'][goal_frame], dtype=torch.float32))
        if goal_type.uses_arrival_time:
            primitive['world_goal_vel'] = self._world_goal_velocity(
                raw_motion, goal_frame)
            # Seconds. Converted to a frame index (round(s * fps)) at the
            # train/eval/planner boundary before the arrival-time PE.
            primitive['time_to_arrival'] = self._time_to_arrival_seconds(
                reference_frame, goal_frame)
            primitive['goal_timestep'] = primitive['time_to_arrival']
        return primitive

    def _generate_motion_primitives(self, sample: Dict[str, Any],
                                    seg_start: int,
                                    goal_offset: Optional[int] = None
                                    ) -> List[Dict[str, Any]]:
        """Generate all primitives from a single motion segment with proper overlapping"""
        if goal_offset is None:
            goal_offset = self.goal_offset
        goal_type = GoalType.parse(self.goal_type)
        # When goal_per_primitive is True, each primitive uses its own last frame
        # as the goal; otherwise, the last frame of the entire snippet is shared.
        snippet_goal_frame = seg_start + self.segment_len - 1 + goal_offset
        world_goal_keypoints = None
        primitives = []

        for primitive_idx in range(self.num_primitive):
            # For proper continuity, each primitive should have overlapping history
            # The key insight: primitive i's last history_len frames should equal
            # primitive i+1's first history_len frames
            prim_start = seg_start + primitive_idx * self.future_len
            prim_end = prim_start + self.future_len + self.history_len + 1

            if self.goal_per_primitive:
                # Goal is the last frame of this specific primitive's window
                goal_frame = (
                    prim_start + self.future_len + self.history_len
                    if motion_dtype.FeatureVersion == 6
                    else prim_start + self.future_len + self.history_len - 1
                )
                goal_frame = (
                    goal_frame
                    + goal_offset
                )
                world_goal_keypoints = None
            else:
                goal_frame = snippet_goal_frame

            clip_len = len(sample['motion']['root_trans_offset'])
            reference_frame = (
                prim_start + self.history_len
                if motion_dtype.FeatureVersion == 6
                else prim_start + self.history_len - 1
            )
            if goal_frame <= reference_frame:
                raise IndexError(
                    f"Goal frame {goal_frame} must follow primitive reference "
                    f"frame {reference_frame}"
                )
            required_last_frame = goal_frame
            if required_last_frame >= clip_len:
                raise IndexError(
                    f"Goal frame {goal_frame} exceeds motion bounds for "
                    f"goal_type={goal_type.value!r}, length={clip_len}"
                )

            if goal_type.uses_keypoints and world_goal_keypoints is None:
                world_goal_keypoints = self._world_goal_keypoints(
                    sample['motion'], goal_frame)

            primitives.append(self._extract_single_primitive(
                sample, prim_start, prim_end, goal_frame,
                world_goal_keypoints=world_goal_keypoints))

        return primitives

    def _sample_motion_batch(self,
                             generator: Optional[torch.Generator
                                                ] = None) -> List[List[Dict[str, Any]]]:
        """Sample a batch of motions and generate all their primitives"""

        if not self.weighted_sample:
            rand_idx = torch.randint(0, len(self.valid_indices), (self.batch_size, ), generator=generator)
        else:
            # Use the worker-local Torch generator for reproducible weighted
            # sequence selection; NumPy's global RNG ignored the caller seed.
            rand_idx = torch.multinomial(
                torch.as_tensor(self.seq_weights, dtype=torch.float64),
                self.batch_size, replacement=True, generator=generator,
            )

        all_motion_primitives = []
        for batch_idx in range(self.batch_size):
            # Get sample
            sample_idx = self.valid_indices[rand_idx[batch_idx].item()]  # type:ignore
            sample = self._hydrate_sample(self.raw_data[sample_idx])

            # Sample segment start ONCE per motion using the generator for reproducibility
            max_start = sample['length'] - self.required_length

            # seg_start = int(
            #         torch.randint(0, max_start, (1, ), generator=generator).item())

            if self.weighted_sample and self.frame_weight:
                seg_start = random.choices(range(max_start + 1), weights=sample['frame_weights'][:max_start + 1], k=1)[0]
            else:
                seg_start = int(torch.randint(0, max_start + 1, (1, ), generator=generator).item())

            # Generate ALL primitives for this motion using the SAME seg_start
            if self.min_goal_offset == self.max_goal_offset:
                goal_offset = self.min_goal_offset
            else:
                goal_offset = int(torch.randint(
                    self.min_goal_offset,
                    self.max_goal_offset + 1,
                    (1, ),
                    generator=generator,
                ).item())
            motion_primitives = self._generate_motion_primitives(
                sample, seg_start, goal_offset)
            all_motion_primitives.append(motion_primitives)

        return all_motion_primitives

    def _organize_primitives_by_index(
        self, all_motion_primitives: List[List[Dict[str, Any]]],
        generator: Optional[torch.Generator] = None,
    ) -> List[Dict[str, Any]]:
        """Organize primitives by primitive index for batching.

        Returns a list of dicts (one per primitive index), each containing:
          - motion: normalized VAE features [B, T, nfeats]
          - scene: list of per-sample occupancy dicts
          - world_goal_pos, world_goal_yaw, history_start_pos, ...
        """
        goal_type = GoalType.parse(self.goal_type)
        tensor_keys = (
            'world_goal_pos', 'world_goal_yaw',
            'history_start_pos', 'history_start_rot',
            'gt_ref_pos', 'gt_ref_rot',
        )
        if goal_type.uses_keypoints:
            tensor_keys += ('world_goal_keypoints',)
        if goal_type is GoalType.JOINT_STATE:
            tensor_keys += ('world_goal_rot', 'world_goal_dof')
        if goal_type.uses_arrival_time:
            tensor_keys += ('world_goal_vel', 'time_to_arrival',
                            'goal_timestep')
        batch_primitives = []

        for primitive_idx in range(self.num_primitive):
            # Collect motion data for this primitive across the batch
            motion_batch = []
            primitives = []

            for batch_idx in range(self.batch_size):
                primitive = all_motion_primitives[batch_idx][primitive_idx]
                primitive['_augmented'] = self._augment_raw_motion(
                    primitive['motion'],
                    generator)
                motion_batch.append(primitive['motion'])
                primitives.append(primitive)

            # Convert to tensors and motion features
            motion_features = self._convert_to_motion_features(motion_batch)
            feature_len = motion_features.shape[1]
            recovery_flags = [
                bool(p.get('is_recovery', False)) for p in primitives
            ]
            for b, primitive in enumerate(primitives):
                if primitive['_augmented']:
                    if motion_dtype.FeatureVersion == 3:
                        # Keep the final history delta clean: it references the
                        # first future pose, which is outside the perturbed window.
                        delta_start = 11 + self.dof_dim
                        clean_delta = self._select_model_dof(
                            primitive['_clean_history_delta_q'].unsqueeze(0))[0]
                        motion_features[b, self.history_len - 1,
                                        delta_start:delta_start + self.dof_dim] = clean_delta
                    # Goals stay clean, but ego-centric conditioning is reset to
                    # the perturbed latest history state (v1 contract).
                    ref = (
                        self.history_len
                        if motion_dtype.FeatureVersion == 6
                        else self.history_len - 1
                    )
                    primitives[b]['gt_ref_rot'] = primitive['motion']['root_rot'][ref].clone()
                    primitives[b]['gt_ref_pos'] = primitive['motion']['root_trans_offset'][ref].clone()

                    # Frame-0 anchor consistency: history_start_* must match the
                    # feature-implied pose at the primitive's first frame, since
                    # _next_rollout_poses integrates generated motion from it.
                    primitives[b]['history_start_rot'] = primitive['motion']['root_rot'][0].clone()
                    primitives[b]['history_start_pos'] = primitive['motion']['root_trans_offset'][0].clone()

            batch = {
                'motion': self.normalize(motion_features),
                'sliding_mask': torch.stack([
                    (
                        p['sliding_mask'][1:feature_len + 1]
                        if motion_dtype.FeatureVersion == 6
                        else p['sliding_mask'][:feature_len]
                    )
                    for p in primitives
                ]),
                'scene': [p['scene'] for p in primitives],
                'action_label': [
                    p.get('action_label') for p in primitives
                ],
                'text_embedding': torch.stack([
                    p['text_embedding'] for p in primitives
                ]),
                'is_recovery': torch.as_tensor(recovery_flags, dtype=torch.bool),
            }
            batch.update({
                key: torch.stack([p[key] for p in primitives])
                for key in tensor_keys
            })
            batch_primitives.append(batch)

        return batch_primitives

    def _convert_to_motion_features(self, motion_batch: List[MotionDict]) -> torch.Tensor:
        """Convert batch of motion data to motion features"""
        # Stack motion tensors
        motion_tensors = {}
        for k in MotionKeys:
            motion_tensors[k] = torch.stack([m[k] for m in motion_batch])

        motion_features, _ = motion_dtype.motion_dict_to_feature(motion_tensors, self.skeleton)

        return motion_features

    def _generate_batch_optimized(self,
                                  generator: Optional[torch.Generator
                                                     ] = None) -> List[Dict[str, Any]]:
        """Generate a batch using motion-first approach"""
        # Step 1: Sample motions and generate all their primitives
        all_motion_primitives = self._sample_motion_batch(generator)

        # Step 2: Organize primitives by index for batching
        batch_primitives = self._organize_primitives_by_index(
            all_motion_primitives, generator)

        return batch_primitives

    def __iter__(self):
        """Iterator that yields batches in the expected format"""
        worker_info = data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        generator = torch.Generator()
        # Each DDP rank gets a distinct data stream via rank offset; each worker
        # within a rank is further offset.  np.random.randint adds per-iterator-
        # instance variation so that different runs (or resumed runs with a fresh
        # iterator) naturally see different sample orderings without needing an
        # explicit epoch counter — this training loop uses stages, not epochs.
        seed_offset = self.rank + worker_id * self.world_size + np.random.randint(0, 1000000)
        generator.manual_seed(seed_offset)

        while True:
            yield self._generate_batch_optimized(generator=generator)

    def __len__(self) -> int:
        return len(self.valid_indices)
