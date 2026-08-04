"""
Final Clean High-Performance Data Loader for Robot Motion Primitive Dataset

This is a clean, well-structured implementation with:
1. Motion-first generation ensuring primitive continuity
2. Small, focused functions for better readability
3. 100% interface compatibility with original
"""

from collections import OrderedDict
from pathlib import Path
import numpy as np
import joblib
import yaml
from typing import Any, Tuple, Dict, List, Optional
import sys
import random
from omegaconf import DictConfig
from loguru import logger

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
# from robotmdar.model.clip import load_and_freeze_clip, encode_text
from robotmdar.skeleton.robot import RobotSkeleton
from robotmdar.utils.goal import GoalType, quaternion_yaw
from robotmdar.dtype.motion import MotionDict, motion_dict_to_feature, AbsolutePose, motion_feature_to_dict, MotionKeys
import json


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
        goal_per_primitive: bool = False,
        goal_timestep_mode: str = "relative",
        weighted_sample: bool = False,
        frame_weight: bool = False,
        use_weighted_meanstd: bool = False,
        split: str = 'train',
        device: str = 'cuda',
        **kwargs: Any
    ):
        super().__init__()
        # Store parameters
        self.batch_size = batch_size
        self.history_len = history_len
        self.future_len = future_len
        self.num_primitive = num_primitive

        self.nfeats = nfeats

        self.segment_len = self.history_len + self.future_len * self.num_primitive + 1
        self.context_len = self.history_len + self.future_len

        self.goal_type = GoalType.parse(goal_type)
        self.goal_per_primitive = goal_per_primitive
        self.goal_timestep_mode = str(goal_timestep_mode).lower()
        if self.goal_timestep_mode not in ("relative", "zero"):
            raise ValueError(
                "goal_timestep_mode must be 'relative' or 'zero', got "
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

        # BODY_EXT uses a forward difference at the goal. A shared snippet goal
        # sits on the final raw frame, so it needs one additional source frame.
        forward_diff_extra = int(
            self.goal_type is GoalType.BODY_EXT and not self.goal_per_primitive
        )
        self.required_length = self.segment_len + max(
            0, self.max_goal_offset + forward_diff_extra
        )
        self.weighted_sample = weighted_sample
        self.frame_weight = frame_weight
        self.action_statistics_path = action_statistics_path
        self.use_weighted_meanstd = use_weighted_meanstd

        self.datadir = Path(datadir)
        self.split = split
        self.device = "cpu"  # Keep embeddings on CPU initially
        self.sample_cache_size = max(0, int(kwargs.get('sample_cache_size', 8)))
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

        logger.info(f" Found {len(self.valid_indices)} valid samples out of {len(self.raw_data)}")

        # Load text embeddings
        # DEPRECATED: text embeddings no longer used (goal+scene conditioning).
        # self._load_text_embeddings()

    def _hydrate_sample(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Load a manifest-backed sample, with a small per-worker LRU cache."""
        relpath = record.get('_data_path')
        if relpath is None:
            return record

        sample_path = self.datadir / relpath
        cache = getattr(self, '_sample_cache', None)
        if cache is None:
            cache = self._sample_cache = OrderedDict()

        cache_key = str(sample_path)
        if cache_key in cache:
            stored = cache.pop(cache_key)
            cache[cache_key] = stored
        else:
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"Manifest sample does not exist: {sample_path}"
                )
            stored = joblib.load(sample_path)
            if not isinstance(stored, dict) or 'motion' not in stored:
                raise ValueError(f"Invalid manifest sample: {sample_path}")
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

    # =========================================================================
    # DEPRECATED: Text embedding loading & computation.
    # TextOp's original text-conditioned pipeline (CLIP → Denoiser) has been
    # replaced by goal + scene conditioning per LDM_goal_scene_design.md.
    # These methods are preserved for reference / future text-based ablation
    # experiments — DO NOT DELETE. The active code path uses zero tensors
    # for the embedding slot (see _extract_single_primitive).
    # =========================================================================
    # def _load_text_embeddings(self) -> None:
    #     """Load or compute text embeddings"""
    #     text_embedding_path = self.datadir / f'{self.split}_text_embed.pkl'
    #     if text_embedding_path.exists():
    #         logger.info(" Loading cached text embeddings...")
    #         self.text_embeddings_dict = torch.load(text_embedding_path, map_location="cpu")
    #     else:
    #         logger.info(" Computing text embeddings...")
    #         clip_model = load_and_freeze_clip(
    #             clip_version='ViT-B/32', device="cuda" if torch.cuda.is_available() else "cpu"
    #         )
    #         self.text_embeddings_dict = self._compute_text_embeddings(self.raw_data, clip_model)
    #         torch.save(self.text_embeddings_dict, text_embedding_path)
    #
    # @staticmethod
    # def _compute_text_embeddings(raw_data: List[Dict[str, Any]],
    #                              clip_model: nn.Module,
    #                              batch_size: int = 64) -> Dict[str, torch.Tensor]:
    #     """Compute text embeddings efficiently"""
    #     # Extract all unique texts
    #     all_texts = set()
    #     for item in raw_data:
    #         for ann in item['frame_ann']:
    #             all_texts.add(ann[2])
    #
    #     uni_texts = list(all_texts)
    #
    #     # Batch encode
    #     embeddings_list = []
    #     for i in range(0, len(uni_texts), batch_size):
    #         batch_texts = uni_texts[i:i + batch_size]
    #         batch_embeddings = encode_text(clip_model, batch_texts)
    #         embeddings_list.append(batch_embeddings.detach().float())
    #
    #     text_embeddings = torch.cat(embeddings_list, dim=0)
    #
    #     # Create dictionary
    #     text_embeddings_dict = dict(zip(uni_texts, text_embeddings))
    #     text_embeddings_dict[''] = torch.zeros_like(text_embeddings[0])
    #
    #     return text_embeddings_dict

    def _load_meanstd(self) -> None:
        """Load or compute mean/std for normalization"""
        meanstd_cache_path = self.datadir / 'meanstd.pkl'
        if meanstd_cache_path.exists():
            logger.info(f" Loading cached mean/std from {meanstd_cache_path}...")
            meanstd = torch.load(meanstd_cache_path, map_location="cpu")
        else:
            logger.info(f" Computing mean/std..")
            assert self.split == 'train', "Compute mean and std from 'train' set"

            # zjk: DART meanstd cal method
            meanstd = self._compute_meanstd()
            # meanstd = self._compute_meanstd_V2()

            torch.save(meanstd, meanstd_cache_path)
            logger.info(f" Saved mean/std to {meanstd_cache_path}")

        self.mean, self.std = meanstd

    def _load_weighted_meanstd(self) -> None:
        """Load or compute mean/std for normalization"""
        meanstd_cache_path = self.datadir / 'weighted_meanstd.pkl'
        if meanstd_cache_path.exists():
            logger.info(f" Loading cached mean/std from {meanstd_cache_path}...")
            meanstd = torch.load(meanstd_cache_path, map_location="cpu")
        else:
            logger.info(f" Computing mean/std..")
            assert self.split == 'train', "Compute mean and std from 'train' set"

            # zjk: DART meanstd cal method
            meanstd = self._compute_meanstd()
            # meanstd = self._compute_meanstd_V2()

            torch.save(meanstd, meanstd_cache_path)
            logger.info(f" Saved mean/std to {meanstd_cache_path}")

        self.mean, self.std = meanstd

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
                motion_tensor = motion_dict_to_feature(batch_primitive_dict)[0]
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
        return (feat - self.mean.to(feat.device)) / self.std.to(feat.device)

    def denormalize(self, feat: torch.Tensor) -> torch.Tensor:
        """Denormalize features"""
        return feat * self.std.to(feat.device) + self.mean.to(feat.device)

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

        if motion_feature_to_dict.__name__ == 'motion_dict_to_feature_v4':
            motion_dict = motion_feature_to_dict(motion_feature, abs_pose, self.skeleton)
        else:
            motion_dict = motion_feature_to_dict(motion_feature, abs_pose)

        if sliding_mask is not None:
            if sliding_mask.shape != motion_dict['contact_mask'].shape:
                raise ValueError(
                    "sliding_mask shape must match contact_mask: "
                    f"{sliding_mask.shape} != {motion_dict['contact_mask'].shape}"
                )
            motion_dict['sliding_mask'] = sliding_mask.to(
                device=motion_feature.device, dtype=torch.float32)

        if ret_fk:
            return self.skeleton.forward_kinematics(motion_dict, return_full=ret_fk_full)
        else:
            return motion_dict

    # ================================================================
    # Sampling from dataset

    def _get_overlap(self, seg1, seg2):
        overlap_len = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
        return overlap_len

    def have_overlap(self, seg1, seg2):
        if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
            return False
        else:
            return True

    def _world_goal_keypoints(self, raw_motion: Dict[str, Any],
                              goal_frame: int) -> torch.Tensor:
        goal_motion = {
            'dof': torch.as_tensor(
                raw_motion['dof'][goal_frame:goal_frame + 1],
                dtype=torch.float32),
            'root_trans_offset': torch.as_tensor(
                raw_motion['root_trans_offset'][goal_frame:goal_frame + 1],
                dtype=torch.float32),
            'root_rot': torch.as_tensor(
                raw_motion['root_rot'][goal_frame:goal_frame + 1],
                dtype=torch.float32),
        }
        goal_fk = self.skeleton.forward_kinematics(goal_motion)
        keypoint_ids = (
            self.skeleton.goal_limb_keypoint_id
            if self.goal_type is GoalType.BODY_EXT
            else self.skeleton.goal_keypoint_id
        )
        return goal_fk['global_translation_extend'][0, 0, keypoint_ids]

    def _world_goal_velocity(self, raw_motion: Dict[str, Any],
                             goal_frame: int) -> torch.Tensor:
        root_position = raw_motion['root_trans_offset']
        if goal_frame < 0 or goal_frame + 1 >= len(root_position):
            raise IndexError(
                f"Goal frame {goal_frame} has no forward-difference frame in "
                f"motion of length {len(root_position)}"
            )
        return (
            torch.as_tensor(root_position[goal_frame + 1], dtype=torch.float32)
            - torch.as_tensor(root_position[goal_frame], dtype=torch.float32)
        ) * float(self.fps)

    def _goal_timestep(self, reference_frame: int,
                       goal_frame: int) -> torch.Tensor:
        if self.goal_timestep_mode == "zero":
            return torch.zeros(1, dtype=torch.float32)
        return torch.tensor(
            [(goal_frame - reference_frame) / float(self.fps)],
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
        motion_data = {}
        for k in MotionKeys:
            if k in sample['motion']:
                motion_data[k] = torch.tensor(sample['motion'][k][prim_start:prim_end], dtype=torch.float32)

        raw_sliding_mask = sample['motion'].get('sliding_mask')
        if raw_sliding_mask is None:
            sliding_mask = torch.zeros_like(motion_data['contact_mask'])
        else:
            sliding_mask = torch.as_tensor(
                raw_sliding_mask[prim_start:prim_end], dtype=torch.float32)

        reference_frame = prim_start + self.history_len - 1
        raw_motion = sample['motion']
        goal_rot = torch.as_tensor(raw_motion['root_rot'][goal_frame])

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
        }
        if self.goal_type.uses_keypoints:
            if world_goal_keypoints is None:
                world_goal_keypoints = self._world_goal_keypoints(
                    raw_motion, goal_frame)
            primitive['world_goal_keypoints'] = world_goal_keypoints
        if self.goal_type is GoalType.BODY_EXT:
            primitive['world_goal_vel'] = self._world_goal_velocity(
                raw_motion, goal_frame)
            primitive['goal_timestep'] = self._goal_timestep(
                reference_frame, goal_frame)
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
                    prim_start + self.future_len + self.history_len - 1
                    + goal_offset
                )
                world_goal_keypoints = None
            else:
                goal_frame = snippet_goal_frame

            clip_len = len(sample['motion']['root_trans_offset'])
            if goal_frame <= prim_start + self.history_len - 1:
                raise IndexError(
                    f"Goal frame {goal_frame} must follow primitive reference "
                    f"frame {prim_start + self.history_len - 1}"
                )
            required_last_frame = goal_frame + int(
                goal_type is GoalType.BODY_EXT
            )
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
            rand_idx = torch.from_numpy(
                np.random.choice(len(self.valid_indices), size=self.batch_size, replace=True, p=self.seq_weights)
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
        self, all_motion_primitives: List[List[Dict[str, Any]]]
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
        if goal_type is GoalType.BODY_EXT:
            tensor_keys += ('world_goal_vel', 'goal_timestep')
        batch_primitives = []

        for primitive_idx in range(self.num_primitive):
            # Collect motion data for this primitive across the batch
            motion_batch = []
            primitives = []

            for batch_idx in range(self.batch_size):
                primitive = all_motion_primitives[batch_idx][primitive_idx]
                motion_batch.append(primitive['motion'])
                primitives.append(primitive)

            # Convert to tensors and motion features
            motion_features = self._convert_to_motion_features(motion_batch)
            feature_len = motion_features.shape[1]

            batch = {
                'motion': self.normalize(motion_features),
                'sliding_mask': torch.stack([
                    p['sliding_mask'][:feature_len] for p in primitives
                ]),
                'scene': [p['scene'] for p in primitives],
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

        motion_features, _ = motion_dict_to_feature(motion_tensors, self.skeleton)

        return motion_features

    def _generate_batch_optimized(self,
                                  generator: Optional[torch.Generator
                                                     ] = None) -> List[Dict[str, Any]]:
        """Generate a batch using motion-first approach"""
        # Step 1: Sample motions and generate all their primitives
        all_motion_primitives = self._sample_motion_batch(generator)

        # Step 2: Organize primitives by index for batching
        batch_primitives = self._organize_primitives_by_index(all_motion_primitives)

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
