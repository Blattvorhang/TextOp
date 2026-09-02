import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import errno
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
import copy

import robotmdar.dtype.motion as motion_dtype
from robotmdar.dtype.motion import perturb_feature_v3
from robotmdar.train.loss import (
    GeometryLoss,
    GoalType,
    JOINT_STATE_GOAL_DIM,
    MOTION_CLASSES,
    _add_per_class_extras,
    _motion_class_labels,
    _standard_normal_kl_mean,
    calc_dar_loss,
    calc_mvae_loss,
)

# Tensorboard tag routing (doc §4.3.6). Log-only scalars are classified into
# four groups instead of the former flat loss/extras split:
#   loss          -> loss/{train,eval}/<k>            (objective terms)
#   metric        -> metric/{train,eval}/<k>          (unweighted diagnostics)
#   metric_class  -> metric_class/{train,eval}/<base>/<cls>  (per-class means)
#   meta          -> meta/<k>                         (schedule state, no phase)
_META_KEYS = frozenset(
    {'stage', 'scene_active', 'augmentation_active', 'feature_version',
     'lr', 'grad_norm', 'eval_time'})


def _classify_extra(name: str, phase: str) -> Tuple[str, str]:
    """Route a log-only scalar key to a (group, tag) pair.

    ``phase`` is 'train' or 'eval'; meta keys ignore it (no per-phase
    split). Per-class extras use the ``base__cls`` key layout and become
    nested ``base/cls`` tags.
    """
    if name in _META_KEYS:
        return 'meta', name
    if '__' in name:
        base, cls = name.split('__', 1)
        return 'metric_class', f'{phase}/{base}/{cls}'
    return 'metric', f'{phase}/{name}'


def is_main_process():
    """Check if current process is the main (rank 0) process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_ddp_model(model):
    """Unwrap DDP model to get the underlying model."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _accumulate_metric(total: dict, key: str, value) -> None:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        total[key] = total[key] + value if key in total else value.clone()
    elif isinstance(value, (int, float)):
        total[key] = total.get(key, 0.0) + value
    else:
        total[key] = value


def _report_value(value, divisor: float = 1.0):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, (int, float)):
        return value / divisor
    return value / divisor if divisor != 1.0 else value


def ddp_reduce_mean(tensor_dict: dict) -> dict:
    """All-reduce mean across all DDP ranks for each tensor in the dict.

    Returns a new dict where every tensor value is the average across all
    ranks. Key sets may differ across ranks (data-dependent extras such as
    per-class tags): the union of all ranks' tensor keys is gathered once
    and each union key is reduced in canonical sorted order, so every rank
    issues the identical sequence of collectives and the NCCL streams never
    diverge. A key missing (or non-tensor) on a rank contributes zeros of
    the gathered shape/dtype; inconsistent shapes across ranks raise
    instead of deadlocking. Handles both CPU and CUDA tensors (NCCL
    requires CUDA tensors).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor_dict

    world_size = dist.get_world_size()
    if world_size <= 1:
        return tensor_dict

    # Gather each rank's tensor shape/dtype spec to build the union and
    # verify cross-rank consistency (a genuine bug would otherwise present
    # as an NCCL timeout).
    local_spec = {
        k: (tuple(v.shape), str(v.dtype))
        for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)
    }
    gathered_specs = [None] * world_size
    dist.all_gather_object(gathered_specs, local_spec)

    spec = {}
    for rank_spec in gathered_specs:
        for k, shape_dtype in rank_spec.items():
            if k in spec and spec[k] != shape_dtype:
                raise ValueError(
                    f'ddp_reduce_mean: key {k!r} has inconsistent '
                    f'shape/dtype across ranks: {spec[k]} vs {shape_dtype}')
            spec[k] = shape_dtype

    union_keys = sorted(
        set(spec) | {k for k, v in tensor_dict.items()
                     if not isinstance(v, torch.Tensor)})

    reduced = {
        k: tensor_dict.get(k)
        for k in union_keys
        if k not in spec
    }
    tensor_groups = defaultdict(list)
    for k in union_keys:
        if k in spec:
            shape, dtype_str = spec[k]
            tensor_groups[(shape, dtype_str)].append(k)

    backend = dist.get_backend()
    reduce_device = torch.device('cuda' if backend == 'nccl' else 'cpu')
    for (shape, dtype_str), keys in tensor_groups.items():
        dtype = getattr(torch, dtype_str.split('.')[-1])
        packed_values = []
        was_cpu = {}
        for k in keys:
            v = tensor_dict.get(k)
            if isinstance(v, torch.Tensor):
                was_cpu[k] = v.device.type == 'cpu'
                value = v.detach().to(device=reduce_device, dtype=dtype)
            else:
                was_cpu[k] = False
                value = torch.zeros(shape, dtype=dtype, device=reduce_device)
            packed_values.append(value.reshape(-1))
        packed = torch.stack(packed_values, dim=0)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed = packed / world_size
        for row, k in zip(packed, keys):
            value = row.reshape(shape)
            if was_cpu[k]:
                value = value.cpu()
            reduced[k] = value
    return reduced


class BaseManager(ABC):
    """
    Abstract base class for training managers.

    Defines common interfaces and functionality for training management,
    including EMA support, history selection, and training lifecycle management.
    """
    optimizer: torch.optim.Optimizer
    dataset: Any

    max_steps: int
    stages: List[int]
    stage_idx: int
    # -1 for not started, 0 for first stage, 1 for second stage, etc.
    use_rollout: bool

    use_static_pose: bool

    anneal_lr: bool
    learning_rate: float
    max_grad_norm: float
    loss_weight: Dict[str, float]
    extra: Dict[str, float]
    ckpt: DictConfig
    device: str
    platform: Any
    save_every: int
    eval_every: int
    eval_steps: int

    def __init__(self, **kwargs):
        # 自动保存所有参数为成员变量
        for k, v in kwargs.items():
            setattr(self, k, v)

        # DDP rank and world_size, set externally after instantiation
        self.rank = 0
        self.world_size = 1

        # 设置默认值
        self.stage_idx = -1
        self._stage_steps = torch.cumsum(torch.tensor(self.stages).int(), dim=0)
        # assert self._stage_steps[
        #     -1] == self.max_steps, "Stage steps must sum to max_steps"
        self.max_steps = int(self._stage_steps[-1])

        # Scene conditioning curriculum: before scene_start_step global steps
        # the denoiser trains on blank scenes (see should_use_scene). The
        # value comes from the data config (data/mob.yaml), passed at
        # instantiation; keep the 0 fallback for managers that never see it.
        self.scene_start_step = int(getattr(self, 'scene_start_step', 0))

        self.step = 0

        self._to_eval_steps = 0
        self._total_eval_loss_dict = {}
        self._total_eval_extras_dict = {}
        self._eval_t0 = None
        self.rec_criterion = nn.HuberLoss(reduction='mean', delta=1.0)

        self.save_dir = Path(self.save_dir)
        self._tqdm = None
        self.extra = {}

        # EMA相关
        self.use_ema = getattr(self, 'use_ema', False)
        self.ema_decay = getattr(self, 'ema_decay', 0.999)
        self.ema_models = {}

        # History选择相关
        self.static_prob = getattr(self, 'static_prob', 0.0)

    def pre_step(self, is_eval: bool = False) -> None:
        """每步训练前调用"""
        self.stage_idx = min(
            torch.searchsorted(
                self._stage_steps, self.step, right=True, out_int32=True
            ).item(),
            len(self.stages) - 1,
        )  # type:ignore
        self.extra['stage'] = self.stage_idx
        self.extra['scene_active'] = float(self.should_use_scene())
        self.extra['augmentation_active'] = float(self.should_use_augmentation())
        self.extra['feature_version'] = float(motion_dtype.FeatureVersion)

        if is_eval and self._eval_t0 is None:
            self._eval_t0 = time.perf_counter()

        if not self._tqdm and is_main_process():
            self._tqdm = tqdm(total=self.max_steps, initial=self.step, ncols=120, desc="Training")
        if self.anneal_lr:
            frac = 1.0 - self.step / self.max_steps
            lrnow = frac * self.learning_rate
            self.extra['lr'] = lrnow
            self.optimizer.param_groups[0]["lr"] = lrnow
        else:
            lrnow = self.learning_rate
            self.extra['lr'] = lrnow
            self.optimizer.param_groups[0]["lr"] = lrnow

    def begin_eval_cycle(self) -> None:
        """Mark the start of a validation cycle for wall-clock timing."""
        self._eval_t0 = time.perf_counter()

    def post_step(
        self,
        is_eval: bool = False,
        loss_dict: Dict[str, torch.Tensor] = {},
        extras: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """每步训练后调用"""
        if extras is None:
            extras = {}

        if is_eval:
            self._to_eval_steps -= 1
            for k, v in loss_dict.items():
                _accumulate_metric(self._total_eval_loss_dict, k, v)
            for k, v in extras.items():
                _accumulate_metric(self._total_eval_extras_dict, k, v)
            if self._to_eval_steps == 0:
                if self._eval_t0 is not None:
                    self._total_eval_extras_dict['eval_time'] = (
                        time.perf_counter() - self._eval_t0
                    )
                # All-reduce accumulated eval metrics across all DDP ranks
                reduced_loss = ddp_reduce_mean(self._total_eval_loss_dict)
                reduced_extras = ddp_reduce_mean(self._total_eval_extras_dict)
                if is_main_process():
                    for k, v in reduced_loss.items():
                        self.platform.report_scalar(
                            "eval/" + k, _report_value(v, self.eval_steps),
                            self.step, group_name="loss"
                        )
                    for k, v in reduced_extras.items():
                        group, tag = _classify_extra(k, 'eval')
                        divisor = 1.0 if k == 'eval_time' else self.eval_steps
                        self.platform.report_scalar(
                            tag, _report_value(v, divisor),
                            self.step,
                            group_name=group
                        )
                    tqdm.write(
                        f"Eval finished at step {self.step} with loss * {self.eval_steps}: {dict(reduced_loss)}"
                    )
                    if self._total_eval_extras_dict:
                        tqdm.write(f"Eval extras * {self.eval_steps}: {dict(reduced_extras)}")
                self._total_eval_loss_dict = {}
                self._total_eval_extras_dict = {}
                self._eval_t0 = None
            return

        self.step += 1

        if is_main_process():
            assert self._tqdm is not None
            self._tqdm.update(1)
            self._tqdm.set_postfix({
                'stage': self.stage_idx,
                'loss': _report_value(loss_dict["total"]),
                'lr': self.extra['lr'],
            })
        if self.step >= self.max_steps and self._tqdm is not None:
            self._tqdm.close()

        # Train loss: log per-rank values directly (no all-reduce).
        # Per-rank loss is close to the global average — the trend is what matters.
        # all-reduce on every step would add ~50ms NCCL overhead per 10+ scalar keys.
        if is_main_process():
            for k, v in loss_dict.items():
                self.platform.report_scalar(
                    "train/" + k, _report_value(v),
                    self.step, group_name="loss")
            for k, v in extras.items():
                group, tag = _classify_extra(k, 'train')
                self.platform.report_scalar(
                    tag, _report_value(v), self.step, group_name=group
                )
            for k, v in self.extra.items():
                group, tag = _classify_extra(k, 'train')
                self.platform.report_scalar(tag, v, self.step, group_name=group)

        # 更新EMA模型
        self.update_ema_models()

        if self.step % self.save_every == 0 or self.step == self.max_steps:
            self.save_model()

        if (self.step % self.eval_every == 0 or self.step == self.max_steps):
            self._to_eval_steps = self.eval_steps
            self._total_eval_loss_dict = {}
            self._total_eval_extras_dict = {}
            if is_main_process():
                details = []
                if hasattr(self, 'eval_full_sample'):
                    details.append(f"eval_full_sample={self.eval_full_sample}")
                tqdm.write(
                    f"Eval starting at step {self.step}: "
                    f"{self.eval_steps} primitive steps"
                    + (f" ({', '.join(details)})" if details else "")
                )

    def should_eval(self) -> bool:
        """是否需要评估"""
        return self._to_eval_steps > 0

    def should_report_eval_visualization(self) -> bool:
        """Report one visualization batch per eval cycle, from DDP rank zero."""
        return (
            self._to_eval_steps == self.eval_steps
            and os.environ.get('RANK', '0') == '0'
        )

    def grad_clip(self, model):
        """
        Apply gradient clipping to model parameters.

        Args:
            model: PyTorch model to clip gradients for
        """
        self.clip_grad_and_check(model)

    def clip_grad_and_check(self, model) -> bool:
        """Clip gradients once and return whether all grads are finite."""
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            self.extra['grad_norm'] = torch.tensor(0.0)
            return True

        if self.max_grad_norm > 0:
            norm = nn.utils.clip_grad_norm_(
                params, self.max_grad_norm, error_if_nonfinite=False)
        else:
            grad_norms = [
                torch.linalg.vector_norm(p.grad.detach(), ord=2)
                for p in params
            ]
            device = grad_norms[0].device
            norm = torch.linalg.vector_norm(
                torch.stack([value.to(device) for value in grad_norms]),
                ord=2,
            )
        self.extra['grad_norm'] = norm.detach()
        return bool(torch.isfinite(norm).item())

    def __bool__(self):
        """Check if training should continue."""
        return self.step < self.max_steps

    def register_ema_model(self, name: str, model: nn.Module):
        """
        Register a model for EMA (Exponential Moving Average) tracking.

        Args:
            name: Unique identifier for the model
            model: PyTorch model to track with EMA
        """
        if self.use_ema:
            ema_model = copy.deepcopy(model)
            for param in ema_model.parameters():
                param.requires_grad = False
            self.ema_models[name] = ema_model
            logger.info(f"Registered EMA model: {name}")

    def update_ema(self, name: str, model: nn.Module):
        """
        Update EMA model parameters using exponential moving average.

        Args:
            name: Model identifier
            model: Current model to update EMA from
        """
        if self.use_ema and name in self.ema_models:
            ema_model = self.ema_models[name]
            for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def get_ema_model(self, name: str) -> Optional[nn.Module]:
        """
        Get EMA model by name.

        Args:
            name: Model identifier

        Returns:
            EMA model if exists, None otherwise
        """
        if self.use_ema and name in self.ema_models:
            return self.ema_models[name]
        return None

    @abstractmethod
    def hold_model(self, *args, **kwargs):
        """子类实现: 绑定模型和优化器"""
        pass

    @abstractmethod
    def save_model(self) -> None:
        ...

    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def calc_loss(self, *args, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """子类实现: 计算损失，返回 (terms, extras)"""
        pass

    def should_use_scene(self) -> bool:
        """
        Whether scene conditioning is active at the current global step.

        scene_start_step is a global-step gate on the same level as the stage
        boundaries: before it the model learns basic goal-driven locomotion on
        blank scenes (occupancy zeroed in training, eval and visualization);
        after it the denoiser's cond_scene_mask_prob dropout applies, keeping
        some blank scenes to retain the basic-motion capability.

        Returns:
            True once step has reached scene_start_step, False otherwise
        """
        return self.step >= self.scene_start_step

    def should_use_augmentation(self) -> bool:
        """
        Whether planner-side history domain randomization is active at the
        current global step.

        Mirrors the train dataset's augmentation gate
        (augmentation_enabled, augmentation_start_step, split == 'train').
        The manager's step is equivalent to the dataset's training_step
        because set_training_step is called at the top of every training
        loop iteration. Reported to tensorboard for monitoring.

        Returns:
            True once the train split enables augmentation and step has
            reached augmentation_start_step, False otherwise
        """
        dataset = getattr(self, 'dataset', None)
        if dataset is None:
            return False
        return (
            bool(getattr(dataset, 'augmentation_enabled', False))
            and getattr(dataset, 'split', None) == 'train'
            and self.step >= int(getattr(dataset, 'augmentation_start_step', 0))
        )

    def should_rollout(self) -> bool:
        """
        Determine whether to use rollout history instead of ground truth.

        Returns:
            True if should use rollout history, False otherwise
        """
        if not self.use_rollout:
            return False
        if self.stage_idx < 1:
            return False
        prob = min(1.0, (self.step - self.stages[0]) / max(float(self.stages[1]), 1e-6))
        return torch.rand(1).item() < prob

    def should_static_pose(self) -> bool:
        """
        Determine whether to use static pose with perturbation.

        Returns:
            True if should use static pose, False otherwise
        """
        if not self.use_static_pose or self.stage_idx < 2:
            return False
        stage_start = self.stages[0] + self.stages[1]
        prob = min(
            1.0,
            (self.step - stage_start) / max(float(self.stages[2]), 1e-6),
        ) * self.static_prob
        return torch.rand(1).item() < prob

    def choose_history(
        self,
        gt_history: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        history_len: Optional[int] = None,
        return_rollout: bool = False,
    ):
        """
        统一的history选择函数
        Args:
            gt_history: 来自数据集的真实历史 [B, H, D]
            prev_motion: 前一个primitive的预测结果 [B, T, D]
            history_len: 历史长度，如果None则使用gt_history的长度
        Returns:
            选择的历史 [B, H, D]
        """
        if history_len is None:
            history_len = gt_history.shape[1]

        # 1. 检查是否使用rollout history
        used_rollout = prev_motion is not None and self.should_rollout()
        if used_rollout:
            history_motion = prev_motion[:, -history_len:, :]
        else:
            # 使用ground truth history
            history_motion = gt_history

        # 2. 检查是否使用static pose
        if self.should_static_pose():
            zero_feature = motion_dtype.get_zero_feature(
                self.dataset.dof_dim
            ).expand_as(history_motion).to(history_motion.device)
            # 添加扰动
            perturbation_scale = getattr(self, 'static_perturbation_scale', 0.0)
            if perturbation_scale > 0:
                # Not Test
                zero_feature = perturb_feature_v3(zero_feature, perturbation_scale)
            history_motion = self.dataset.normalize(zero_feature)

        if return_rollout:
            return history_motion, used_rollout
        return history_motion

    @abstractmethod
    def update_ema_models(self):
        """子类实现: 更新EMA模型"""
        pass


class MVAEManager(BaseManager, GeometryLoss):
    """
    Training manager for MVAE (Motion Variational AutoEncoder).

    Inherits from BaseManager for common training functionality and GeometryLoss
    for geometric loss computation capabilities.
    """

    vae: nn.Module
    optimizer: torch.optim.Optimizer

    static_prob: float

    def hold_model(self, vae, optimizer, dataset):
        self.vae = vae.to(self.device)
        self.optimizer = optimizer
        self.dataset = dataset
        logger.info("MVAEManager: Holding VAE model and optimizer")

        # 注册EMA模型
        self.register_ema_model('vae', self.vae)

        if self.ckpt.vae is not None:
            self.load_model(Path(self.ckpt.vae))

        # self.save_model()

    calc_loss = calc_mvae_loss

    def save_model(self) -> None:
        if not is_main_process():
            return

        save_path = self.save_dir / f"ckpt_{self.step}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Unwrap DDP model to save the underlying model's state_dict
        model_to_save = get_ddp_model(self.vae)

        save_dict = {
            'vae': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'feature_version': motion_dtype.FeatureVersion,
            'dof_dim': int(getattr(self.dataset, 'dof_dim', 29)),
            'nfeats': int(getattr(self.dataset, 'nfeats', 69)),
        }

        # 保存EMA模型
        if self.use_ema and self.ema_models:
            save_dict['ema_models'] = {name: model.state_dict() for name, model in self.ema_models.items()}

        torch.save(save_dict, save_path)
        logger.info(f"Saved model & optimizer to {save_path}")
        logger.info(f"Current step: {self.step}")

    def load_model(self, ckpt_path: Path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        ckpt_feature_version = state_dict.get('feature_version')
        if ckpt_feature_version is not None and int(ckpt_feature_version) != motion_dtype.FeatureVersion:
            raise ValueError(
                f"Checkpoint FeatureVersion {ckpt_feature_version} does not "
                f"match active FeatureVersion {motion_dtype.FeatureVersion}"
            )
        model_to_load = get_ddp_model(self.vae)
        model_to_load.load_state_dict(state_dict['vae'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step = state_dict['step']

        # 加载EMA模型
        if self.use_ema and 'ema_models' in state_dict:
            for name, ema_state in state_dict['ema_models'].items():
                if name in self.ema_models:
                    self.ema_models[name].load_state_dict(ema_state)
                    logger.info(f"Loaded EMA model: {name}")

        logger.info(f"Loaded CKPT model & optimizer from {ckpt_path}")
        logger.info(f"CKPT step: {self.step}")

    def update_ema_models(self):
        """更新EMA模型"""
        self.update_ema('vae', self.vae)


class DARManager(BaseManager, GeometryLoss):
    """
    Training manager for DAR (Diffusion AutoRegressive) model.

    Manages both VAE and denoiser models, with support for full DDPM sampling
    and advanced training strategies.
    """
    vae: nn.Module
    denoiser: nn.Module

    static_prob: float
    use_full_sample: bool

    def hold_model(self, vae, denoiser, optimizer, dataset):
        self.vae = vae.to(self.device)
        self.denoiser = denoiser.to(self.device)
        self.optimizer = optimizer
        self.dataset = dataset

        logger.info("DARManager: Holding denoiser models and optimizer. VAE loaded from checkpoint.")

        # 注册EMA模型
        self.register_ema_model('denoiser', self.denoiser)

        if self.ckpt.dar is not None:
            self.load_model(Path(self.ckpt.dar))
            logger.info(f"Loaded DAR model & optimizer from {self.ckpt.dar}")

            old_vae_path = Path(self.ckpt.dar).parent / "vae.pth"
            assert old_vae_path.exists(), f"VAE checkpoint not found at {old_vae_path}"
            self.ckpt.vae = str(old_vae_path)

        # Search Logic
        # 1. Search cache path
        # 2. Search self.ckpt.vae
        # 3. Search nearby 'train-mvae-*'
        cache_vae_path = self.save_dir / "vae.pth"
        if not cache_vae_path.exists():
            if self.ckpt.vae is None:
                maybe_vae_path = self.try_search_vae_path()
                if not maybe_vae_path:
                    raise ValueError("VAE checkpoint path must be provided in ckpt.vae")
                self.ckpt.vae = str(maybe_vae_path)
                logger.warning(f"VAE checkpoint path not provided, using the searched one: {self.ckpt.vae}")
            if self.save_dir.exists():
                try:
                    # Hard Link, not soft link. It should be more safe
                    os.link(self.ckpt.vae, cache_vae_path)
                    logger.info(f"VAE cached to {cache_vae_path}")
                    if is_main_process():
                        vae_src_path = self.save_dir / "vae_src.log"
                        with open(vae_src_path, "w") as f:
                            f.write(str(self.ckpt.vae))
                except OSError as e:
                    # FileExistsError (EEXIST): another DDP rank already cached it — OK to skip
                    # EXDEV/EPERM/EACCES: cross-filesystem link not allowed — fallback to copy
                    if e.errno in (errno.EEXIST, errno.EXDEV, errno.EPERM, errno.EACCES):
                        pass
                    else:
                        raise
            else:
                logger.warning(f"Save dir {self.save_dir} not exists, skip caching VAE")
                cache_vae_path = Path(self.ckpt.vae)

        self._load_vae_from_checkpoint(cache_vae_path)
        # self.save_model()

    def try_search_vae_path(self) -> Optional[Path]:
        exp_dir = self.save_dir.parent
        # 按修改时间排序（最新在前）
        mvae_dirs = sorted(exp_dir.glob("train-mvae-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(mvae_dirs) > 1:
            logger.warning("Multiple MVAE checkpoints found, using the latest one")

        for mvae_dir in mvae_dirs:
            ckpt_files = sorted(mvae_dir.glob("ckpt_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpt_files:
                return ckpt_files[0]
        return None

    def _load_vae_from_checkpoint(self, ckpt_path: Path):
        """Load VAE from checkpoint following DART's approach"""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        ckpt_feature_version = checkpoint.get('feature_version')
        if ckpt_feature_version is not None and int(ckpt_feature_version) != motion_dtype.FeatureVersion:
            raise ValueError(
                f"VAE checkpoint FeatureVersion {ckpt_feature_version} does not "
                f"match active FeatureVersion {motion_dtype.FeatureVersion}"
            )
        vae_state_dict = checkpoint['vae']

        if 'latent_mean' not in vae_state_dict:
            vae_state_dict['latent_mean'] = torch.tensor(0.0, device=self.device)
        if 'latent_std' not in vae_state_dict:
            vae_state_dict['latent_std'] = torch.tensor(1.0, device=self.device)

        self.vae.load_state_dict(vae_state_dict)

        self.vae.latent_mean = vae_state_dict['latent_mean']
        self.vae.latent_std = vae_state_dict['latent_std']

        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        logger.info(f"Loaded VAE from checkpoint: {ckpt_path}")
        logger.info(f"(latent_mean, latent_std): {self.vae.latent_mean}, {self.vae.latent_std}")

    calc_loss = calc_dar_loss

    def save_model(self) -> None:
        if not is_main_process():
            return

        save_path = self.save_dir / f"ckpt_{self.step}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Unwrap DDP model if needed
        model_to_save = get_ddp_model(self.denoiser)

        save_dict = {
            'denoiser': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'feature_version': motion_dtype.FeatureVersion,
            'dof_dim': int(getattr(self.dataset, 'dof_dim', 29)),
            'nfeats': int(getattr(self.dataset, 'nfeats', 69)),
        }

        # 保存EMA模型
        if self.use_ema and self.ema_models:
            save_dict['ema_models'] = {name: model.state_dict() for name, model in self.ema_models.items()}

        torch.save(save_dict, save_path)
        goal_stats_source = getattr(self.dataset, 'goal_stats_path', None)
        if goal_stats_source is not None:
            goal_stats_source = Path(goal_stats_source)
            if goal_stats_source.exists():
                goal_stats_target = self.save_dir / "goal_stats.pkl"
                shutil.copy2(goal_stats_source, goal_stats_target)
                logger.info(
                    "Copied goal stats from {} to {}",
                    goal_stats_source, goal_stats_target,
                )
        logger.info(f"Saved DAR model & optimizer to {save_path}")
        logger.info(f"Current step: {self.step}")

    @staticmethod
    def _fill_missing_arrival_embedder_state(model: nn.Module,
                                             checkpoint_state: Dict[str, Any],
                                             ckpt_path: Path):
        current_state = model.state_dict()
        missing_keys = [
            key for key in current_state
            if key.startswith('arrival_embedder.')
            and key not in checkpoint_state
        ]
        if not missing_keys:
            return checkpoint_state, False

        checkpoint_state = dict(checkpoint_state)
        for key in missing_keys:
            checkpoint_state[key] = current_state[key]
        logger.warning(
            "DAR checkpoint {} is missing {} arrival_embedder keys; "
            "initializing them from the current model and keeping all other "
            "state-dict checks strict",
            ckpt_path, len(missing_keys))
        return checkpoint_state, True

    def load_model(self, ckpt_path: Path):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        ckpt_feature_version = state_dict.get('feature_version')
        if ckpt_feature_version is not None and int(ckpt_feature_version) != motion_dtype.FeatureVersion:
            raise ValueError(
                f"DAR checkpoint FeatureVersion {ckpt_feature_version} does not "
                f"match active FeatureVersion {motion_dtype.FeatureVersion}"
            )
        denoiser_state, initialized_arrival = (
            self._fill_missing_arrival_embedder_state(
                self.denoiser, state_dict['denoiser'], ckpt_path))
        model_to_load = get_ddp_model(self.denoiser)
        model_to_load.load_state_dict(denoiser_state)
        if self.optimizer is not None:
            if initialized_arrival:
                logger.warning(
                    "Skipping optimizer state restore for {} because the "
                    "arrival_embedder parameters were initialized fresh",
                    ckpt_path)
            else:
                self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step = state_dict['step']

        # 加载EMA模型
        if self.use_ema and 'ema_models' in state_dict:
            for name, ema_state in state_dict['ema_models'].items():
                if name in self.ema_models:
                    ema_state, _ = self._fill_missing_arrival_embedder_state(
                        self.ema_models[name], ema_state, ckpt_path)
                    self.ema_models[name].load_state_dict(ema_state)
                    logger.info(f"Loaded EMA model: {name}")

        logger.info(f"Loaded DAR model & optimizer from {ckpt_path}")
        logger.info(f"CKPT step: {self.step}")

    def update_ema_models(self):
        """更新EMA模型"""
        self.update_ema('denoiser', self.denoiser)

    def should_use_full_sample(self) -> bool:
        """判断是否使用完整DDPM采样"""
        if not self.use_full_sample:
            return False
        return self.stage_idx > 0
