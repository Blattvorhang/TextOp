from typing import List, Tuple
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from robotmdar.utils.ego_condition import build_ego_goal, query_local_occupancy
from robotmdar.dtype import seed, logger as dtypelogger
from robotmdar.dtype.motion import motion_dict_to_qpos, QPos, get_zero_abs_pose, motion_dict_to_abs_pose
from robotmdar.dtype.device import tree_to_numpy
from robotmdar.dtype.abc import Dataset, VAE, Denoiser, Diffusion, SSampler
from robotmdar.train.manager import DARManager
from robotmdar.dtype.vis_mjc import VisState, get_keycb_fn, mjc_autoloop_mdar
from robotmdar.dtype.motion import get_zero_feature_v3
from robotmdar.eval.generate_dar import generate_next_motion


def add_batch_fn(motion_buff, val_dataiter, vae, denoiser, diffusion, val_data,
                 num_primitive, future_len, history_len, cfg):

    def add_batch():
        pd_abs_pose = get_zero_abs_pose((cfg.data.batch_size, ),
                                        device=cfg.device)
        gt_abs_pose = get_zero_abs_pose((cfg.data.batch_size, ),
                                        device=cfg.device)
        batch = next(val_dataiter)
        pd_buff: List[Tuple[QPos, torch.Tensor]] = []
        gt_buff: List[Tuple[QPos, torch.Tensor]] = []

        # Configuration for generation modes
        use_autoregressive = getattr(cfg, 'use_autoregressive', False)
        use_full_sample = getattr(cfg, 'use_full_sample', False)

        # Track predicted motion for autoregressive generation
        prev_predicted_motion = None

        for pidx in range(num_primitive):
            batch_item = batch[pidx]
            motion = batch_item['motion'].to(cfg.device)

            future_motion_gt = motion[:, -future_len:, :]
            history_motion_gt = motion[:, :history_len, :]

            if use_autoregressive:
                if prev_predicted_motion is not None:
                    # Use suffix of previous prediction as history
                    history_motion = prev_predicted_motion[:, -history_len:, :]
                else:
                    # First primitive: use zero history for true autoregressive mode
                    # history_motion = get_zero_feature_v3().expand_as(
                    #     history_motion_gt).to(history_motion_gt.device)
                    history_motion = history_motion_gt
            else:
                # Teacher forcing mode: use ground truth history
                history_motion = history_motion_gt

            # Build ego_goal and voxel from batch item
            # For vis_dar (teacher forcing), we use gt_ref_pos/gt_ref_rot as the
            # reference pose matching training logic
            ego_goal = build_ego_goal(
                batch_item['world_goal_pos'].to(cfg.device),
                batch_item['world_goal_yaw'].to(cfg.device),
                batch_item['gt_ref_pos'].to(cfg.device),
                batch_item['gt_ref_rot'].to(cfg.device),
            )  # [B, 5]
            voxel = query_local_occupancy(
                batch_item['scene'],
                batch_item['gt_ref_pos'],
                batch_item['gt_ref_rot'],
                grid_size=cfg.denoiser.grid_size,
                grid_unit=cfg.data.occupancy_unit,
            )  # [B, grid_size^3]

            # Generate prediction using the common generate_next_motion function
            future_motion_pred, future_motion_pred_dict, pd_abs_pose = generate_next_motion(
                vae=vae,
                denoiser=denoiser,
                diffusion=diffusion,
                val_data=val_data,
                goal=ego_goal,
                voxel=voxel,
                history_motion=history_motion,
                abs_pose=pd_abs_pose,
                future_len=future_len,
                use_full_sample=use_full_sample,
                guidance_scale=cfg.guidance_scale)

            # Store the full predicted motion (history + future) for next iteration
            if use_autoregressive:
                prev_predicted_motion = future_motion_pred

            # Reconstruct ground truth motion dictionary
            future_motion_gt_dict = val_data.reconstruct_motion(
                torch.cat([history_motion_gt, future_motion_gt], dim=1),
                abs_pose=gt_abs_pose,
                ret_fk=False)

            # Update ground truth absolute pose for next primitive
            gt_abs_pose = motion_dict_to_abs_pose(future_motion_gt_dict,
                                                  idx=-2)

            pd_buff.append(
                tree_to_numpy(motion_dict_to_qpos(
                    future_motion_pred_dict)))  # type: ignore
            gt_buff.append(
                tree_to_numpy(motion_dict_to_qpos(
                    future_motion_gt_dict)))  # type: ignore

        motion_buff['pd'].append(pd_buff)
        motion_buff['gt'].append(gt_buff)

    return add_batch


def main(cfg: DictConfig):
    dtypelogger.set(cfg)
    seed.set(cfg.seed)

    val_data: Dataset = instantiate(cfg.data.val)
    val_dataiter = iter(val_data)

    # Load models
    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)

    # Load diffusion
    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    # Set models to eval mode
    vae.eval()
    denoiser.eval()

    # Load checkpoints using manager
    manager: DARManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)
    cfg_denoiser = denoiser

    num_primitive = cfg.data.num_primitive
    future_len = cfg.data.future_len
    history_len = cfg.data.history_len

    motion_buff = {
        'pd': [],
        'gt': [],
    }
    vs = VisState()

    add_batch = add_batch_fn(motion_buff, val_dataiter, vae, cfg_denoiser,
                             diffusion, val_data, num_primitive, future_len,
                             history_len, cfg)

    keycb_fn = get_keycb_fn(vs)

    fps = val_data.fps

    mjc_autoloop_mdar(vs, fps, num_primitive, future_len, history_len,
                      motion_buff, add_batch, keycb_fn)
