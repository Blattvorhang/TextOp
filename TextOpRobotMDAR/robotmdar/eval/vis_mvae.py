import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Literal, Callable, List, Tuple
from dataclasses import dataclass
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from robotmdar.dtype import seed, logger
from robotmdar.dtype.motion import motion_dict_to_qpos, QPos, get_zero_abs_pose, motion_dict_to_abs_pose, FeatureVersion, get_blended_feature, transform_feature_to_world, dict_concat, dict_to_tensor
from robotmdar.dtype.device import tree_to_numpy
from robotmdar.dtype.abc import Dataset, VAE, Optimizer, Distribution
from robotmdar.train.manager import MVAEManager
from robotmdar.dtype.vis_mjc import mjc_load_everything, VisState, get_keycb_fn, mjc_autoloop_mdar
from robotmdar.utils.dof_contract import configure_dof_contract


def add_batch_fn(motion_buff, val_dataiter, vae, val_data, num_primitive,
                 future_len, history_len, cfg):

    def add_batch():
        pd_abs_pose = get_zero_abs_pose((cfg.data.batch_size, ),
                                        device=cfg.device)
        gt_abs_pose = get_zero_abs_pose((cfg.data.batch_size, ),
                                        device=cfg.device)
        batch = next(val_dataiter)
        # breakpoint()
        # motion_buff['pd'].append([])
        # motion_buff['gt'].append([])
        pd_buff: List[Tuple[QPos, torch.Tensor]] = []
        gt_buff: List[Tuple[QPos, torch.Tensor]] = []
        for pidx in range(num_primitive):
            motion = batch[pidx]['motion'].to(cfg.device)

            future_motion_gt = motion[:, -future_len:, :]
            history_motion = motion[:, :history_len, :]

            # if FeatureVersion == 4:
            #     history_motion_dict = val_data.reconstruct_motion(history_motion, ret_fk=False)
            #     transf_rotmat, transf_transl, blended_feature_dict = get_blended_feature(history_motion_dict, val_data.skeleton)
            #     history_motion = val_data.normalize(dict_to_tensor(blended_feature_dict))

            latent, dist = vae.encode(future_motion=future_motion_gt,
                                      history_motion=history_motion)
            future_motion_pred = vae.decode(latent,
                                            history_motion,
                                            nfuture=future_len)
            future_motion_pred_dict = val_data.reconstruct_motion(
                torch.cat([history_motion, future_motion_pred], dim=1),
                abs_pose=pd_abs_pose,
                ret_fk=False)
            future_motion_gt_dict = val_data.reconstruct_motion(
                torch.cat([history_motion, future_motion_gt], dim=1),
                abs_pose=gt_abs_pose,
                ret_fk=False)
            
            # if FeatureVersion == 4:
            # if False:
            #     print("Transforming to world coordinates using blended features...")
            #     history_motion_dict = val_data.reconstruct_motion(history_motion, ret_fk=False)
            #     transf_rotmat, transf_transl, _ = get_blended_feature(history_motion_dict, val_data.skeleton)
            #     future_motion_pred_dict.update({
            #         'transf_rotmat': transf_rotmat,
            #         'transf_transl': transf_transl
            #     })
            #     future_motion_gt_dict.update({
            #         'transf_rotmat': transf_rotmat,
            #         'transf_transl': transf_transl
            #     })
            #     future_motion_pred_dict = transform_feature_to_world(future_motion_pred_dict)
            #     future_motion_gt_dict = transform_feature_to_world(future_motion_gt_dict)

            #     # future_motion_pred_dict = dict_concat(history_motion_dict, future_motion_pred_dict)
            #     # future_motion_gt_dict = dict_concat(history_motion_dict, future_motion_gt_dict)

            pd_abs_pose = motion_dict_to_abs_pose(future_motion_pred_dict,
                                                  idx=-2)
            gt_abs_pose = motion_dict_to_abs_pose(future_motion_gt_dict,
                                                  idx=-2)

            pd_buff.append(
                tree_to_numpy(motion_dict_to_qpos(
                    future_motion_pred_dict)))  # type: ignore
            gt_buff.append(
                tree_to_numpy(motion_dict_to_qpos(
                    future_motion_gt_dict)))  # type: ignore
            # breakpoint()

        motion_buff['pd'].append(pd_buff)
        motion_buff['gt'].append(gt_buff)

    return add_batch


def main(cfg: DictConfig):
    configure_dof_contract(cfg)
    logger.set(cfg)
    seed.set(cfg.seed)

    val_data: Dataset = instantiate(cfg.data.val)
    val_dataiter = iter(val_data)

    vae: VAE = instantiate(cfg.vae)
    vae.eval()
    manager: MVAEManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, None, val_data)

    num_primitive = cfg.data.num_primitive
    future_len = cfg.data.future_len
    history_len = cfg.data.history_len

    motion_buff = {
        'pd': [],
        'gt': [],
    }
    vs = VisState()

    add_batch = add_batch_fn(motion_buff, val_dataiter, vae, val_data,
                             num_primitive, future_len, history_len, cfg)

    keycb_fn = get_keycb_fn(vs)

    fps = val_data.fps

    mjc_autoloop_mdar(vs, fps, num_primitive, future_len, history_len,
                      motion_buff, add_batch, keycb_fn)
