"""
DAR Motion Generation Module

Common motion generation functionality extracted from vis_dar.py and loop_dar.py.
Supports both full_sample (complete DDPM sampling) and single_step_sample modes.

Functions:
- generate_next_motion: Generate next motion segment using DAR model
"""

import torch
from torch import nn
from typing import Tuple, Dict, Any, Optional, Union
from robotmdar.dtype.motion import motion_dict_to_abs_pose


class ClassifierFreeWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y: Dict[str, Any]):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        # print('scale:', y['scale'])
        # diff_cond = (out - out_uncond).norm()
        # print('diff_cond:', diff_cond, 'out', out.norm(), 'out_uncond',
        #       out_uncond.norm())
        return out_uncond + (y['scale'] * (out - out_uncond))

    @property
    def noise_shape(self):
        return self.model.noise_shape


def generate_next_motion(
        vae,
        denoiser,
        diffusion,
        val_data,
        goal: torch.Tensor,
        voxel: torch.Tensor,
        history_motion: torch.Tensor,
        abs_pose,  #  AbsolutePose
        future_len: int,
        use_full_sample: bool = False,
        guidance_scale: Optional[float] = None,
        initial_noise: Optional[torch.Tensor] = None,
        ret_fk: bool = False,
        ret_fk_full: bool = False,
        use_vae=True,
        use_ddim=False,
        force_drop_goal_root: bool = False,
        force_drop_goal_yaw: bool = False,
        force_drop_goal_time: bool = False,
        force_drop_goal_body: bool = False):
    """
    Generate next motion segment using DAR model.

    Args:
        vae: VAE model for encoding/decoding
        denoiser: Denoiser model for diffusion
        diffusion: Diffusion model
        val_data: Dataset for motion reconstruction
        goal: Ego-centric goal tensor [B, denoiser.goal_dim].
        voxel: Scene occupancy tensor [B, grid_size^3]
        history_motion: History motion tensor [B, T_hist, D]
        abs_pose: Current absolute pose
        future_len: Length of future motion to generate
        use_full_sample: Whether to use full DDPM sampling loop
        force_drop_goal_root: Mask the root-position component at inference.
        force_drop_goal_yaw: Mask the root-yaw component at inference.
        force_drop_goal_time: Mask the remaining-time component at inference.
        force_drop_goal_body: Mask the limb-keypoint component at inference.

    Returns:
        Tuple of (future_motion_pred, motion_dict, new_abs_pose)
    """
    device = history_motion.device
    with torch.no_grad():
        batch_size = goal.shape[0]
        # latent_shape = (batch_size, 1, 128
        #                 )  # [B, T=1, D] - latent_dim from config
        latent_shape = (batch_size, *denoiser.noise_shape)

        # Online replanning should reuse one noise realization. Resampling at
        # every MPC update makes otherwise-nearby conditions decode to visibly
        # different motions at the plan seam.
        if initial_noise is None:
            x_start_noise = torch.randn(latent_shape, device=device)
        else:
            if tuple(initial_noise.shape) != tuple(latent_shape):
                raise ValueError(
                    f"initial_noise has shape {tuple(initial_noise.shape)}, "
                    f"expected {tuple(latent_shape)}")
            x_start_noise = initial_noise.to(device=device)

        # Sample a random timestep for demonstration (or use t=0 for no noise)
        t = torch.zeros(batch_size, dtype=torch.int32,
                        device=device) + diffusion.num_timesteps - 1

        # Prepare conditioning
        y: Dict[str, Any] = {
            'goal': goal,
            'voxel': voxel,
            'history_motion_normalized': history_motion,
            'force_drop_goal_root': force_drop_goal_root,
            'force_drop_goal_yaw': force_drop_goal_yaw,
            'force_drop_goal_time': force_drop_goal_time,
            'force_drop_goal_body': force_drop_goal_body,
        }
        if guidance_scale is not None:
            y['scale'] = guidance_scale

        # print(diffusion.num_timesteps)
        if use_full_sample:
            if not use_ddim:
                # Use complete DDPM sampling loop
                sample_fn = diffusion.p_sample_loop
                x_start_pred = sample_fn(
                    denoiser,
                    latent_shape,
                    clip_denoised=False,
                    model_kwargs={'y': y},  # Wrap y in the expected structure
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
            else:
                # zjk: use DDIM sampling loop
                sample_fn = diffusion.ddim_sample_loop
                x_start_pred = sample_fn(
                    denoiser,
                    latent_shape,
                    clip_denoised=False,
                    model_kwargs={'y': y},  # Wrap y in the expected structure
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    eta=0.0,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )


            assert isinstance(x_start_pred, torch.Tensor), \
                f"Expected tensor, got {type(x_start_pred)}"
        else:
            # Single step denoising (default mode)
            x_start_pred = denoiser(x_t=x_start_noise,
                                    timesteps=diffusion._scale_timesteps(t),
                                    y=y)  # [B, T=1, D]

        if use_vae:
            # Convert to VAE format [T=1, B, D]
            latent_pred = x_start_pred.permute(1, 0, 2)

            # breakpoint()

            # Decode using VAE
            # latent_pred: (1, 1, 128)  history_motion: (1, 2, 57)

            future_motion_pred = vae.decode(latent_pred,
                                            history_motion,
                                            nfuture=future_len)
        else:
            future_motion_pred = x_start_pred[:, -future_len:]

        # Reconstruct motion dictionary
        motion_dict = val_data.reconstruct_motion(torch.cat(
            [history_motion, future_motion_pred], dim=1),
                                                  abs_pose=abs_pose,
                                                  ret_fk=ret_fk,
                                                  ret_fk_full=ret_fk_full)

        # Update absolute pose for next primitive
        new_abs_pose = motion_dict_to_abs_pose(motion_dict, idx=-2)

        return future_motion_pred, motion_dict, new_abs_pose
