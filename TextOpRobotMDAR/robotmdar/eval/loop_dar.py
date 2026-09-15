"""
Loop DAR Script - Continuous Text-Conditioned Motion Generation

Continuously generates motion using DAR model in an autoregressive manner.
Starts from a zero pose and generates an infinite trajectory from the
text condition alone: the goal and scene conditions are force-dropped, and
the history is fed back from the generated blocks (the same regime as
planner_dar's generated_history with align_to_g1=null).

Usage:
- python eval/loop_dar.py --config-name=loop_dar
- Interactive commands:
  - Type any text line in the terminal: update the text condition for the
    next generated block (e.g. 'walking, turning')
  - Empty line: clear the text condition (unconditional motion prior)
  - Space or 'p': Pause/resume generation
  - Esc or 'q': Quit
"""

import atexit
import os
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from robotmdar.utils.planner_convert import (
    mujoco_to_isaaclab_dof,
)
from robotmdar.dtype import seed, logger as dtype_logger
from robotmdar.dtype.abc import Dataset, VAE, Denoiser, Diffusion, SSampler
import robotmdar.dtype.motion as motion_dtype
from robotmdar.dtype.motion import (
    get_zero_abs_pose,
    motion_dict_to_abs_pose,
    motion_dict_to_qpos,
)
from robotmdar.dtype.vis_mjc import mjc_load_everything
from robotmdar.eval.generate_dar import (
    denoiser_supports_text_guidance,
    generate_next_motion,
)
from robotmdar.model.clip import encode_text, load_and_freeze_clip
from robotmdar.train.manager import DARManager
from robotmdar.utils.dof_contract import configure_dof_contract

from robotmdar.dtype.debug import pdb_decorator

# ---------------------------------------------------------------------------
# NPZ saving: accumulates FK results from every generated block.
# Saved on graceful exit (Esc/q) or Ctrl+C via atexit.
# Set env var NPZ_OUTPUT to change output path (default: ./loop_motion.npz)
#
# NPZ structure (matching Tracker expectations):
#   joint_pos   [T, 29]   – joint angles (IsaacLab order, 29-DoF)
#   joint_vel   [T, 29]   – joint velocities
#   body_pos_w  [T, N, 3] – all N body world positions (FK result)
#   body_quat_w [T, N, 4] – all N body world orientations (wxyz, FK result)
#   fps         [1]       – frames per second (50)
#
# A companion file <output>.body_names.json lists body name → index for
# the 14 bodies that the Tracker specifically needs.
# ---------------------------------------------------------------------------
_NPZ_BUFFER: list = []          # each entry: (dof_pos, dof_vel, body_trans, body_rot)
_NPZ_OUTPUT = os.environ.get("NPZ_OUTPUT", "loop_motion.npz")
_NPZ_FPS = None
_NPZ_HISTORY_LEN = None
_NPZ_SKELETON_BODY_NAMES: list = []

# The 14 body names the TextOp Tracker expects (motion_loader.cpp body_names)
_NPZ_TRACKER_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

def _npz_save():
    """Called on exit. Concatenates all accumulated blocks and writes NPZ."""
    global _NPZ_BUFFER, _NPZ_SKELETON_BODY_NAMES
    if not _NPZ_BUFFER:
        return
    fps = _NPZ_FPS or 30

    all_dof_pos, all_dof_vel, all_body_trans, all_body_rot = [], [], [], []
    for dof_pos, dof_vel, body_trans, body_rot in _NPZ_BUFFER:
        all_dof_pos.append(dof_pos)
        all_dof_vel.append(dof_vel)
        all_body_trans.append(body_trans)
        all_body_rot.append(body_rot)

    dof_pos_all  = np.concatenate(all_dof_pos, axis=0)    # [T, 29], MuJoCo order
    dof_vel_all  = np.concatenate(all_dof_vel, axis=0)    # [T, 29], MuJoCo order
    body_trans_all = np.concatenate(all_body_trans, axis=0)  # [T, N, 3]
    body_rot_all = np.concatenate(all_body_rot, axis=0)      # [T, N, 4] xyzw

    dof_pos_isaaclab = mujoco_to_isaaclab_dof(dof_pos_all)
    dof_vel_isaaclab = mujoco_to_isaaclab_dof(dof_vel_all)

    # Convert body rotations: xyzw → wxyz (MuJoCo convention → IsaacLab convention)
    body_rot_all_wxyz = body_rot_all[..., [3, 0, 1, 2]]

    out = Path(_NPZ_OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        joint_pos=dof_pos_isaaclab,          # [T, 29]
        joint_vel=dof_vel_isaaclab,          # [T, 29]
        body_pos_w=body_trans_all,           # [T, N, 3] — ALL FK bodies
        body_quat_w=body_rot_all_wxyz,       # [T, N, 4] — ALL FK bodies (wxyz)
        fps=np.array([fps]),
    )

    # Write body name → index mapping for Tracker's 14-body subset
    name_to_idx = {name: i for i, name in enumerate(_NPZ_SKELETON_BODY_NAMES)}
    tracker_map = {}
    for name in _NPZ_TRACKER_BODIES:
        idx = name_to_idx.get(name, -1)
        tracker_map[name] = idx
    missing = [k for k, v in tracker_map.items() if v < 0]
    if missing:
        print(f"\n[NPZ] WARNING: bodies not found in skeleton: {missing}")

    body_map_path = out.with_suffix(out.suffix + ".body_names.json")
    import json as _json
    body_map_path.write_text(_json.dumps({
        "all_body_names": _NPZ_SKELETON_BODY_NAMES,
        "tracker_body_indices": tracker_map,
        "note": "Tracker expects body_pos_w indexed as listed; use tracker_body_indices to subset",
    }, indent=2))

    file_mb = out.stat().st_size / 1024 / 1024
    T_final = dof_pos_isaaclab.shape[0]
    N_bodies = body_trans_all.shape[1]
    print(f"\n[NPZ] Saved {T_final} frames × {N_bodies} bodies ({T_final/fps:.1f}s) → {out} ({file_mb:.2f} MB)")
    print(f"[NPZ] Body index map → {body_map_path}")


atexit.register(_npz_save)
# ---------------------------------------------------------------------------

# import torch_tensorrt


class LoopState:
    """State management for continuous motion generation."""

    def __init__(self):
        self.paused = False
        # Current text condition ("" = unconditional motion prior)
        self.text = ""
        self.quit_requested = False


def interactive_input_thread(loop_state: LoopState):
    """Interactive input thread for the text condition.

    Every non-empty line becomes the text condition for the next generated
    block. An empty line clears the condition (unconditional prior).
    """
    print("Enter text prompt for the motion prior (empty line = clear)")
    while not loop_state.quit_requested:
        try:
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            break
        text = user_input.strip()
        if text:
            loop_state.text = text
            print(f"Text updated: {text!r} (applies at the next block)")
        else:
            loop_state.text = ""
            print("Text cleared (unconditional prior)")


@pdb_decorator
def main(cfg: DictConfig):
    configure_dof_contract(cfg)
    dtype_logger.set(cfg)
    seed.set(cfg.seed)

    # Load models
    val_data: Dataset = instantiate(cfg.data.val)
    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)

    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    vae.eval()
    denoiser.eval()

    # Load checkpoints
    manager: DARManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)

    # vae_trt = torch.compile(vae, backend='tensorrt')
    # denoiser_trt = torch.compile(denoiser, backend='tensorrt')
    vae_trt = vae
    denoiser_trt = denoiser
    cfg_denoiser = denoiser_trt

    # Text condition support comes from the checkpoint architecture.
    text_condition_supported = denoiser_supports_text_guidance(denoiser)
    clip_model = None
    text_embedding_cache: dict = {}
    if text_condition_supported:
        clip_model = load_and_freeze_clip(
            str(cfg.data.get("clip_version", "ViT-B/32")),
            device=str(cfg.device),
            clip_model_path=cfg.data.get("clip_model_path"),
        )
        with torch.no_grad():
            _warmup = encode_text(clip_model, ["text condition warmup"])
        del _warmup
        if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
            torch.cuda.synchronize()
        logger.info("Text conditioning enabled; type text lines to change it")
    else:
        logger.warning(
            "Denoiser has no text-condition weights; text input will be "
            "ignored (unconditional generation)")

    future_len = cfg.data.future_len
    history_len = cfg.data.history_len
    replanning_period_frames = int(cfg.replanning_period_frames)
    if not history_len <= replanning_period_frames <= future_len:
        raise ValueError(
            "replanning_period_frames must satisfy "
            f"history_len ({history_len}) <= replanning_period_frames "
            f"({replanning_period_frames}) <= future_len ({future_len})"
        )

    # Store for NPZ saving
    global _NPZ_FPS, _NPZ_HISTORY_LEN, _NPZ_SKELETON_BODY_NAMES
    _NPZ_FPS = val_data.fps
    _NPZ_HISTORY_LEN = history_len
    _NPZ_SKELETON_BODY_NAMES = list(val_data.skeleton.body_names)

    # Initialize state
    loop_state = LoopState()

    # Initialize motion generation state
    if motion_dtype.FeatureVersion == 4:
        init_motion = motion_dtype.get_zero_feature(val_data.skeleton)
        history_motion = val_data.normalize(
            init_motion.unsqueeze(0).expand(1, history_len, -1).to(cfg.device))
    else:
        history_motion = val_data.normalize(
            motion_dtype.get_zero_feature().unsqueeze(0).expand(1, history_len,
                                                                -1).to(cfg.device))
    abs_pose = get_zero_abs_pose((1, ), device=cfg.device)

    # Goal/scene conditions are force-dropped: this loop is pure text prior.
    # The tensors keep the checkpoint's dimensions (66-D no-log
    # split_end_effector goal, 25^3 scene voxels) but are fully masked at
    # inference, matching the text-only regime of text_prior training.
    goal = torch.zeros(1, int(cfg.denoiser.goal_dim), device=cfg.device)
    voxel = torch.zeros(1, cfg.denoiser.grid_size**3, device=cfg.device)
    # A present goal tensor requires an arrival-time frame for the split
    # denoiser; it is force-dropped together with the goal itself.
    time_to_arrival_frame = torch.zeros(
        1, 1, dtype=torch.long, device=cfg.device)

    # Reuse one diffusion noise realization across all blocks, mirroring the
    # planner's resample_noise_each_plan: false. Independent noise at every
    # block boundary makes nearby histories decode to visibly different
    # motions and would pollute the seam-smoothness analysis.
    fixed_sampling_noise = torch.randn(
        (1, *denoiser.noise_shape), device=cfg.device)

    # Setup visualization with keyboard callback
    dt = 1.0 / val_data.fps

    def keycb_fn(key):
        """Handle keyboard input for interactive control."""
        # Space (32) or 'p' key: pause/resume
        if key == ord(' ') or key == ord('P') or key == ord('p'):
            loop_state.paused = not loop_state.paused
            status = "paused" if loop_state.paused else "resumed"
            logger.info(f"Generation {status}")
        # Esc (256 is GLFW ESC, 27 is ASCII ESC) or 'q' key: quit
        elif key == 256 or key == 27 or key == ord('Q') or key == ord('q'):
            logger.info("Quit requested")
            loop_state.quit_requested = True

    # Keep visualization on the exact MJCF selected by the active skeleton
    # config.  The helper's historical default is a cwd-relative path and can
    # otherwise silently diverge from the model/FK asset when launched from a
    # different directory.
    show_fn, viewer = mjc_load_everything(
        dt,
        keycb_fn,
        humanoid_xml=str(val_data.skeleton.fk.mjcf_file),
    )

    # Start interactive input thread
    input_thread = threading.Thread(target=interactive_input_thread,
                                    args=(loop_state, ))
    input_thread.daemon = True
    input_thread.start()

    logger.info("Starting continuous motion generation...")
    logger.info(
        "Replanning after {} executed frames ({:.3f} s); model horizon={} "
        "frames; history={} frames",
        replanning_period_frames,
        replanning_period_frames * dt,
        future_len,
        history_len,
    )
    logger.info(
        "Commands: type a text line to change the motion prompt, "
        "empty line to clear it, Space/p(pause), Esc/q(quit)")

    # Main generation loop
    current_text = ""
    text_embedding = None
    frame_idx = 0
    while not loop_state.quit_requested and viewer.is_running():
        # Update the text condition when the prompt changed.
        if text_condition_supported:
            text_prompt = loop_state.text.strip()
            if text_prompt != current_text:
                current_text = text_prompt
                if current_text:
                    text_embedding = text_embedding_cache.get(current_text)
                    if text_embedding is None:
                        with torch.no_grad():
                            text_embedding = encode_text(
                                clip_model, [current_text]).to(cfg.device)
                        text_embedding_cache[current_text] = text_embedding
                        logger.info("Text condition: {!r}", current_text)
                else:
                    text_embedding = None
                    logger.info("Text condition cleared (unconditional prior)")

        # Generate next motion if not paused
        if not loop_state.paused:
            # ``abs_pose`` is the pose at the start of the new future window.
            # Keep it separate: the generic generator also reconstructs a
            # history+future tensor, which is useful for legacy consumers but
            # is incorrect for V6 transition features when history is already
            # anchored at this pose (it would integrate the 16 history frames
            # a second time).
            future_start_abs_pose = abs_pose
            # breakpoint()
            future_motion, motion_dict, abs_pose = generate_next_motion(
                vae=vae_trt,
                denoiser=cfg_denoiser,
                diffusion=diffusion,
                val_data=val_data,
                goal=goal,
                voxel=voxel,
                history_motion=history_motion,
                abs_pose=abs_pose,
                future_len=future_len,
                use_full_sample=cfg.use_full_sample,
                guidance_scale=cfg.guidance_scale,
                initial_noise=fixed_sampling_noise,
                text_embedding=text_embedding,
                text_valid=True if text_embedding is not None else None,
                force_drop_goal_root=True,
                force_drop_goal_yaw=True,
                force_drop_goal_time=True,
                force_drop_goal_orientation=True,
                force_drop_goal_joint=True,
                force_drop_goal_velocity=True,
                force_drop_goal_end_effector=True,
                force_drop_scene=True,
                time_to_arrival_frame=time_to_arrival_frame,
                ret_fk=True)

            # Reconstruct only the newly generated future from the current
            # absolute pose.  This is the sequence that is played, saved, and
            # used to anchor the next autoregressive block.
            motion_dict = val_data.reconstruct_motion(
                future_motion,
                abs_pose=future_start_abs_pose,
                ret_fk=True,
            )
            # Advance only through the execution horizon. Predicted states
            # after this frame are deliberately discarded.
            abs_pose = motion_dict_to_abs_pose(
                motion_dict, idx=replanning_period_frames - 1)

            # ── NPZ: accumulate the newly reconstructed future ──
            # Each entry contains only executed frames, so saved seams land at
            # replanning_period_frames boundaries rather than future_len.
            global _NPZ_BUFFER
            executed = slice(0, replanning_period_frames)
            dof_pos = motion_dict['dof_pos'][0, executed].detach().cpu().numpy()             # [P, 29]
            dof_vel = motion_dict['dof_vel'][0, executed].detach().cpu().numpy()             # [P, 29]
            body_t  = motion_dict['global_translation'][0, executed].detach().cpu().numpy()  # [P, N, 3]
            body_r  = motion_dict['global_rotation'][0, executed].detach().cpu().numpy()     # [P, N, 4] xyzw
            _NPZ_BUFFER.append((dof_pos, dof_vel, body_t, body_r))
            # ────────────────────────────────────────────────────────────────────

            # Update history for next generation (autoregressive)
            history_motion = future_motion[
                :,
                replanning_period_frames - history_len:
                replanning_period_frames,
                :,
            ]

            # Visualize the motion
            qpos_data, contact_data = motion_dict_to_qpos(motion_dict)

            # Convert to numpy - qpos_data and contact_data are torch tensors
            qpos_np = qpos_data.detach().cpu().numpy()  # [B, T, dof]
            contact_np = contact_data.detach().cpu().numpy()  # [B, T, 2]

            # Display only the execution horizon. The unused suffix of the
            # 64-frame prediction is discarded before the next inference.
            for t in range(replanning_period_frames):
                if loop_state.quit_requested or not viewer.is_running():
                    break
                show_fn(qpos_np[0, t], contact_np[0, t])
                time.sleep(dt)
                frame_idx += 1
                # print("Frame ID: ", frame_idx)
        else:
            time.sleep(0.1)  # Small sleep when paused

    logger.info("Shutting down...")
    viewer.close()
