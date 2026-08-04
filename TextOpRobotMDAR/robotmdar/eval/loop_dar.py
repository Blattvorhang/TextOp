"""
Loop DAR Script - Continuous Motion Generation

Continuously generates motion using DAR model in an autoregressive manner.
Starts from zero pose and generates infinite trajectory based on goal+scene conditioning.

Usage:
- python eval/loop_dar.py --config-name=loop_dar
- Interactive commands:
  - Enter 'x y z yaw(deg)' in terminal: Set world-space goal (yaw degrees, 0°=+X)
  - Space or 'p': Pause/resume generation
  - Esc or 'q': Quit
- Goal visualization: green arrow for heading
"""

import atexit
import math
import os
import threading
import time
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig

from robotmdar.utils.goal import (
    GoalType,
    build_ego_goal,
    validate_goal_config,
)
from robotmdar.utils.planner_convert import load_goal_keypoints_from_reference
from robotmdar.dtype import seed, logger as dtype_logger
from robotmdar.dtype.abc import Dataset, VAE, Denoiser, Diffusion, SSampler
from robotmdar.dtype.motion import (G1_ROOT_HEIGHT, motion_dict_to_qpos,
                                    get_zero_abs_pose, motion_dict_to_abs_pose,
                                    get_zero_feature, FeatureVersion)
from robotmdar.dtype.vis_mjc import mjc_load_everything
from robotmdar.eval.generate_dar import generate_next_motion
from robotmdar.train.manager import DARManager

from robotmdar.wrapper.vae_decode import DecoderWrapper
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
_NPZ_FIRST_BLOCK = True
_NPZ_OUTPUT = os.environ.get("NPZ_OUTPUT", "loop_motion.npz")
_NPZ_FPS = None
_NPZ_HISTORY_LEN = None
_NPZ_SKELETON_BODY_NAMES: list = []

# MuJoCo → IsaacLab joint reindex
_NPZ_MJC2ISAAC = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
    18, 25, 19, 26, 20, 27, 21, 28
]

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


def _npz_expand_23_to_29(v: np.ndarray) -> np.ndarray:
    """Pad 23-DoF (wrists locked) → 29-DoF for IsaacLab."""
    T = v.shape[0]
    out = np.zeros((T, 29), dtype=v.dtype)
    out[:, :19] = v[:, :19]
    out[:, 22:26] = v[:, 19:23]
    return out


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

    dof_pos_all  = np.concatenate(all_dof_pos, axis=0)    # [T, 23]
    dof_vel_all  = np.concatenate(all_dof_vel, axis=0)    # [T, 23]
    body_trans_all = np.concatenate(all_body_trans, axis=0)  # [T, N, 3]
    body_rot_all = np.concatenate(all_body_rot, axis=0)      # [T, N, 4] xyzw

    dof_pos_29 = _npz_expand_23_to_29(dof_pos_all)
    dof_vel_29 = _npz_expand_23_to_29(dof_vel_all)
    dof_pos_isaaclab = dof_pos_29[:, _NPZ_MJC2ISAAC]
    dof_vel_isaaclab = dof_vel_29[:, _NPZ_MJC2ISAAC]

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
        # World-space goal: (x, y, z, yaw_deg), yaw=0° → +X direction
        # Default Z = G1_ROOT_HEIGHT (0.77m), the canonical standing root height
        self.world_goal = [0.0, 0.0, G1_ROOT_HEIGHT, 0.0]
        self.goal_received = False  # True after first valid user input
        self.quit_requested = False


def interactive_input_thread(loop_state: LoopState):
    """Interactive input thread for goal input.

    Accepts world-space goal as: x y z yaw(deg)
    - x, y, z: target position in world frame (meters)
    - yaw: target heading in DEGREES, 0° = +X direction, 90° = +Y
    Example: "1.0 0.5 0.0 90" (target 1m forward, 0.5m left, facing +Y)
    """
    print("Enter world goal: x y z yaw(deg)")
    print("  yaw(deg): 0°=+X, 90°=+Y, 180°=-X, -90°=-Y")
    while not loop_state.quit_requested:
        try:
            user_input = input()
            parts = user_input.strip().split()
            if len(parts) == 4:
                x, y, z, yaw_deg = map(float, parts)
                yaw_rad = math.radians(yaw_deg)
                loop_state.world_goal = [x, y, z, yaw_rad]
                loop_state.goal_received = True
                print(f"Goal updated: x={x:.3f} y={y:.3f} z={z:.3f} "
                      f"yaw={yaw_deg:.1f}° ({yaw_rad:.3f} rad)")
            else:
                print(f"Invalid: expected 4 values (x y z yaw_deg), got {len(parts)}")
        except (EOFError, KeyboardInterrupt):
            break
        except ValueError as e:
            print(f"Parse error: {e}. Format: x y z yaw_deg "
                  f"(e.g. '1.0 0.0 0.0 90')")


def _update_goal_vis(viewer, world_goal: list, goal_received: bool):
    """Draw goal heading as an arrow + base sphere via ``mjv_initGeom``.

    Uses ``mjGEOM_ARROW`` (cone+cylinder along local +Z) for the heading
    arrow and ``mjGEOM_SPHERE`` for the base position marker.  Avoids
    ``mjv_connector`` which segfaults in some MuJoCo builds.
    """
    viewer.user_scn.ngeom = 0
    if not goal_received:
        return

    x, y, z, yaw = world_goal
    cos_h = math.cos(yaw)
    sin_h = math.sin(yaw)

    pos = np.array([x, y, z], dtype=np.float64)

    arrow_radius = 0.025
    arrow_length = 0.5
    sphere_radius = 0.05

    rgba = np.array([0.2, 1.0, 0.2, 0.9], dtype=np.float32)

    # arrow
    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]

        mat = np.array([
            -sin_h, 0.0, cos_h,
            cos_h, 0.0, sin_h,
            0.0,   1.0, 0.0,
        ], dtype=np.float64)

        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.array(
                [arrow_radius, arrow_radius, arrow_length],
                dtype=np.float32,
            ),
            pos,
            mat,
            rgba,
        )

        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        viewer.user_scn.ngeom += 1

    # base sphere
    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([sphere_radius, sphere_radius, sphere_radius], dtype=np.float32),
            pos,
            np.eye(3).reshape(-1),
            rgba,
        )
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        viewer.user_scn.ngeom += 1


@pdb_decorator
def main(cfg: DictConfig):
    dtype_logger.set(cfg)
    seed.set(cfg.seed)
    goal_type = validate_goal_config(
        cfg.data.goal_type, cfg.denoiser.goal_dim)
    goal_reference_path = cfg.get("goal_reference_path")
    if goal_reference_path is not None:
        goal_reference_path = to_absolute_path(str(goal_reference_path))
    if goal_type is GoalType.BODY and goal_reference_path is None:
        raise ValueError("Body-goal loop requires goal_reference_path")
    # torch.set_default_device(cfg.device)

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

    future_len = cfg.data.future_len
    history_len = cfg.data.history_len

    # Store for NPZ saving
    global _NPZ_FPS, _NPZ_HISTORY_LEN, _NPZ_SKELETON_BODY_NAMES
    _NPZ_FPS = val_data.fps
    _NPZ_HISTORY_LEN = history_len
    _NPZ_SKELETON_BODY_NAMES = list(val_data.skeleton.body_names)

    # Initialize state
    loop_state = LoopState()

    # Initialize motion generation state
    if FeatureVersion == 4:
        init_motion = get_zero_feature(val_data.skeleton)
        history_motion = val_data.normalize(
            init_motion.unsqueeze(0).expand(1, history_len, -1).to(cfg.device))
    else:
        history_motion = val_data.normalize(
            get_zero_feature().unsqueeze(0).expand(1, history_len,
                                                   -1).to(cfg.device))
    abs_pose = get_zero_abs_pose((1, ), device=cfg.device)

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

    show_fn, viewer = mjc_load_everything(dt, keycb_fn)

    # Start interactive input thread
    input_thread = threading.Thread(target=interactive_input_thread,
                                    args=(loop_state, ))
    input_thread.daemon = True
    input_thread.start()

    logger.info("Starting continuous motion generation...")
    logger.info(
        "Commands: Enter 'x y z yaw(deg)' in terminal, Space/p(pause), Esc/q(quit)"
    )
    logger.info("  yaw(deg): 0°=+X, 90°=+Y, 180°=-X")
    logger.info("  (goal defaults to zero until first input)")

    # Pre-compute grid_size for zero voxel
    grid_size = cfg.denoiser.grid_size

    # Main generation loop
    frame_idx = 0
    while not loop_state.quit_requested and viewer.is_running():
        # Build ego_goal from world goal + current robot pose.
        # Before first user input, use all-zero ego_goal (stand still).
        if loop_state.goal_received:
            world_goal_pos = torch.tensor(
                loop_state.world_goal[:3], device=cfg.device
            ).float().unsqueeze(0)  # [1, 3]
            world_goal_yaw = torch.tensor(
                [loop_state.world_goal[3]], device=cfg.device
            ).float()  # [1]

            reference_pos = abs_pose['root_trans_offset']  # [1, 3]
            reference_rot = abs_pose['root_rot']  # [1, 4] xyzw
            goal_keypoints = None
            if goal_type is GoalType.BODY:
                goal_keypoints_np = load_goal_keypoints_from_reference(
                    goal_reference_path,
                    loop_state.world_goal[:3],
                    float(loop_state.world_goal[3]),
                )
                goal_keypoints = torch.as_tensor(
                    goal_keypoints_np, dtype=torch.float32,
                    device=cfg.device).unsqueeze(0)

            ego_goal = build_ego_goal(
                world_goal_pos, world_goal_yaw,
                reference_pos, reference_rot,
                goal_type=goal_type,
                world_goal_keypoints=goal_keypoints,
            )
        else:
            ego_goal = torch.zeros(
                1, goal_type.dimension, device=cfg.device)

        # Scene condition: all zeros
        voxel = torch.zeros(1, grid_size**3, device=cfg.device)

        # Generate next motion if not paused
        if not loop_state.paused:
            # breakpoint()
            future_motion, motion_dict, abs_pose = generate_next_motion(
                vae=vae_trt,
                denoiser=cfg_denoiser,
                diffusion=diffusion,
                val_data=val_data,
                goal=ego_goal,
                voxel=voxel,
                history_motion=history_motion,
                abs_pose=abs_pose,
                future_len=future_len,
                use_full_sample=cfg.use_full_sample,
                guidance_scale=cfg.guidance_scale,
                ret_fk=True)

            # ── NPZ: accumulate FK results (new frames only, skip first-block history padding) ──
            global _NPZ_BUFFER, _NPZ_FIRST_BLOCK
            skip = history_len if _NPZ_FIRST_BLOCK else 0
            _NPZ_FIRST_BLOCK = False
            dof_pos = motion_dict['dof_pos'][0, skip:].detach().cpu().numpy()             # [T', 23]
            dof_vel = motion_dict['dof_vel'][0, skip:].detach().cpu().numpy()             # [T', 23]
            body_t  = motion_dict['global_translation'][0, skip:].detach().cpu().numpy()  # [T', N, 3]
            body_r  = motion_dict['global_rotation'][0, skip:].detach().cpu().numpy()     # [T', N, 4] xyzw
            _NPZ_BUFFER.append((dof_pos, dof_vel, body_t, body_r))
            # ────────────────────────────────────────────────────────────────────

            # Update history for next generation (autoregressive)
            history_motion = future_motion[:, -history_len:, :]

            # Visualize the motion
            qpos_data, contact_data = motion_dict_to_qpos(motion_dict)

            # Convert to numpy - qpos_data and contact_data are torch tensors
            qpos_np = qpos_data.detach().cpu().numpy()  # [B, T, 30]
            contact_np = contact_data.detach().cpu().numpy()  # [B, T, 2]

            # Update goal vis once per generated block (not every frame)
            _update_goal_vis(viewer, loop_state.world_goal,
                             loop_state.goal_received)

            # Show each frame of the generated motion
            for t in range(qpos_np.shape[1]):
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
