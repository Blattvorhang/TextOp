#!/usr/bin/env python3
"""Generate SONIC-validated flat fall-recovery augmentation clips.

This implements Planner V7.1 section 9 for the motion_lib stage:
``stand_up_lying*`` clips are perturbed over a short prefix, validated by one
headless SONIC trial, and written back as motion_lib-schema PKLs that stage 2
packing can ingest directly.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.data_process.pack_motion_lib_to_textop import TARGET_DOF_NAMES

DEFAULT_INPUT_DIR = REPO_ROOT / "data/motion_lib_filtered"
DEFAULT_OUTPUT_SUBDIR = "aug_fall_recovery"
DEFAULT_OCC_HIPC_ROOT = REPO_ROOT.parent / "occHIPC"
DEFAULT_G1_XML = (
    REPO_ROOT / "TextOpRobotMDAR/description/robots/g1/g1_29dof.xml"
)
DEFAULT_COLLISION_XML = (
    REPO_ROOT
    / "TextOpRobotMDAR/description/robots/g1/g1_29dof_with_collision.xml"
)

JOINT_AUG_AMPS_RECOVERY = (
    ("shoulder", 1.50),
    ("elbow", 1.50),
    ("hip", 0.80),
    ("knee", 1.00),
    ("ankle", 0.10),
    ("waist", 0.50),
    ("wrist", 0.20),
)
ROOT_AUG_AMPS_RECOVERY = {"x": 0.05, "y": 0.10, "z": 1.50, "h": 0.03}
DEFAULT_CANDIDATE_MULTIPLIER = 8
DEFAULT_CANDIDATE_SLACK = 16
DEFAULT_MAX_AUTO_WORKERS = 8


@dataclass(frozen=True)
class AugmentConfig:
    accepted_per_original: int
    max_candidates_per_original: int
    n_lying_min: int
    n_lying_max: int
    collision_xml: Path
    fk_backend: str
    torch_device: str
    occ_hipc_root: Path
    device: str
    validate: bool
    recompute_contact: bool
    quiet_policy_init: bool


@dataclass(frozen=True)
class SourceResult:
    index: int
    source_name: str
    accepted_entries: list[tuple[str, dict]]
    rejects: list[str]
    candidates: int


_WORKER_CONFIG: AugmentConfig | None = None
_WORKER_JOINT_LIMITS: np.ndarray | None = None
_WORKER_VALIDATOR: "SonicValidator | None" = None


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return 3.0 * u * u - 2.0 * u * u * u


def _window_weights(window_len: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, window_len + 1, dtype=np.float32)
    return (1.0 - _smoothstep(u)).astype(np.float32)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(np.asarray(a), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    ).astype(np.float32)


def _dof_amp_vector() -> np.ndarray:
    values = []
    for name in TARGET_DOF_NAMES:
        for prefix, amplitude in JOINT_AUG_AMPS_RECOVERY:
            if prefix in name:
                values.append(amplitude)
                break
        else:
            raise ValueError(f"No recovery augmentation amplitude for {name!r}")
    return np.asarray(values, dtype=np.float32)


def _load_joint_limits(xml_path: Path) -> np.ndarray:
    import xml.etree.ElementTree as ETree

    limits = np.full((len(TARGET_DOF_NAMES), 2), (-np.pi, np.pi),
                     dtype=np.float32)
    try:
        ranges = {}
        for node in ETree.parse(xml_path).getroot().findall(".//joint"):
            name = node.attrib.get("name")
            rng = node.attrib.get("range")
            if not name or not rng:
                continue
            parts = [float(value) for value in rng.split()]
            if len(parts) == 2:
                ranges[name] = parts
        for idx, name in enumerate(TARGET_DOF_NAMES):
            if name in ranges:
                limits[idx] = ranges[name]
    except (OSError, ETree.ParseError, ValueError) as exc:
        print(f"[WARN] Could not parse joint limits from {xml_path}: {exc}")
    return limits


def _should_augment_name(name: str) -> bool:
    lower = name.lower()
    return (
        "stand_up_lying" in lower
        and "faint_stand_up_lying" not in lower
        and "lying_side" not in lower
    )


def _clean_original_name(name: str) -> str:
    stem = Path(str(name)).stem
    lower = stem.lower()
    idx = lower.find("stand_up_lying")
    if idx >= 0:
        return stem[idx:]
    return stem


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _iter_source_entries(
    input_dir: Path,
    output_dir: Path,
) -> Iterable[tuple[Path, str, dict]]:
    for path in sorted(input_dir.rglob("stand_up_lying*.pkl")):
        if _is_under(path, output_dir):
            continue
        raw = joblib.load(path)
        if not isinstance(raw, dict) or not raw:
            print(f"[WARN] Skipping non-dict PKL: {path}")
            continue

        if "motion" in raw and isinstance(raw["motion"], dict):
            raw_name = str(raw.get("_source") or path.stem)
            entries = [(raw_name, raw["motion"])]
        else:
            entries = [(str(key), value) for key, value in raw.items()]

        for raw_name, entry in entries:
            source_name = _clean_original_name(raw_name)
            if _should_augment_name(source_name):
                yield path, source_name, entry


def _copy_motion_entry(entry: dict) -> dict:
    out = dict(entry)
    for key in ("root_trans_offset", "root_rot", "dof", "contact_mask"):
        out[key] = np.asarray(entry[key], dtype=np.float32).copy()
    out["fps"] = int(entry.get("fps", 50))
    out["dof_order"] = "mujoco"
    out["dof_names"] = list(TARGET_DOF_NAMES)
    if "scene" in entry:
        out["scene"] = entry["scene"]
    if "sliding_mask" in entry:
        sliding = np.asarray(entry["sliding_mask"], dtype=np.float32)
    else:
        sliding = np.zeros_like(out["contact_mask"], dtype=np.float32)
    out["sliding_mask"] = np.zeros_like(sliding, dtype=np.float32)
    return out


def _validate_entry_shape(name: str, entry: dict) -> None:
    required = ("root_trans_offset", "root_rot", "dof", "contact_mask")
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")
    dof = np.asarray(entry["dof"])
    root_pos = np.asarray(entry["root_trans_offset"])
    root_rot = np.asarray(entry["root_rot"])
    contact = np.asarray(entry["contact_mask"])
    if dof.ndim != 2 or dof.shape[1] != len(TARGET_DOF_NAMES):
        raise ValueError(f"{name} has invalid dof shape {dof.shape}")
    if root_pos.shape != (dof.shape[0], 3):
        raise ValueError(f"{name} has invalid root_trans_offset shape {root_pos.shape}")
    if root_rot.shape != (dof.shape[0], 4):
        raise ValueError(f"{name} has invalid root_rot shape {root_rot.shape}")
    if contact.shape != (dof.shape[0], 2):
        raise ValueError(f"{name} has invalid contact_mask shape {contact.shape}")


def _augment_entry(
    source_name: str,
    entry: dict,
    rng: np.random.Generator,
    n_lying: int,
    joint_limits: np.ndarray,
) -> tuple[dict, dict]:
    _validate_entry_shape(source_name, entry)
    out = _copy_motion_entry(entry)
    T = out["dof"].shape[0]
    if T <= n_lying:
        raise ValueError(
            f"{source_name} has only {T} frames, cannot leave frame "
            f"n_lying={n_lying} unperturbed"
        )

    w = _window_weights(n_lying)[:n_lying]
    amps = _dof_amp_vector()

    x = out["dof"][:n_lying]
    active = w > 0.0
    lo_q = np.max(
        (joint_limits[:, 0] - x[active]) / w[active, None],
        axis=0,
    )
    hi_q = np.min(
        (joint_limits[:, 1] - x[active]) / w[active, None],
        axis=0,
    )
    lower = np.maximum(lo_q, -amps)
    upper = np.minimum(hi_q, amps)
    q = lower + (upper - lower) * rng.random(lower.shape, dtype=np.float32)
    q = q.astype(np.float32)
    q[lower > upper] = 0.0
    out["dof"][:n_lying] += w[:, None] * q

    rx = float(rng.uniform(-1.0, 1.0) * ROOT_AUG_AMPS_RECOVERY["x"])
    ry = float(rng.uniform(-1.0, 1.0) * ROOT_AUG_AMPS_RECOVERY["y"])
    rz = float(rng.uniform(-1.0, 1.0) * ROOT_AUG_AMPS_RECOVERY["z"])
    half = 0.5 * w
    zeros = np.zeros_like(half)
    q_x = np.stack([np.sin(half * rx), zeros, zeros, np.cos(half * rx)], axis=-1)
    q_y = np.stack([zeros, np.sin(half * ry), zeros, np.cos(half * ry)], axis=-1)
    q_z = np.stack([zeros, zeros, np.sin(half * rz), np.cos(half * rz)], axis=-1)
    q_off = _quat_mul_xyzw(q_x, _quat_mul_xyzw(q_y, q_z))
    out["root_rot"][:n_lying] = _quat_mul_xyzw(
        out["root_rot"][:n_lying],
        q_off,
    )

    dh = float(rng.uniform(-1.0, 1.0) * ROOT_AUG_AMPS_RECOVERY["h"])
    out["root_trans_offset"][:n_lying, 2] += w * dh
    return out, {
        "n_lying": n_lying,
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "dh": dh,
        "q": q,
    }


def _recompute_contact(
    entry: dict,
    xml_path: Path,
    fk_backend: str,
    torch_device: str,
) -> bool:
    try:
        from dataset.data_process.convert_soma_csv_to_motion_lib import (
            compute_contact_and_mob,
        )

        result = compute_contact_and_mob(
            entry["root_trans_offset"],
            entry["root_rot"],
            entry["dof"],
            float(entry.get("fps", 50)),
            str(xml_path),
            mob=False,
            fk_backend=fk_backend,
            torch_device=torch_device,
        )
        entry["contact_mask"] = result["contact_mask"].astype(np.float32)
        entry["sliding_mask"] = np.zeros_like(entry["contact_mask"], dtype=np.float32)
        return True
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"[WARN] Contact recompute failed; keeping original mask: {exc}")
        return False


class SonicValidator:
    def __init__(self, occ_hipc_root: Path, device: str):
        occ_hipc_root = occ_hipc_root.resolve()
        if str(occ_hipc_root) not in sys.path:
            sys.path.insert(0, str(occ_hipc_root))

        import mujoco
        from pretrain.sonic_trackingmode import SonicWbcPolicy
        from scripts.eval_fall_recovery import (
            G1_XML,
            _lift_above_ground,
            check_success,
        )
        from utils.g1_hardware import (
            ROBOT_QPOS_END,
            action_scale,
            default_angles,
            isaaclab_to_mujoco_reindex,
            kds,
            kps,
            mujoco_to_isaaclab_reindex,
        )
        from utils.g1_motion_loader import G1MotionLoader
        from utils.robot_state import build_robot_state, pd_control
        from utils.sim_config import (
            CONTROL_DECIMATION,
            NUM_ACTIONS,
            SIMULATION_DT,
            resolve_policy_paths,
        )

        enc_path, dec_path = resolve_policy_paths(str(occ_hipc_root / "scripts"))
        self.mujoco = mujoco
        self.G1MotionLoader = G1MotionLoader
        self.check_success = check_success
        self._lift_above_ground = _lift_above_ground
        self.build_robot_state = build_robot_state
        self.pd_control = pd_control
        self.ROBOT_QPOS_END = ROBOT_QPOS_END
        self.action_scale = action_scale
        self.default_angles = default_angles
        self.isaaclab_to_mujoco_reindex = isaaclab_to_mujoco_reindex
        self.mujoco_to_isaaclab_reindex = mujoco_to_isaaclab_reindex
        self.kps = kps
        self.kds = kds
        self.CONTROL_DECIMATION = CONTROL_DECIMATION
        self.NUM_ACTIONS = NUM_ACTIONS

        self.model = mujoco.MjModel.from_xml_path(str(G1_XML))
        self.model.opt.timestep = SIMULATION_DT
        self.data = mujoco.MjData(self.model)
        self.policy = SonicWbcPolicy(enc_path, dec_path, device=device)
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.target_dof_pos = default_angles.copy()
        self.zero_dq = np.zeros(NUM_ACTIONS, dtype=np.float64)

    def _loader_from_entry(self, entry: dict):
        dof_il = entry["dof"][:, self.mujoco_to_isaaclab_reindex].astype(np.float32)
        joint_vel = np.gradient(
            dof_il,
            1.0 / float(entry.get("fps", 50)),
            axis=0,
        ).astype(np.float32)
        root_ori = entry["root_rot"][:, [3, 0, 1, 2]].astype(np.float32)
        payload = {
            "joint_pos": dof_il,
            "joint_vel": joint_vel,
            "root_pos": entry["root_trans_offset"].astype(np.float32),
            "root_ori": root_ori,
            "framerate": float(entry.get("fps", 50)),
            "contact_mask": entry["contact_mask"].astype(np.float32),
            "sliding_mask": entry["sliding_mask"].astype(np.float32),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            return self.G1MotionLoader.from_dict(payload)

    def validate(self, entry: dict) -> tuple[bool, float, float]:
        loader = self._loader_from_entry(entry)
        mujoco = self.mujoco
        d = self.data
        m = self.model
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = loader.body_pos[0, 0]
        d.qpos[3:7] = loader.body_ori[0, 0]
        d.qpos[7:self.ROBOT_QPOS_END] = (
            loader.joint_pos[0][self.isaaclab_to_mujoco_reindex]
        )
        mujoco.mj_forward(m, d)
        self._lift_above_ground(m, d, verbose=False)
        self.target_dof_pos[:] = d.qpos[7:self.ROBOT_QPOS_END].copy()
        self.action[:] = 0.0
        self.policy.reset()
        mujoco.mj_step(m, d)

        pelvis_z = [0.0] * loader.T
        for frame_idx in range(loader.T):
            for _ in range(self.CONTROL_DECIMATION):
                tau = self.pd_control(
                    self.target_dof_pos,
                    d.qpos[7:self.ROBOT_QPOS_END],
                    self.kps,
                    self.zero_dq,
                    d.qvel[6:35],
                    self.kds,
                )
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

            robot_state = self.build_robot_state(d)
            self.action[:] = self.policy.step(loader, frame_idx, robot_state)
            self.target_dof_pos[:] = (
                self.action[self.isaaclab_to_mujoco_reindex]
                * self.action_scale
                + self.default_angles
            )
            pelvis_z[frame_idx] = float(d.qpos[2])

        return self.check_success(pelvis_z)


def _resolve_num_workers(requested: int, source_count: int) -> int:
    if requested < 0:
        raise SystemExit("--num-workers must be >= 0")
    if source_count <= 1:
        return 1
    if requested > 0:
        return min(requested, source_count)

    cpu_count = os.cpu_count() or 1
    return max(1, min(source_count, cpu_count, DEFAULT_MAX_AUTO_WORKERS))


def _limit_worker_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def _source_seeds(seed: int, source_count: int) -> list[int]:
    sequences = np.random.SeedSequence(seed).spawn(source_count)
    return [int(seq.generate_state(1, dtype=np.uint64)[0]) for seq in sequences]


def _init_worker(config: AugmentConfig, joint_limits: np.ndarray) -> None:
    global _WORKER_CONFIG, _WORKER_JOINT_LIMITS, _WORKER_VALIDATOR
    _WORKER_CONFIG = config
    _WORKER_JOINT_LIMITS = joint_limits
    if not config.validate:
        _WORKER_VALIDATOR = None
    elif config.quiet_policy_init:
        with contextlib.redirect_stdout(io.StringIO()):
            _WORKER_VALIDATOR = SonicValidator(config.occ_hipc_root, config.device)
    else:
        _WORKER_VALIDATOR = SonicValidator(config.occ_hipc_root, config.device)


def _augment_source_task(
    source_idx: int,
    source_name: str,
    source_entry: dict,
    seed: int,
) -> SourceResult:
    if _WORKER_CONFIG is None or _WORKER_JOINT_LIMITS is None:
        raise RuntimeError("Augmentation worker was not initialized")

    config = _WORKER_CONFIG
    joint_limits = _WORKER_JOINT_LIMITS
    validator = _WORKER_VALIDATOR
    rng = np.random.default_rng(seed)

    accepted = 0
    candidates = 0
    accepted_entries = []
    rejects = []
    T = int(np.asarray(source_entry["dof"]).shape[0])
    n_max = min(config.n_lying_max, T - 1)
    if config.n_lying_min > n_max:
        raise RuntimeError(
            f"{source_name} has {T} frames, shorter than n_lying_min="
            f"{config.n_lying_min} plus boundary frame"
        )

    while (
        accepted < config.accepted_per_original
        and candidates < config.max_candidates_per_original
    ):
        candidates += 1
        n_lying = int(rng.integers(config.n_lying_min, n_max + 1))
        candidate, _meta = _augment_entry(
            source_name,
            source_entry,
            rng,
            n_lying,
            joint_limits,
        )
        if config.recompute_contact:
            _recompute_contact(
                candidate,
                config.collision_xml,
                config.fk_backend,
                config.torch_device,
            )

        if validator is None:
            ok, final_h, var_h = True, float("nan"), float("nan")
        else:
            ok, final_h, var_h = validator.validate(candidate)

        if ok:
            aug_name = f"{source_name}_aug_{accepted:03d}"
            accepted_entries.append((aug_name, candidate))
            accepted += 1
        else:
            rejects.append(
                f"[REJECT] {source_name} candidate {candidates}: "
                f"h={final_h:.3f}, var={var_h:.4f}"
            )

    if accepted < config.accepted_per_original:
        reject_tail = "\n".join(rejects[-8:])
        suffix = f"\nLast rejects:\n{reject_tail}" if reject_tail else ""
        raise RuntimeError(
            f"{source_name}: accepted {accepted}/"
            f"{config.accepted_per_original} after {candidates} candidates."
            f" Increase --max-candidates-per-original if this motion is valid "
            f"but hard to sample.{suffix}"
        )

    return SourceResult(
        index=source_idx,
        source_name=source_name,
        accepted_entries=accepted_entries,
        rejects=rejects,
        candidates=candidates,
    )


def _run_augmentation_tasks(
    sources: list[tuple[Path, str, dict]],
    config: AugmentConfig,
    joint_limits: np.ndarray,
    num_workers: int,
    seed: int,
) -> list[SourceResult]:
    seeds = _source_seeds(seed, len(sources))
    results: list[SourceResult | None] = [None] * len(sources)

    if num_workers == 1:
        _init_worker(config, joint_limits)
        iterator = enumerate(sources)
        for source_idx, (_path, source_name, source_entry) in tqdm(
            iterator,
            total=len(sources),
            desc="Original clips",
            file=sys.stdout,
        ):
            result = _augment_source_task(
                source_idx,
                source_name,
                source_entry,
                seeds[source_idx],
            )
            for message in result.rejects:
                tqdm.write(message, file=sys.stdout)
            results[result.index] = result
    else:
        _limit_worker_threads()
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(config, joint_limits),
        ) as executor:
            futures = [
                executor.submit(
                    _augment_source_task,
                    source_idx,
                    source_name,
                    source_entry,
                    seeds[source_idx],
                )
                for source_idx, (_path, source_name, source_entry)
                in enumerate(sources)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Original clips",
                file=sys.stdout,
            ):
                result = future.result()
                for message in result.rejects:
                    tqdm.write(message, file=sys.stdout)
                results[result.index] = result

    missing = [idx for idx, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"Missing augmentation results for source indices {missing}")
    return [result for result in results if result is not None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Planner V7.1 fall-recovery augmentation clips."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--accepted-per-original", type=int, default=8)
    parser.add_argument(
        "--max-candidates-per-original",
        type=int,
        default=None,
        help=(
            "Candidate trials per original clip before giving up. Default is "
            f"max(k*{DEFAULT_CANDIDATE_MULTIPLIER}, "
            f"k+{DEFAULT_CANDIDATE_SLACK})."
        ),
    )
    parser.add_argument("--n-lying-min", type=int, default=20)
    parser.add_argument("--n-lying-max", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    parser.add_argument("--collision-xml", type=Path, default=DEFAULT_COLLISION_XML)
    parser.add_argument("--fk-backend", choices=["torch", "mujoco"], default="torch")
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--occ-hipc-root", type=Path, default=DEFAULT_OCC_HIPC_ROOT)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "tensorrt"],
        default="cpu",
        help="SONIC ONNX Runtime device for validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Worker processes over original clips. Use 0 for auto, "
            f"capped at {DEFAULT_MAX_AUTO_WORKERS}; 1 for serial."
        ),
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Generate clips without SONIC acceptance checks.",
    )
    parser.add_argument(
        "--keep-contact",
        dest="recompute_contact",
        action="store_false",
        help="Keep original contact_mask instead of running stage-1 contact FK.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(validate=True, recompute_contact=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.accepted_per_original < 1:
        raise SystemExit("--accepted-per-original must be positive")
    if args.n_lying_min < 1 or args.n_lying_max < args.n_lying_min:
        raise SystemExit("Invalid n_lying bounds")

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / DEFAULT_OUTPUT_SUBDIR
    )
    sources = list(_iter_source_entries(input_dir, output_dir))
    if not sources:
        raise SystemExit(f"No stand_up_lying*.pkl sources found under {input_dir}")

    max_candidates = args.max_candidates_per_original
    if max_candidates is None:
        max_candidates = max(
            args.accepted_per_original * DEFAULT_CANDIDATE_MULTIPLIER,
            args.accepted_per_original + DEFAULT_CANDIDATE_SLACK,
        )
    if max_candidates < args.accepted_per_original:
        raise SystemExit("--max-candidates-per-original must be >= accepted target")

    num_workers = _resolve_num_workers(args.num_workers, len(sources))
    print(f"[Discover] {len(sources)} original clip(s) under {input_dir}")
    print(f"[Output]   {output_dir}")
    print(
        "[Augment]  "
        f"k={args.accepted_per_original}, "
        f"n_lying=[{args.n_lying_min}, {args.n_lying_max}], "
        f"max candidates/original={max_candidates}"
    )
    print(f"[Validate] {'SONIC one-trial acceptance' if args.validate else 'disabled'}")
    worker_mode = "auto" if args.num_workers == 0 else "requested"
    print(f"[Workers]  {num_workers} process(es) ({worker_mode})")
    sys.stdout.flush()
    if args.dry_run:
        for path, source_name, _entry in sources:
            print(f"  {source_name}  ({path})")
        return

    existing_outputs = []
    if output_dir.exists():
        existing_outputs = sorted(output_dir.glob("*_aug_*.pkl"))
        legacy_bundles = sorted(output_dir.glob("aug_*.pkl"))
        existing_outputs.extend(
            path for path in legacy_bundles if path not in existing_outputs
        )
    if existing_outputs and not args.overwrite:
        raise SystemExit(
            f"{output_dir} already contains augmentation PKLs; pass --overwrite "
            "to replace them."
        )
    if existing_outputs and args.overwrite:
        for path in existing_outputs:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_limits = _load_joint_limits(args.g1_xml)
    config = AugmentConfig(
        accepted_per_original=args.accepted_per_original,
        max_candidates_per_original=max_candidates,
        n_lying_min=args.n_lying_min,
        n_lying_max=args.n_lying_max,
        collision_xml=args.collision_xml.resolve(),
        fk_backend=args.fk_backend,
        torch_device=args.torch_device,
        occ_hipc_root=args.occ_hipc_root.resolve(),
        device=args.device,
        validate=args.validate,
        recompute_contact=args.recompute_contact,
        quiet_policy_init=num_workers > 1,
    )
    results = _run_augmentation_tasks(
        sources,
        config,
        joint_limits,
        num_workers,
        args.seed,
    )
    for result in results:
        for aug_name, candidate in result.accepted_entries:
            out_path = output_dir / f"{aug_name}.pkl"
            joblib.dump({aug_name: candidate}, out_path, compress=3)
    total_candidates = sum(result.candidates for result in results)
    print(
        f"[Done] Wrote {len(sources) * args.accepted_per_original} "
        f"single-motion PKL file(s) "
        f"from {total_candidates} candidate(s)."
    )


if __name__ == "__main__":
    main()
