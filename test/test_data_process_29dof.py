"""Parity and format checks for native 29-DoF Bones-SEED preparation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "dataset" / "data_process" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_script("convert_soma_csv_to_motion_lib")
packer = _load_script("pack_motion_lib_to_textop")


def test_packer_retains_all_wrist_dofs():
    frames = 8
    dof = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29)
    entry = {
        "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
        "root_rot": np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32),
        "dof": dof,
        "contact_mask": np.ones((frames, 2), dtype=np.float32),
        "fps": 50,
    }

    packed = packer.motion_lib_entry_to_textop("idle_test", entry)

    assert packed is not None
    np.testing.assert_array_equal(packed["motion"]["dof"], dof)
    assert packer.FEATURE_DIM_V3 == 69


def test_parallel_packer_matches_serial_output(tmp_path):
    frames = 8
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_pkls = []
    for idx in range(3):
        dof = np.full((frames, 29), idx, dtype=np.float32)
        entry = {
            "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
            "root_rot": np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32),
            "dof": dof,
            "contact_mask": np.ones((frames, 2), dtype=np.float32),
            "fps": 50,
        }
        source_path = source_dir / f"motion_{idx}.pkl"
        joblib.dump({f"idle_{idx}": entry}, source_path)
        source_pkls.append((source_path, source_dir))

    outputs = []
    for workers in (1, 2):
        out = tmp_path / f"out_{workers}"
        (out / "samples").mkdir(parents=True)
        outputs.append(packer.pack_source_files(
            source_pkls, out, min_frames=0, sample_compress=0,
            workers=workers,
        ))

    serial_manifest, serial_skipped, serial_fps = outputs[0]
    parallel_manifest, parallel_skipped, parallel_fps = outputs[1]
    assert serial_manifest == parallel_manifest
    assert serial_skipped == parallel_skipped == 0
    assert serial_fps == parallel_fps == {50}

    for record in parallel_manifest:
        serial = joblib.load(tmp_path / "out_1" / record["_data_path"])
        parallel = joblib.load(tmp_path / "out_2" / record["_data_path"])
        np.testing.assert_array_equal(
            serial["motion"]["dof"], parallel["motion"]["dof"]
        )


def test_torch_fk_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    xml_path = ROOT / "TextOpRobotMDAR" / "description" / "robots" / "g1" / "g1_29dof_with_collision.xml"
    model, data = converter._get_mj_model(str(xml_path))
    torch_fk = converter._get_torch_fk_model(str(xml_path), "cpu")

    rng = np.random.default_rng(7)
    frames = 12
    root_trans = rng.normal(scale=0.1, size=(frames, 3)).astype(np.float32)
    root_trans[:, 2] += 0.8
    root_rot = Rotation.from_rotvec(
        rng.normal(scale=0.2, size=(frames, 3))
    ).as_quat().astype(np.float32)
    dof = rng.normal(scale=0.25, size=(frames, 29)).astype(np.float32)
    sampled = np.arange(0, frames, 2)

    foot_torch, geom_pos_torch, geom_rot_torch = torch_fk.forward(
        root_trans, root_rot, dof, sampled
    )
    foot_mujoco = np.empty_like(foot_torch)
    geom_pos_mujoco = np.empty_like(geom_pos_torch)
    geom_rot_mujoco = np.empty_like(geom_rot_torch)
    left_id, right_id = converter._find_foot_body_ids(model)
    sample_lookup = {int(frame): i for i, frame in enumerate(sampled)}

    for frame in range(frames):
        data.qpos[:3] = root_trans[frame]
        data.qpos[3:7] = root_rot[frame, [3, 0, 1, 2]]
        data.qpos[7:36] = dof[frame]
        mujoco.mj_forward(model, data)
        foot_mujoco[frame] = data.xpos[[left_id, right_id]]
        sample_idx = sample_lookup.get(frame)
        if sample_idx is not None:
            geom_pos_mujoco[sample_idx] = data.geom_xpos[torch_fk.geom_ids]
            geom_rot_mujoco[sample_idx] = data.geom_xmat[
                torch_fk.geom_ids
            ].reshape(-1, 3, 3)

    np.testing.assert_allclose(foot_torch, foot_mujoco, atol=2e-6)
    np.testing.assert_allclose(geom_pos_torch, geom_pos_mujoco, atol=2e-6)
    np.testing.assert_allclose(geom_rot_torch, geom_rot_mujoco, atol=2e-6)


def test_vectorized_mob_is_identical_to_scalar_reference():
    pytest.importorskip("mujoco")
    pytest.importorskip("torch")
    xml_path = ROOT / "TextOpRobotMDAR" / "description" / "robots" / "g1" / "g1_29dof_with_collision.xml"
    rng = np.random.default_rng(11)
    frames = 60
    root_trans = np.zeros((frames, 3), dtype=np.float32)
    root_trans[:, 0] = np.linspace(0.0, 0.8, frames)
    root_trans[:, 2] = 0.8
    root_rot = Rotation.from_euler(
        "z", np.linspace(-0.4, 0.6, frames)
    ).as_quat().astype(np.float32)
    dof = rng.normal(scale=0.3, size=(frames, 29)).astype(np.float32)
    common = {
        "fps": 50,
        "xml_path": str(xml_path),
        "mob": True,
        "mob_frame_stride": 2,
        "fk_backend": "torch",
        "torch_device": "cpu",
    }

    scalar = converter.compute_contact_and_mob(
        root_trans, root_rot, dof, mob_raster_backend="scalar", **common
    )
    vectorized = converter.compute_contact_and_mob(
        root_trans, root_rot, dof, mob_raster_backend="vectorized", **common
    )

    np.testing.assert_array_equal(
        vectorized["contact_mask"], scalar["contact_mask"]
    )
    np.testing.assert_array_equal(
        vectorized["sliding_mask"], scalar["sliding_mask"]
    )
    np.testing.assert_array_equal(
        vectorized["scene"]["occu_global"], scalar["scene"]["occu_global"]
    )
    np.testing.assert_array_equal(vectorized["scene"]["llb"], scalar["scene"]["llb"])
