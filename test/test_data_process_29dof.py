"""Parity and format checks for native 29-DoF Bones-SEED preparation."""

import xml.etree.ElementTree as ET
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


def _mjcf_dof_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        joint.attrib["name"]
        for joint in root.findall(".//worldbody//joint")
        if joint.attrib.get("type") != "free"
    ]


def _mjcf_actuator_joint_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [node.attrib["joint"] for node in root.findall("./actuator/*")]


def test_all_mjcf_and_preprocessing_joint_orders_match():
    robot_dir = (
        ROOT / "TextOpRobotMDAR" / "description" / "robots" / "g1"
    )
    expected = list(packer.TARGET_DOF_NAMES)
    assert list(converter.MUJOCO_DOF_JOINT_NAMES) == expected

    for xml_path in sorted(robot_dir.glob("g1_29dof*.xml")):
        assert _mjcf_dof_names(xml_path) == expected, xml_path.name
        assert _mjcf_actuator_joint_names(xml_path) == expected, xml_path.name

    old_23 = _mjcf_dof_names(
        robot_dir / "g1_23dof_lock_wrist_fitmotionONLY.xml"
    )
    assert old_23 == expected[:19] + expected[22:26]


def test_packer_rejects_semantically_permuted_29dof_entry():
    frames = 2
    wrong_names = list(packer.TARGET_DOF_NAMES)
    wrong_names[19], wrong_names[22] = wrong_names[22], wrong_names[19]
    entry = {
        "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
        "root_rot": np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32),
        "dof": np.zeros((frames, 29), dtype=np.float32),
        "dof_order": "mujoco",
        "dof_names": wrong_names,
        "contact_mask": np.ones((frames, 2), dtype=np.float32),
        "fps": 50,
    }

    with pytest.raises(ValueError, match="joint order"):
        packer.motion_lib_entry_to_textop("bad_order", entry)


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
    metadata_lookup = {}
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
        motion_name = f"idle_{idx}"
        joblib.dump({motion_name: entry}, source_path)
        source_pkls.append((source_path, source_dir))
        metadata_lookup[motion_name] = {
            "filename": motion_name,
            "content_short_description": "standing idle",
            "content_type_of_movement": "standing idle",
        }

    outputs = []
    for workers in (1, 2):
        out = tmp_path / f"out_{workers}"
        (out / "samples").mkdir(parents=True)
        outputs.append(packer.pack_source_files(
            source_pkls, out, min_frames=0, sample_compress=0,
            workers=workers, metadata_lookup=metadata_lookup,
            temporal_lookup={},
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


def test_directory_packer_uses_folder_prefix_without_repeating_motion_name(tmp_path):
    frames = 8
    source_dir = tmp_path / "source"
    session_dir = source_dir / "221010"
    session_dir.mkdir(parents=True)
    entry = {
        "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
        "root_rot": np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32),
        "dof": np.zeros((frames, 29), dtype=np.float32),
        "contact_mask": np.ones((frames, 2), dtype=np.float32),
        "fps": 50,
    }
    source_path = session_dir / "walk_ff_loop_180_R_003__A045_M.pkl"
    joblib.dump({"walk_ff_loop_180_R_003__A045_M": entry}, source_path)
    metadata_lookup = {
        "walk_ff_loop_180_R_003__A045_M": {
            "filename": "walk_ff_loop_180_R_003__A045_M",
            "content_short_description": "walk forward",
            "content_type_of_movement": "walking",
        }
    }
    temporal_lookup = {
        "walk_ff_loop_180_R_003__A045_M": {
            "filename": "walk_ff_loop_180_R_003__A045_M",
            "events": [
                {
                    "start_time": 0.013,
                    "end_time": 0.087,
                    "description": "A person walks forward.",
                }
            ],
        }
    }

    out = tmp_path / "out"
    (out / "samples").mkdir(parents=True)
    manifest, skipped, fps_values = packer.pack_source_files(
        [(source_path, source_dir)],
        out,
        min_frames=0,
        sample_compress=0,
        workers=1,
        metadata_lookup=metadata_lookup,
        temporal_lookup=temporal_lookup,
    )

    assert skipped == 0
    assert fps_values == {50}
    assert manifest[0]["_source"] == "221010__walk_ff_loop_180_R_003__A045_M"
    assert manifest[0]["frame_ann"] == [
        (0.0, frames / 50, ["walk forward"], ["walk"]),
        (0.0, 0.1, ["A person walks forward.", "walks forward", "walk forward"], ["walk"]),
    ]
    stored = joblib.load(out / manifest[0]["_data_path"])
    assert stored["_source"] == manifest[0]["_source"]
    assert stored["frame_ann"] == manifest[0]["frame_ann"]
    assert stored["length"] == frames


def test_temporal_times_snap_to_50hz_frame_grid():
    assert packer._snap_event_times_to_fps(0.0, 1.88, 50, 120) == (0.0, 1.88)
    assert packer._snap_event_times_to_fps(0.013, 0.087, 50, 120) == (0.0, 0.1)


def test_augmented_motion_uses_base_metadata_lookup(tmp_path):
    frames = 8
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    entry = {
        "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
        "root_rot": np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32),
        "dof": np.zeros((frames, 29), dtype=np.float32),
        "contact_mask": np.ones((frames, 2), dtype=np.float32),
        "fps": 50,
    }
    source_name = "walk_ff_loop_180_R_003__A045_M_aug_003"
    source_path = source_dir / f"{source_name}.pkl"
    joblib.dump({source_name: entry}, source_path)

    metadata_lookup = {
        "walk_ff_loop_180_R_003__A045_M": {
            "filename": "walk_ff_loop_180_R_003__A045_M",
            "content_short_description": "walk forward",
            "content_type_of_movement": "walking",
        }
    }
    temporal_lookup = {
        "walk_ff_loop_180_R_003__A045_M": {
            "filename": "walk_ff_loop_180_R_003__A045_M",
            "events": [
                {
                    "start_time": 0.0,
                    "end_time": 0.2,
                    "description": "A person walks forward.",
                }
            ],
        }
    }

    out = tmp_path / "out"
    (out / "samples").mkdir(parents=True)
    manifest, skipped, fps_values = packer.pack_source_files(
        [(source_path, source_dir)],
        out,
        min_frames=0,
        sample_compress=0,
        workers=1,
        metadata_lookup=metadata_lookup,
        temporal_lookup=temporal_lookup,
    )

    assert skipped == 0
    assert fps_values == {50}
    assert manifest[0]["_source"] == f"walk_ff_loop_180_R_003__A045_M_aug_003"
    assert manifest[0]["frame_ann"][0][2] == ["walk forward"]


def test_motion_split_key_groups_original_and_mirror_sources():
    original = "221010__walk_ff_loop_180_R_003__A045"
    mirrored = "221010__walk_ff_loop_180_R_003__A045_M"
    augmented_original = "221010__walk_ff_loop_180_R_003__A045_aug_003"
    augmented_mirror = "221010__walk_ff_loop_180_R_003__A045_M_aug_003"

    split_key = packer._motion_split_key(original)
    assert split_key == packer._motion_split_key(mirrored)
    assert split_key == packer._motion_split_key(augmented_original)
    assert split_key == packer._motion_split_key(augmented_mirror)


def test_grouped_train_val_split_keeps_mirrors_together():
    mirror_original = "walk_ff_loop_180_R_003__A045"
    mirror_copy = "walk_ff_loop_180_R_003__A045_M"
    augmented = "walk_ff_loop_180_R_003__A045_aug_003"
    augmented_mirror = "walk_ff_loop_180_R_003__A045_M_aug_003"
    manifest = [
        {"_source": mirror_original, "length": 10, "_fps": 50, "frame_ann": []},
        {"_source": mirror_copy, "length": 10, "_fps": 50, "frame_ann": []},
        {"_source": augmented, "length": 10, "_fps": 50, "frame_ann": []},
        {"_source": augmented_mirror, "length": 10, "_fps": 50, "frame_ann": []},
        {"_source": "idle_001__A001", "length": 10, "_fps": 50, "frame_ann": []},
        {"_source": "jump_001__A002", "length": 10, "_fps": 50, "frame_ann": []},
    ]

    train_data, val_data, stats = packer.split_manifest_train_val(
        manifest, val_ratio=0.5, seed=7
    )

    train_keys = {packer._motion_split_key(record["_source"]) for record in train_data}
    val_keys = {packer._motion_split_key(record["_source"]) for record in val_data}
    mirror_key = packer._motion_split_key(mirror_original)
    mirror_split_count = int(mirror_key in train_keys) + int(mirror_key in val_keys)

    assert train_keys.isdisjoint(val_keys)
    assert mirror_split_count == 1
    assert stats["leakage_groups"] == 0
    assert stats["paired_original_mirror_groups"] == 1
    assert stats["augmented_records"] == 2


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
