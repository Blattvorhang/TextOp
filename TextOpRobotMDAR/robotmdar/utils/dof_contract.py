"""Resolve and validate the model-facing 23/29-DoF G1 contract."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from robotmdar.dtype.motion import (
    G1_23DOF_FROM_29DOF_INDICES,
    G1_MUJOCO_DOF_JOINT_NAMES,
    G1_MUJOCO_DOF_LINK_NAMES,
    motion_feature_dim_for_dof,
)


def expected_g1_names(dof_dim: int):
    """Return canonical MuJoCo joint/body order for a selected contract."""
    if int(dof_dim) == 29:
        return G1_MUJOCO_DOF_JOINT_NAMES, G1_MUJOCO_DOF_LINK_NAMES
    if int(dof_dim) == 23:
        indices = G1_23DOF_FROM_29DOF_INDICES
        return (
            tuple(G1_MUJOCO_DOF_JOINT_NAMES[index] for index in indices),
            tuple(G1_MUJOCO_DOF_LINK_NAMES[index] for index in indices),
        )
    motion_feature_dim_for_dof(dof_dim)
    raise AssertionError("unreachable")


def configure_dof_contract(cfg: DictConfig) -> int:
    """Select feature width and the matching locked/full-wrist skeleton."""
    dof_dim = int(cfg.data.dof_dim)
    expected_nfeats = motion_feature_dim_for_dof(dof_dim)
    if int(cfg.data.nfeats) != expected_nfeats:
        raise ValueError(
            f"dof_dim={dof_dim} requires data.nfeats={expected_nfeats}, "
            f"got {cfg.data.nfeats}"
        )

    if dof_dim == 23:
        variant_path = (
            Path(__file__).resolve().parents[1]
            / 'config/skeleton/g1_23dof.yaml'
        )
        cfg.skeleton = OmegaConf.merge(
            cfg.skeleton, OmegaConf.load(variant_path)
        )
    cfg.skeleton.dof_dim = dof_dim
    return dof_dim


def validate_training_contract(cfg, datasets, vae, denoiser=None) -> None:
    """Fail early if data, FK, normalization, and models disagree."""
    dof_dim = int(cfg.data.dof_dim)
    nfeats = motion_feature_dim_for_dof(dof_dim)
    expected_joint_names, expected_link_names = expected_g1_names(dof_dim)
    if int(cfg.data.nfeats) != nfeats:
        raise ValueError(
            f"dof_dim={dof_dim} requires data.nfeats={nfeats}, "
            f"got {cfg.data.nfeats}"
        )

    for split, dataset in datasets:
        source_dof_dim = int(dataset.source_dof_dim)
        source_nfeats = motion_feature_dim_for_dof(source_dof_dim)
        stats = dataset.statistics
        stats_dof = int(stats.get('dof_dim', source_dof_dim))
        stats_nfeats = int(stats.get('nfeats', source_nfeats))
        if stats_dof != source_dof_dim or stats_nfeats != source_nfeats:
            raise ValueError(
                f"{split} statistics describe dof_dim={stats_dof}, "
                f"nfeats={stats_nfeats}, but stored motion was inferred as "
                f"{source_dof_dim}-DoF/{source_nfeats}-D"
            )
        if source_dof_dim != dof_dim and not (
            source_dof_dim == 29 and dof_dim == 23
        ):
            raise ValueError(
                f"{split} cannot adapt stored {source_dof_dim}-DoF motion to "
                f"the selected {dof_dim}-DoF contract"
            )
        if int(dataset.dof_dim) != dof_dim:
            raise ValueError(
                f"{split} dataset exposes {dataset.dof_dim} DoFs, expected "
                f"{dof_dim}"
            )
        if int(dataset.skeleton.fk.num_dof) != dof_dim:
            raise ValueError(
                f"{split} skeleton has {dataset.skeleton.fk.num_dof} DoFs, "
                f"expected {dof_dim}"
            )

        stats_order = stats.get('dof_order')
        if stats_order is not None and str(stats_order).lower() != 'mujoco':
            raise ValueError(
                f"{split} dataset uses {stats_order!r} DOF order, expected "
                "'mujoco'"
            )
        stats_names = stats.get('dof_names')
        source_joint_names, _ = expected_g1_names(source_dof_dim)
        if stats_names is not None and tuple(stats_names) != source_joint_names:
            raise ValueError(
                f"{split} stored DOF names do not match the canonical "
                f"{source_dof_dim}-DoF MuJoCo order"
            )
        if tuple(dataset.skeleton.fk.dof_joint_names) != expected_joint_names:
            raise ValueError(
                f"{split} MJCF joint order does not match the canonical "
                f"{dof_dim}-DoF training order"
            )
        if tuple(dataset.skeleton.fk.body_names[1:]) != expected_link_names:
            raise ValueError(
                f"{split} MJCF body order does not match the canonical "
                f"{dof_dim}-DoF training order"
            )
        if dataset.mean.shape[-1] != nfeats or dataset.std.shape[-1] != nfeats:
            raise ValueError(
                f"{split} normalization has shape mean={tuple(dataset.mean.shape)}, "
                f"std={tuple(dataset.std.shape)}; expected {nfeats}-D"
            )

    if vae.skel_embedding.in_features != nfeats:
        raise ValueError(
            f"VAE encoder expects {vae.skel_embedding.in_features} features, "
            f"expected {nfeats}"
        )
    if vae.final_layer.out_features != nfeats:
        raise ValueError(
            f"VAE decoder emits {vae.final_layer.out_features} features, "
            f"expected {nfeats}"
        )

    if denoiser is not None:
        expected_history_shape = (int(cfg.data.history_len), nfeats)
        actual_history_shape = tuple(int(dim) for dim in denoiser.history_shape)
        if actual_history_shape != expected_history_shape:
            raise ValueError(
                f"Denoiser history_shape={actual_history_shape}, expected "
                f"{expected_history_shape}"
            )
