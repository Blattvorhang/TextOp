from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ETree

import torch


END_EFFECTOR_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")


@dataclass(frozen=True)
class EndEffectorAnchor:
    name: str
    source_type: str
    source_name: str
    parent_body: str
    parent_body_index: int
    local_pos: torch.Tensor

    def meta(self) -> dict:
        data = {
            "name": self.name,
            "type": self.source_type,
            "source_name": self.source_name,
            "parent_body": self.parent_body,
            "local_pos": [float(v) for v in self.local_pos.tolist()],
        }
        if self.source_type == "body":
            data["resolved_from"] = self.parent_body
        return data


def _parse_vec3(raw: str | None) -> torch.Tensor:
    if raw is None:
        return torch.zeros(3, dtype=torch.float32)
    values = [float(v) for v in raw.split()]
    if len(values) != 3:
        raise ValueError(f"MJCF vec3 must have 3 values, got {raw!r}")
    return torch.tensor(values, dtype=torch.float32)


def _mjcf_elements_by_body(mjcf_file: Path):
    tree = ETree.parse(mjcf_file)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"Invalid MJCF {mjcf_file}: missing worldbody")

    bodies = {}
    sites = {}
    geoms_by_mesh = {}
    geoms_by_name = {}

    def visit(body):
        body_name = body.attrib.get("name")
        if body_name is None:
            return
        bodies[body_name] = body
        for site in body.findall("site"):
            name = site.attrib.get("name")
            if name is not None:
                sites[name] = (body_name, site)
        for geom in body.findall("geom"):
            name = geom.attrib.get("name")
            mesh = geom.attrib.get("mesh")
            if name is not None:
                geoms_by_name[name] = (body_name, geom)
            if mesh is not None:
                geoms_by_mesh.setdefault(mesh, []).append((body_name, geom))
        for child in body.findall("body"):
            visit(child)

    for root_body in worldbody.findall("body"):
        visit(root_body)
    return bodies, sites, geoms_by_mesh, geoms_by_name


def _body_index(skeleton, body_name: str) -> int:
    try:
        return skeleton.fk.body_names.index(body_name)
    except ValueError as exc:
        raise ValueError(
            f"End-effector parent body {body_name!r} is not present in "
            f"active FK body list from {skeleton.fk.mjcf_file}"
        ) from exc


def _resolve_geom_anchor(
    skeleton,
    *,
    name: str,
    mesh: str,
    expected_parent_body: str,
    geoms_by_mesh: dict,
    geoms_by_name: dict,
) -> EndEffectorAnchor:
    candidates = [
        item for item in geoms_by_mesh.get(mesh, [])
        if item[0] == expected_parent_body
    ]
    if not candidates and mesh in geoms_by_name:
        body_name, geom = geoms_by_name[mesh]
        if body_name == expected_parent_body:
            candidates = [(body_name, geom)]
    if not candidates:
        raise ValueError(
            f"Active MJCF {skeleton.fk.mjcf_file} has no geom mesh/name "
            f"{mesh!r} under {expected_parent_body!r}"
        )
    parent_body, geom = candidates[0]
    return EndEffectorAnchor(
        name=name,
        source_type="geom",
        source_name=mesh,
        parent_body=parent_body,
        parent_body_index=_body_index(skeleton, parent_body),
        local_pos=_parse_vec3(geom.attrib.get("pos")),
    )


def _resolve_site_or_body_anchor(
    skeleton,
    *,
    name: str,
    site: str,
    fallback_body: str,
    bodies: dict,
    sites: dict,
) -> EndEffectorAnchor:
    if site in sites:
        parent_body, site_node = sites[site]
        if parent_body != fallback_body:
            raise ValueError(
                f"End-effector site {site!r} is under {parent_body!r}, "
                f"expected {fallback_body!r} in {skeleton.fk.mjcf_file}"
            )
        return EndEffectorAnchor(
            name=name,
            source_type="site",
            source_name=site,
            parent_body=parent_body,
            parent_body_index=_body_index(skeleton, parent_body),
            local_pos=_parse_vec3(site_node.attrib.get("pos")),
        )
    if fallback_body not in bodies:
        raise ValueError(
            f"Active MJCF {skeleton.fk.mjcf_file} has neither site {site!r} "
            f"nor fallback body {fallback_body!r}"
        )
    return EndEffectorAnchor(
        name=name,
        source_type="body",
        source_name=fallback_body,
        parent_body=fallback_body,
        parent_body_index=_body_index(skeleton, fallback_body),
        local_pos=torch.zeros(3, dtype=torch.float32),
    )


def resolve_end_effector_anchors(skeleton) -> tuple[EndEffectorAnchor, ...]:
    """Resolve V10 end-effector anchors from the active MJCF only."""
    mjcf_file = Path(skeleton.fk.mjcf_file)
    bodies, sites, geoms_by_mesh, geoms_by_name = _mjcf_elements_by_body(
        mjcf_file)
    return (
        _resolve_geom_anchor(
            skeleton,
            name="left_hand",
            mesh="left_rubber_hand",
            expected_parent_body="left_wrist_yaw_link",
            geoms_by_mesh=geoms_by_mesh,
            geoms_by_name=geoms_by_name,
        ),
        _resolve_geom_anchor(
            skeleton,
            name="right_hand",
            mesh="right_rubber_hand",
            expected_parent_body="right_wrist_yaw_link",
            geoms_by_mesh=geoms_by_mesh,
            geoms_by_name=geoms_by_name,
        ),
        _resolve_site_or_body_anchor(
            skeleton,
            name="left_foot",
            site="left_foot",
            fallback_body="left_ankle_roll_link",
            bodies=bodies,
            sites=sites,
        ),
        _resolve_site_or_body_anchor(
            skeleton,
            name="right_foot",
            site="right_foot",
            fallback_body="right_ankle_roll_link",
            bodies=bodies,
            sites=sites,
        ),
    )


def end_effector_anchor_meta(skeleton) -> list[dict]:
    return [anchor.meta() for anchor in resolve_end_effector_anchors(skeleton)]


def extract_end_effector_positions(
    fk_result: dict,
    skeleton,
    anchors: tuple[EndEffectorAnchor, ...] | None = None,
) -> torch.Tensor:
    """Extract end-effector positions from an FK result in canonical order.

    Returns shape [..., 4, 3], preserving all leading FK batch/time dimensions.
    """
    if anchors is None:
        anchors = resolve_end_effector_anchors(skeleton)
    global_translation = fk_result["global_translation"]
    global_rotation = fk_result["global_rotation_mat"]
    points = []
    for anchor in anchors:
        idx = int(anchor.parent_body_index)
        parent_pos = global_translation[..., idx, :]
        parent_rot = global_rotation[..., idx, :, :]
        local_pos = anchor.local_pos.to(
            device=parent_pos.device, dtype=parent_pos.dtype)
        point = parent_pos + torch.matmul(
            parent_rot, local_pos.reshape(3, 1)).squeeze(-1)
        points.append(point)
    return torch.stack(points, dim=-2)
