"""Offline G1 goalkeeper retargeting using named URDF FK and bounded K1 IK.

The upstream clips contain 21 joint columns, not 29. Unmapped G1 waist/wrist
DOFs are zero, matching the upstream MotionLib. Unnamed 17-link tracks are
used only to cross-check FK; their inferred ordering is not an IK contract.
No Isaac Gym, GPU, network, or upstream Python imports are needed.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Initialize mjlab's task entry points before importing robot constants.
# Otherwise their EntityCfg import re-enters a partially initialized robots package.
import mjlab  # noqa: F401

from booster_mjlab.motion.motion_data import MotionFile
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER, K1_XML
from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    K1_PARALLEL_ACTUATED_JOINTS,
    K1_PARALLEL_JOINT_ORDER,
    K1_PARALLEL_XML,
)
from booster_mjlab.robots.booster_k1.parallel_retarget import retarget_dof_pos

CLIP_NAMES = ("lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep")


@dataclass(frozen=True)
class RetargetOptions:
    fps: float = 30.0
    max_nfev: int = 100
    position_weight: float = 1.0
    intermediate_weight: float = 0.25
    orientation_weight: float = 0.06
    pose_weight: float = 0.003
    velocity_weight: float = 0.01
    acceleration_weight: float = 0.005
    collision_weight: float = 5.0
    collision_margin: float = 0.004
    contact_height: float = 0.025
    contact_speed: float = 0.25
    ground_clearance: float = 0.002
    playback_speed: float = 1.0

    def validate(self) -> None:
        for name in ("fps", "max_nfev", "playback_speed"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class UrdfKinematics:
    """URDF FK with explicit names, including fixed joints and origin rotations."""

    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        links = {element.attrib["name"] for element in root.findall("link")}
        self.joints = []
        child_links = set()
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            axis = joint.find("axis")
            vector = (
                np.fromstring(axis.get("xyz", "1 0 0"), sep=" ")
                if axis is not None
                else np.array([1.0, 0.0, 0.0])
            )
            vector /= np.linalg.norm(vector)
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            transform[:3, 3] = xyz
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            child_links.add(child)
            self.joints.append(
                (
                    joint.attrib["name"],
                    joint.attrib["type"],
                    parent,
                    child,
                    transform,
                    vector,
                )
            )
        roots = links - child_links
        if roots != {"pelvis"}:
            raise ValueError(f"Expected G1 pelvis root, got {roots}")
        self.joint_names = {joint[0] for joint in self.joints}

    def forward(self, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
        transforms = {"pelvis": np.eye(4)}
        pending = list(self.joints)
        while pending:
            remaining = []
            for name, kind, parent, child, origin, axis in pending:
                if parent not in transforms:
                    remaining.append((name, kind, parent, child, origin, axis))
                    continue
                motion = np.eye(4)
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(
                        axis * joint_values.get(name, 0.0)
                    ).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis * joint_values.get(name, 0.0)
                elif kind != "fixed":
                    raise ValueError(f"Unsupported joint type {kind}")
                transforms[child] = transforms[parent] @ origin @ motion
            if len(remaining) == len(pending):
                raise ValueError("Disconnected or cyclic URDF")
            pending = remaining
        return transforms


def load_clip(
    path: Path, mapping_path: Path
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Restricted torch loading; reject malformed layouts rather than truncate."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a tensor dictionary")
    arrays = {}
    for name, value in raw.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{path}: {name} is not a tensor")
        arrays[name] = value.detach().numpy().astype(np.float64)
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}: nonfinite {name}")
    mapping = {}
    for line in mapping_path.read_text().splitlines():
        if line.strip():
            index, name = line.split()
            index = int(index)
            if index in mapping or name in mapping.values():
                raise ValueError("Duplicate joint mapping")
            mapping[index] = name
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Joint mapping indices must be contiguous from zero")
    names = [mapping[index] for index in range(len(mapping))]
    n = arrays.get("joint_position", np.empty((0,))).shape[0]
    for key, shape in {
        "joint_position": (n, len(names)),
        "base_position": (n, 3),
        "base_pose": (n, 4),
    }.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f"{path}: expected {key} shape {shape}")
    if n < 3:
        raise ValueError("At least three frames are required")
    norms = np.linalg.norm(arrays["base_pose"], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Source base_pose must contain unit xyzw quaternions")
    return arrays, names


def _indices(
    model: mujoco.MjModel, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.array([model.joint(name).id for name in names])
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids], model.jnt_range[ids]


def _sole_vertices(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Actual collision mesh vertices, not just foot-site heights."""
    points = []
    for side in ("left", "right"):
        geom = model.geom(f"{side}_foot_collision").id
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start : start + count]
        points.append(
            vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
    return np.concatenate(points)


def source_targets(
    arrays: dict[str, np.ndarray],
    names: list[str],
    fk: UrdfKinematics,
    model: mujoco.MjModel,
    options: RetargetOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Segment-scaled targets in K1's torso frame; preserve initial-heading motion."""
    unknown = set(names) - fk.joint_names
    if unknown:
        raise ValueError(f"Unknown source joints: {sorted(unknown)}")
    frames = [fk.forward(dict(zip(names, q))) for q in arrays["joint_position"]]
    # 得到G1的初始位姿，作为参考
    zero = fk.forward({})
    # 把K1的模型加载进来，并把根位置设为零
    data = mujoco.MjData(model)
    data.qpos[:3] = 0
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = 0
    mujoco.mj_kinematics(model, data)
    source_rotation = Rotation.from_quat(arrays["base_pose"]).as_matrix()
    torso_rotation = np.stack([frame["torso_link"][:3, :3] for frame in frames])
    world_torso = source_rotation @ torso_rotation
    heading = np.arctan2(world_torso[0, 1, 0], world_torso[0, 0, 0])
    heading_inverse = Rotation.from_euler("z", -heading).as_matrix()
    root_rotations = heading_inverse @ world_torso
    point_names = (
        "left_elbow",
        "left_hand",
        "right_elbow",
        "right_hand",
        "left_knee",
        "left_ankle",
        "right_knee",
        "right_ankle",
    )
    targets = np.zeros((len(frames), 8, 3))
    feet_rotation = np.zeros((len(frames), 2, 3, 3))
    ratios = {}
    for side_i, side in enumerate(("left", "right")):
        cap = side.title()
        shoulder = data.xpos[model.body(f"{cap}_Arm_1").id].copy()
        elbow = data.xpos[model.body(f"{side}_hand_link").id].copy()
        hand = data.site_xpos[model.site(f"{side}_hand").id].copy()
        hip = data.xpos[model.body(f"{cap}_Hip_Pitch").id].copy()
        knee = data.xpos[model.body(f"{cap}_Shank").id].copy()
        ankle = data.site_xpos[model.site(f"{side}_foot").id].copy()
        source_pairs = (
            (f"{side}_shoulder_pitch_link", f"{side}_elbow_link"),
            (f"{side}_elbow_link", f"{side}_rubber_hand"),
            (f"{side}_hip_pitch_link", f"{side}_knee_link"),
            (f"{side}_knee_link", f"{side}_ankle_roll_link"),
        )
        lengths = (
            np.linalg.norm(elbow - shoulder),
            np.linalg.norm(hand - elbow),
            np.linalg.norm(knee - hip),
            np.linalg.norm(ankle - knee),
        )
        # 算出K1比G1的比例，分别是上臂、前臂、大腿、小腿
        scales = [
            length / np.linalg.norm(zero[end][:3, 3] - zero[start][:3, 3])
            for length, (start, end) in zip(lengths, source_pairs)
        ]
        ratios[side] = dict(zip(("upper_arm", "forearm", "thigh", "shank"), scales))
        for t, frame in enumerate(frames):
            torso_inverse = frame["torso_link"][:3, :3].T
            # 用G1的位姿算出K1各个连杆的长度和方向
            vectors = [
                torso_inverse @ (frame[end][:3, 3] - frame[start][:3, 3]) * scale
                for (start, end), scale in zip(source_pairs, scales)
            ]
            arm_index, leg_index = side_i * 2, 4 + side_i * 2
            # 从K1的肩膀开始，算出肘部和手的位置
            targets[t, arm_index] = shoulder + vectors[0]
            targets[t, arm_index + 1] = targets[t, arm_index] + vectors[1]
            # G1 has a waist; K1 legs must compensate for torso vs pelvis rotation.
            hip_target = torso_inverse @ hip
            targets[t, leg_index] = hip_target + vectors[2]
            targets[t, leg_index + 1] = targets[t, leg_index] + vectors[3]
            # 脚掌目标朝向
            feet_rotation[t, side_i] = (
                torso_inverse @ frame[f"{side}_ankle_roll_link"][:3, :3]
            )
    trajectory_scale = float(
        np.mean([ratios[s][k] for s in ratios for k in ("thigh", "shank")])
    )
    root_pos = (
        (arrays["base_position"] - arrays["base_position"][0])
        @ heading_inverse.T
        * trajectory_scale
    )
    # Establish source floor from ankle minima. Relative flight height is retained.
    source_ankles = np.stack(
        [
            [
                source_rotation[t] @ frame[f"{side}_ankle_roll_link"][:3, 3]
                + arrays["base_position"][t]
                for side in ("left", "right")
            ]
            for t, frame in enumerate(frames)
        ]
    )
    # 估计站立参考高度，取5%分位数
    floor = float(np.percentile(source_ankles[:, :, 2].min(axis=1), 5))
    # 计算离地高度
    clearance = (
        np.maximum(0, source_ankles[:, :, 2].min(axis=1) - floor) * trajectory_scale
    )
    # 根据脚踝目标反推K1的根位置，使得脚踝离地高度为ground_clearance
    world_ankles = np.einsum("tij,tkj->tki", root_rotations, targets[:, [5, 7]])
    # 设置根位置的z坐标，使得脚踝离地高度为ground_clearance
    root_pos[:, 2] = 0.03 + clearance - world_ankles[:, :, 2].min(axis=1)
    # 计算脚踝速度和接触状态
    source_speed = np.linalg.norm(
        np.gradient(source_ankles, 1 / options.fps, axis=0)[:, :, :2], axis=2
    )
    # 计算接触状态，脚踝高度小于contact_height且速度小于contact_speed则认为接触
    contact = (
        (source_ankles[:, :, 2] - floor) * trajectory_scale < options.contact_height
    ) & (source_speed * trajectory_scale < options.contact_speed)
    validation = {}
    if "link_position" in arrays and arrays["link_position"].shape == (
        len(frames),
        17,
        3,
    ):
        # These six correspondences were verified against the published clip.
        for index, link in (
            (2, "left_knee_link"),
            (3, "left_ankle_roll_link"),
            (5, "right_knee_link"),
            (6, "right_ankle_roll_link"),
            (11, "left_elbow_link"),
            (15, "right_elbow_link"),
        ):
            positions = np.stack(
                [
                    source_rotation[t] @ frame[link][:3, 3] + arrays["base_position"][t]
                    for t, frame in enumerate(frames)
                ]
            )
            error = np.linalg.norm(
                positions - arrays["link_position"][:, index], axis=1
            )
            validation[link] = {
                "mean_m": float(error.mean()),
                "max_m": float(error.max()),
            }
        if max(item["max_m"] for item in validation.values()) > 0.03:
            raise ValueError(
                "Published keypoints disagree with G1 FK by >3 cm; check URDF, joint mapping, and quaternion convention"
            )
    meta = {
        "target_names": point_names,
        "segment_scales": ratios,
        "root_translation_scale": trajectory_scale,
        "source_floor_ankle_z_m": floor,
        "source_fk_check": validation,
        "missing_source_joint_policy": "zero, matching upstream MotionLib",
    }
    return root_pos, root_rotations, targets, feet_rotation, contact, meta


class K1Solver:
    """MuJoCo FK with analytic position/orientation Jacobians and joint bounds."""

    def __init__(self, model: mujoco.MjModel, options: RetargetOptions):
        self.model, self.options = model, options
        self.data = mujoco.MjData(model)
        self.qadr, self.dadr, limits = _indices(model, K1_JOINT_ORDER)
        self.lower, self.upper = limits[:, 0] + 1e-6, limits[:, 1] - 1e-6
        self.active = np.arange(2, 22)  # Hold head at zero; no source head DOFs.
        self.points = [
            ("body", f"{side}_hand_link")
            if name == "elbow"
            else ("site", f"{side}_hand")
            if name == "hand"
            else ("body", f"{side.title()}_Shank")
            if name == "knee"
            else ("site", f"{side}_foot")
            for side, name in (
                ("left", "elbow"),
                ("left", "hand"),
                ("right", "elbow"),
                ("right", "hand"),
                ("left", "knee"),
                ("left", "ankle"),
                ("right", "knee"),
                ("right", "ankle"),
            )
        ]
        self.weights = np.array(
            [
                options.intermediate_weight if i % 2 == 0 else options.position_weight
                for i in range(8)
            ]
        )
        self.foot_ids = [
            model.body(f"{side}_foot_link").id for side in ("left", "right")
        ]
        # Physical no-load limits from k1_constants; report again after conversion.
        self.speed_limits = np.array(
            [7.85, 7.85] + [33.51] * 8 + [14.66, 12.57, 17.59, 12.57, 17.59, 17.59] * 2
        )
        # Hand/arm sweeps are the main failure mode when scaling the G1 motion.
        geoms = [
            g for g in range(model.ngeom) if model.geom(g).name.endswith("collision")
        ]
        pairs = set()
        for side in ("left", "right"):
            arm = [
                g
                for g in geoms
                if model.geom(g).name.startswith(side)
                and ("hand" in model.geom(g).name or "arm" in model.geom(g).name)
            ]
            for a in arm:
                for b in geoms:
                    name = model.geom(b).name
                    if (
                        b == a
                        or (name.startswith(side) and ("hand" in name or "arm" in name))
                        or "head" in name
                    ):
                        continue
                    ba, bb = model.geom_bodyid[[a, b]]
                    if (
                        ba == bb
                        or model.body_parentid[ba] == bb
                        or model.body_parentid[bb] == ba
                    ):
                        continue
                    pairs.add(tuple(sorted((a, b))))
        self.collision_pairs = sorted(pairs)

    def collision_terms(self, *, jacobian: bool = False) -> np.ndarray:
        residual = np.zeros(len(self.collision_pairs))
        derivative = np.zeros((len(residual), len(self.active)))
        segment = np.zeros(6)
        jp1, jp2 = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for i, (a, b) in enumerate(self.collision_pairs):
            distance = mujoco.mj_geomDistance(
                self.model, self.data, a, b, 0.05, segment
            )
            if distance >= self.options.collision_margin:
                continue
            residual[i] = self.options.collision_weight * (
                self.options.collision_margin - distance
            )
            if jacobian:
                vector = segment[3:] - segment[:3]
                # Signed distance keeps the normal pointing out of penetration.
                normal = (
                    vector / distance
                    if abs(distance) > 1e-9
                    else vector / max(np.linalg.norm(vector), 1e-9)
                )
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    jp1,
                    None,
                    segment[:3],
                    int(self.model.geom_bodyid[a]),
                )
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    jp2,
                    None,
                    segment[3:],
                    int(self.model.geom_bodyid[b]),
                )
                derivative[i] = (
                    -self.options.collision_weight
                    * normal
                    @ (jp2 - jp1)[:, self.dadr[self.active]]
                )
        return derivative if jacobian else residual

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:3] = 0
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        points = np.stack(
            [
                self.data.xpos[self.model.body(name).id]
                if kind == "body"
                else self.data.site_xpos[self.model.site(name).id]
                for kind, name in self.points
            ]
        )
        rotations = self.data.xmat[self.foot_ids].reshape(2, 3, 3).copy()
        return points, rotations

    def solve(
        self,
        target: np.ndarray,
        foot_rotation: np.ndarray,
        initial: np.ndarray,
        previous: np.ndarray | None,
        previous2: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        opt = self.options
        # 创建上下界限
        lower, upper = self.lower.copy(), self.upper.copy()
        if previous is not None:
            delta = self.speed_limits / (opt.fps * opt.playback_speed) * 0.95
            lower = np.maximum(lower, previous - delta)
            upper = np.minimum(upper, previous + delta)
        qfixed = np.zeros(22)

        def evaluate(x, jacobian=False):
            q = qfixed.copy()
            q[self.active] = x
            points, rotations = self.forward(q)
            if not jacobian:
                residual = [
                    ((points - target) * self.weights[:, None]).ravel(),
                    (
                        (rotations[:, :, [0, 2]] - foot_rotation[:, :, [0, 2]])
                        * opt.orientation_weight
                    ).ravel(),
                    opt.pose_weight * (q - initial)[self.active],
                    self.collision_terms(),
                ]
                if previous is not None:
                    residual.append(opt.velocity_weight * (q - previous)[self.active])
                if previous2 is not None:
                    residual.append(
                        opt.acceleration_weight
                        * (q - 2 * previous + previous2)[self.active]
                    )
                return np.concatenate(residual)
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            blocks = []
            for weight, (kind, name) in zip(self.weights, self.points):
                if kind == "body":
                    mujoco.mj_jacBody(
                        self.model, self.data, jp, jr, self.model.body(name).id
                    )
                else:
                    mujoco.mj_jacSite(
                        self.model, self.data, jp, jr, self.model.site(name).id
                    )
                blocks.append(weight * jp[:, self.dadr[self.active]].copy())
            for body, rotation in zip(self.foot_ids, rotations):
                mujoco.mj_jacBody(self.model, self.data, jp, jr, body)
                angular = jr[:, self.dadr[self.active]].T
                # Residual flattening order is (xyz, axis), hence transpose.
                derivative = np.stack(
                    [np.cross(angular, rotation[:, axis]).T for axis in (0, 2)], axis=1
                )
                blocks.append(
                    opt.orientation_weight * derivative.reshape(6, len(self.active))
                )
            blocks.append(opt.pose_weight * np.eye(len(self.active)))
            blocks.append(self.collision_terms(jacobian=True))
            if previous is not None:
                blocks.append(opt.velocity_weight * np.eye(len(self.active)))
            if previous2 is not None:
                blocks.append(opt.acceleration_weight * np.eye(len(self.active)))
            return np.vstack(blocks)

        start = previous if previous is not None else initial
        result = least_squares(
            evaluate,
            np.clip(start[self.active], lower[self.active], upper[self.active]),
            jac=lambda x: evaluate(x, True),
            bounds=(lower[self.active], upper[self.active]),
            max_nfev=opt.max_nfev,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        qfixed[self.active] = result.x
        return qfixed, bool(result.success)


def _initial_pose(arrays: dict, names: list[str], solver: K1Solver) -> np.ndarray:
    n = len(arrays["joint_position"])
    out = np.zeros((n, 22))
    mapping = {name: i for i, name in enumerate(names)}
    for i, target in enumerate(K1_JOINT_ORDER):
        if target.startswith("Head"):
            continue
        side, part = target.split("_", 1)
        source = f"{side.lower()}_{part.lower()}_joint"
        if part == "Knee_Pitch":
            source = f"{side.lower()}_knee_joint"
        elif part == "Elbow_Pitch":
            source = f"{side.lower()}_elbow_joint"
        elif part == "Elbow_Yaw":
            out[:, i] = -0.4 if side == "Left" else 0.4
            continue  # G1 shoulder yaw is not K1 elbow yaw.
        if source in mapping:
            out[:, i] = arrays["joint_position"][:, mapping[source]]
        if part == "Shoulder_Roll":
            # K1's zero arms extend sideways; G1's zero arms hang down.
            out[:, i] += -np.pi / 2 if side == "Left" else np.pi / 2
    return np.clip(out, solver.lower, solver.upper)


def retarget_clip(
    path: Path, mapping: Path, urdf: Path, options: RetargetOptions
) -> tuple[MotionFile, MotionFile, dict, dict]:
    options.validate()
    arrays, names = load_clip(path, mapping)
    fk = UrdfKinematics(urdf)
    model = mujoco.MjModel.from_xml_path(str(K1_XML))
    root, rotations, targets, foot_rotations, contact, meta = source_targets(
        arrays, names, fk, model, options
    )
    solver = K1Solver(model, options)
    initial = _initial_pose(arrays, names, solver)
    joints = np.zeros_like(initial)
    converged = []
    initial_errors, final_errors = [], []
    # Contact anchors live in world space; do not re-anchor at every frame.
    anchors = [None, None]
    for t in range(len(joints)):
        frame_target = targets[t].copy()
        for side_i, index in enumerate((5, 7)):
            if contact[t, side_i]:
                if anchors[side_i] is None:
                    anchors[side_i] = root[t] + rotations[t] @ frame_target[index]
                frame_target[index] = rotations[t].T @ (anchors[side_i] - root[t])
            else:
                anchors[side_i] = None
        initial_points, _ = solver.forward(initial[t])
        q, ok = solver.solve(
            frame_target,
            foot_rotations[t],
            initial[t],
            joints[t - 1] if t else None,
            joints[t - 2] if t > 1 else None,
        )
        joints[t] = q
        points, _ = solver.forward(q)
        initial_errors.append(np.linalg.norm(initial_points - frame_target, axis=1))
        final_errors.append(np.linalg.norm(points - frame_target, axis=1))
        converged.append(ok)
    # A forward kinematic floor correction retains source airborne clearance.
    data = mujoco.MjData(model)
    qadr, _, _ = _indices(model, K1_JOINT_ORDER)
    correction = np.zeros(len(joints))
    for t, q in enumerate(joints):
        data.qpos[:3], data.qpos[3:7] = (
            root[t],
            Rotation.from_matrix(rotations[t]).as_quat()[[3, 0, 1, 2]],
        )
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        z = _sole_vertices(model, data)[:, 2].min()
        correction[t] = max(0, options.ground_clearance - z)
    root[:, 2] += correction
    serial = MotionFile(
        options.fps * options.playback_speed,
        root.astype(np.float32),
        Rotation.from_matrix(rotations).as_quat().astype(np.float32),
        joints.astype(np.float32),
    )
    parallel_joints, stats = retarget_dof_pos(joints)
    parallel = MotionFile(
        serial.fps,
        root.copy().astype(np.float32),
        np.asarray(serial.root_rot).copy(),
        parallel_joints.astype(np.float32),
    )
    initial_errors, final_errors = np.asarray(initial_errors), np.asarray(final_errors)
    report = {
        "clip": path.stem,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "frames": len(joints),
        "source_fps_assumption": options.fps,
        "output_fps": serial.fps,
        "duration_s": (len(joints) - 1) / serial.fps,
        "source_joint_count": len(names),
        **meta,
        "solver_converged_frames": sum(converged),
        "mean_target_error_m": dict(
            zip(meta["target_names"], final_errors.mean(axis=0).tolist())
        ),
        "max_target_error_m": dict(
            zip(meta["target_names"], final_errors.max(axis=0).tolist())
        ),
        "initial_mean_target_error_m": float(initial_errors.mean()),
        "optimized_mean_target_error_m": float(final_errors.mean()),
        "floor_lift_max_m": float(correction.max()),
        "parallel_adjusted_frames": stats,
        "serial": validate_motion(serial, parallel=False, contact=contact),
        "parallel": validate_motion(parallel, parallel=True, contact=contact),
    }
    issues = []
    for variant in ("serial", "parallel"):
        quality = report[variant]
        if not quality["all_finite"]:
            issues.append(f"{variant}: nonfinite motion values")
        if quality["self_collision_frames"]:
            issues.append(
                f"{variant}: {quality['self_collision_frames']} self-collision frames"
            )
        if quality["min_foot_mesh_z_m"] < -0.001:
            issues.append(f"{variant}: foot penetration")
        if quality["speed_limit_exceeding_joints"]:
            issues.append(f"{variant}: motor speed limits exceeded")
        if quality["max_joint_limit_violation_rad"] > 1e-5:
            issues.append(f"{variant}: joint limits exceeded")
        if quality["nonfoot_ground_contact_frames"]:
            issues.append(f"{variant}: non-foot ground penetration")
        if (
            quality["stance_foot_speed_max_m_s"] is not None
            and quality["stance_foot_speed_max_m_s"] > 0.15
        ):
            issues.append(f"{variant}: inferred stance foot speed exceeds 0.15 m/s")
    if sum(stats.values()):
        issues.append(
            "Parallel ankle projection changed the reference; inspect tracking error and foot contacts"
        )
    if (
        max(report["max_target_error_m"][name] for name in ("left_hand", "right_hand"))
        > 0.1
    ):
        issues.append("Hand target error exceeds 10 cm")
    if not all(converged):
        issues.append("Some IK frames reached the evaluation budget")
    report["acceptance"] = {
        "status": "needs_review" if issues else "kinematic_checks_passed",
        "issues": issues,
        "dynamic_feasibility": "not tested: FK playback is not torque-controlled tracking or hardware validation",
    }
    diagnostics = {
        "targets_local": targets,
        "foot_rotations_local": foot_rotations,
        "contact_mask": contact,
        "initial_error_m": initial_errors,
        "final_error_m": final_errors,
    }
    return serial, parallel, report, diagnostics


def validate_motion(motion: MotionFile, *, parallel: bool, contact: np.ndarray) -> dict:
    spec = mujoco.MjSpec.from_file(str(K1_PARALLEL_XML if parallel else K1_XML))
    spec.worldbody.add_geom(
        name="validation_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    names = K1_PARALLEL_JOINT_ORDER if parallel else K1_JOINT_ORDER
    qadr, _, limits = _indices(model, names)
    foot_points = []
    foot_minima = []
    knee_minima = []
    collision_frames = 0
    pairs = {}
    nonfoot_ground_frames = 0
    equality_residual = 0.0
    for t in range(motion.num_frames):
        data.qpos[:3] = motion.root_pos[t]
        data.qpos[3:7] = np.asarray(motion.root_rot[t])[[3, 0, 1, 2]]
        data.qpos[qadr] = motion.dof_pos[t]
        mujoco.mj_forward(model, data)
        foot_points.append(
            data.site_xpos[
                [model.site(f"{side}_foot_sole").id for side in ("left", "right")]
            ].copy()
        )
        foot_minima.append(float(_sole_vertices(model, data)[:, 2].min()))
        knee_minima.append(
            float(
                data.xpos[
                    [model.body(f"{side}_Shank").id for side in ("Left", "Right")], 2
                ].min()
            )
        )
        collided = False
        ground_penetrated = False
        for c in data.contact[: data.ncon]:
            if c.dist < -0.001:
                geom_names = [model.geom(int(g)).name for g in c.geom]
                if "validation_ground" in geom_names:
                    other = next(
                        name for name in geom_names if name != "validation_ground"
                    )
                    ground_penetrated |= other not in (
                        "left_foot_collision",
                        "right_foot_collision",
                    )
                    continue
                key = " / ".join(sorted(geom_names))
                pairs[key] = pairs.get(key, 0) + 1
                collided = True
        collision_frames += int(collided)
        nonfoot_ground_frames += int(ground_penetrated)
        if parallel:
            eq = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            if np.any(eq):
                equality_residual = max(
                    equality_residual,
                    float(np.abs(data.efc_pos[: data.nefc][eq]).max()),
                )
    limited = model.jnt_limited[[model.joint(name).id for name in names]].astype(bool)
    violation = np.maximum(
        limits[:, 0] - motion.dof_pos, motion.dof_pos - limits[:, 1]
    )[:, limited]
    velocity = np.gradient(motion.dof_pos.astype(np.float64), motion.dt, axis=0)
    acceleration = np.gradient(velocity, motion.dt, axis=0)
    actuated = K1_PARALLEL_ACTUATED_JOINTS if parallel else K1_JOINT_ORDER
    speeds = {
        name: float(np.abs(velocity[:, names.index(name)]).max()) for name in actuated
    }
    solver = K1Solver(mujoco.MjModel.from_xml_path(str(K1_XML)), RetargetOptions())
    speed_limits = dict(zip(K1_JOINT_ORDER, solver.speed_limits))
    excessive = [
        name
        for name, speed in speeds.items()
        if speed > speed_limits.get(name, 17.59) + 1e-3
    ]
    foot_speed = (
        np.linalg.norm(np.diff(foot_points, axis=0)[:, :, :2], axis=2) / motion.dt
    )
    stance = contact[1:] & contact[:-1]
    return {
        "all_finite": bool(
            np.isfinite(motion.dof_pos).all()
            and np.isfinite(motion.root_pos).all()
            and np.isfinite(motion.root_rot).all()
        ),
        "max_joint_limit_violation_rad": max(0.0, float(violation.max())),
        "min_foot_mesh_z_m": min(foot_minima),
        "min_knee_z_m": min(knee_minima),
        "self_collision_frames": collision_frames,
        "nonfoot_ground_contact_frames": nonfoot_ground_frames,
        "self_collision_pairs": pairs,
        "max_motor_speed_rad_s": max(speeds.values()),
        "motor_peak_speeds_rad_s": speeds,
        "speed_limit_exceeding_joints": excessive,
        "max_joint_acceleration_rad_s2": float(np.abs(acceleration).max()),
        "stance_foot_speed_mean_m_s": float(foot_speed[stance].mean())
        if stance.any()
        else None,
        "stance_foot_speed_max_m_s": float(foot_speed[stance].max())
        if stance.any()
        else None,
        "contact_frames_per_foot": contact.sum(axis=0).tolist(),
        "max_equality_residual_m": equality_residual if parallel else None,
    }
