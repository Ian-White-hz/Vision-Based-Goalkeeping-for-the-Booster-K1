"""Level the stance feet of K1 goalkeeper clips so joint-only replay is balanced.

The AMP discriminator sees joint positions only, and the policy stands on flat
feet. In the published G1 clips (and their K1 retargets) the supporting foot is
pitched toes-down by ~4..8 deg, so the same joint angles on a flat foot rotate
the whole body backward and put the static COM behind the heel.

Two steps, ankle pitch only:
1. Level each low foot in the clip's own world frame (root and other joints
   unchanged).
2. Where the flat-foot COM is still closer than BACK_MARGIN to the rearmost
   support point (deep step crouches), lean the body forward about the ankles
   and move the root so the stance foot stays in place.
Lateral balance is untouched: the clips are one-legged dives and lunges whose
COM is beside the stance foot by design.
"""

from __future__ import annotations

import mujoco
import numpy as np

from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER, K1_XML

SIDES = ("Left", "Right")
# Sole height band (m): full correction below LOW, none above HIGH (swing/flight).
LOW, HIGH = 0.02, 0.08
# Largest ankle-pitch change (rad); steeper feet are toe-off or flight poses.
MAX_CORRECTION = 0.26
# Minimum distance (m) of the flat-foot COM ahead of the rearmost support point.
# Default stance is ~.10, the trained policy's stance ~.05.
BACK_MARGIN = 0.05
# Largest forward lean about the ankles (rad).
MAX_LEAN = 0.2
# The other foot counts as support within this height (m) of the stance sole.
SUPPORT_BAND = 0.03
# Temporal Gaussian (frames) on the correction to avoid ankle jitter.
SMOOTH_SIGMA = 1.5


def k1_model() -> mujoco.MjModel:
    return mujoco.MjSpec.from_file(str(K1_XML)).compile()


class _Clip:
    def __init__(self, model: mujoco.MjModel):
        self.m, self.d = model, mujoco.MjData(model)
        self.qadr = np.array(
            [model.jnt_qposadr[model.joint(n).id] for n in K1_JOINT_ORDER]
        )
        self.feet = [model.body(f"{s.lower()}_foot_link").id for s in SIDES]
        self.geoms = [model.geom(f"{s.lower()}_foot_collision").id for s in SIDES]
        self.ankle = [K1_JOINT_ORDER.index(f"{s}_Ankle_Pitch") for s in SIDES]
        self.limits = [
            model.jnt_range[model.joint(f"{s}_Ankle_Pitch").id] for s in SIDES
        ]

    def pose(self, root_pos, root_rot_xyzw, dof):
        self.d.qpos[:3] = root_pos
        self.d.qpos[3:7] = np.asarray(root_rot_xyzw)[[3, 0, 1, 2]]
        self.d.qpos[self.qadr] = dof
        mujoco.mj_forward(self.m, self.d)

    def foot_pitch(self, side: int) -> float:
        """World pitch of the foot x axis; positive is toes down."""
        return float(np.arcsin(-self.d.xmat[self.feet[side]].reshape(3, 3)[2, 0]))

    def sole(self, side: int) -> np.ndarray:
        g = self.geoms[side]
        mesh = self.m.geom_dataid[g]
        start, count = self.m.mesh_vertadr[mesh], self.m.mesh_vertnum[mesh]
        vertices = self.m.mesh_vert[start : start + count]
        return vertices @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]

    def sole_min_z(self, side: int) -> float:
        return float(self.sole(side)[:, 2].min())

    def stance_side(self) -> int | None:
        heights = [self.sole_min_z(0), self.sole_min_z(1)]
        side = int(np.argmin(heights))
        return side if heights[side] <= LOW else None

    def back_margin(self, side: int) -> float:
        """Flat-foot COM x minus the rearmost support x, in the stance-foot frame.

        Independent of the root pose: with that foot flat, its frame is the ground.
        """
        b = self.feet[side]
        rot, pos = self.d.xmat[b].reshape(3, 3), self.d.xpos[b]
        own = (self.sole(side) - pos) @ rot
        rear = own[:, 0].min()
        other = (self.sole(1 - side) - pos) @ rot
        low = other[other[:, 2] - own[:, 2].min() < SUPPORT_BAND]
        if len(low):
            rear = min(rear, low[:, 0].min())
        return float(((self.d.subtree_com[1] - pos) @ rot)[0] - rear)

    def com_in_foot(self, side: int) -> np.ndarray:
        b = self.feet[side]
        rot = self.d.xmat[b].reshape(3, 3)
        return rot.T @ (self.d.subtree_com[1] - self.d.xpos[b])


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(x.shape[1])],
        axis=1,
    )


def flatten_stance_feet(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    model: mujoco.MjModel | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return corrected (dof_pos, root_pos, root_rot) and before/after statistics."""
    clip = _Clip(model or k1_model())
    frames = len(dof_pos)
    delta = np.zeros((frames, 2))
    for t in range(frames):
        clip.pose(root_pos[t], root_rot[t], dof_pos[t])
        for side in range(2):
            weight = np.clip((HIGH - clip.sole_min_z(side)) / (HIGH - LOW), 0.0, 1.0)
            if weight == 0:
                continue
            # Newton on the ankle pitch joint; the sign and the roll coupling
            # come from the numeric derivative.
            q = dof_pos[t].copy()
            total = 0.0
            for _ in range(4):
                pitch = clip.foot_pitch(side)
                q[clip.ankle[side]] += 1e-4
                clip.pose(root_pos[t], root_rot[t], q)
                slope = (clip.foot_pitch(side) - pitch) / 1e-4
                step = -pitch / slope if abs(slope) > 1e-3 else 0.0
                q[clip.ankle[side]] += step - 1e-4
                total += step
                clip.pose(root_pos[t], root_rot[t], q)
            delta[t, side] = weight * np.clip(total, -MAX_CORRECTION, MAX_CORRECTION)
    delta = _smooth(delta, SMOOTH_SIGMA)
    level = np.array(dof_pos, dtype=np.float64, copy=True)
    for side in range(2):
        low, high = clip.limits[side]
        level[:, clip.ankle[side]] = np.clip(
            level[:, clip.ankle[side]] + delta[:, side], low, high
        )

    # Step 2: forward lean about the ankles where the COM is still too far back.
    lean = np.zeros((frames, 1))
    stance = np.full(frames, -1)
    for t in range(frames):
        clip.pose(root_pos[t], root_rot[t], level[t])
        side = clip.stance_side()
        if side is None:
            continue
        stance[t] = side
        q = level[t].copy()
        total = 0.0
        for _ in range(4):
            margin = clip.back_margin(side)
            if margin >= BACK_MARGIN and total == 0.0:
                break
            q[clip.ankle] += 1e-4
            clip.pose(root_pos[t], root_rot[t], q)
            slope = (clip.back_margin(side) - margin) / 1e-4
            step = (BACK_MARGIN - margin) / slope if abs(slope) > 1e-3 else 0.0
            q[clip.ankle] += step - 1e-4
            total += step
            clip.pose(root_pos[t], root_rot[t], q)
        lean[t, 0] = np.clip(total, -MAX_LEAN, MAX_LEAN)
    lean = _smooth(lean, SMOOTH_SIGMA)
    out = level.copy()
    for side in range(2):
        low, high = clip.limits[side]
        out[:, clip.ankle[side]] = np.clip(
            out[:, clip.ankle[side]] + lean[:, 0], low, high
        )
    new_root_pos = np.array(root_pos, dtype=np.float64, copy=True)
    new_root_rot = np.array(root_rot, dtype=np.float64, copy=True)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    for t in range(frames):
        if stance[t] < 0 or abs(lean[t, 0]) < 1e-6:
            continue
        # Keep the stance foot where it was; the body rotates about its ankle.
        b = clip.feet[stance[t]]
        clip.pose(root_pos[t], root_rot[t], level[t])
        foot_rot, foot_pos = clip.d.xmat[b].reshape(3, 3).copy(), clip.d.xpos[b].copy()
        clip.pose(np.zeros(3), identity, out[t])
        local_rot, local_pos = clip.d.xmat[b].reshape(3, 3), clip.d.xpos[b]
        rot = foot_rot @ local_rot.T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        new_root_rot[t] = quat[[1, 2, 3, 0]]
        new_root_pos[t] = foot_pos - rot @ local_pos
    report = {
        "before": clip_statistics(root_pos, root_rot, dof_pos, clip),
        "after": clip_statistics(new_root_pos, new_root_rot, out, clip),
        "ankle_pitch_change_rad": {
            "level_mean_abs": float(np.abs(delta).mean()),
            "level_max_abs": float(np.abs(delta).max()),
            "lean_mean_abs": float(np.abs(lean).mean()),
            "lean_max_abs": float(np.abs(lean).max()),
            "lean_frames": int((np.abs(lean[:, 0]) > 1e-3).sum()),
        },
    }
    return (
        out.astype(np.asarray(dof_pos).dtype),
        new_root_pos.astype(np.asarray(root_pos).dtype),
        new_root_rot.astype(np.asarray(root_rot).dtype),
        report,
    )


def clip_statistics(root_pos, root_rot, dof_pos, clip: _Clip | None = None) -> dict:
    """Stance-foot pitch and the flat-foot static COM margin (joint angles only)."""
    clip = clip or _Clip(k1_model())
    pitch, margin = [], []
    for t in range(len(dof_pos)):
        clip.pose(root_pos[t], root_rot[t], dof_pos[t])
        side = clip.stance_side()
        if side is None:
            continue
        pitch.append(np.degrees(clip.foot_pitch(side)))
        margin.append(clip.back_margin(side))
    if not margin:
        return {"stance_frames": 0}
    return {
        "stance_frames": len(margin),
        "stance_foot_pitch_deg_p10_p50_p90": np.percentile(pitch, [10, 50, 90])
        .round(2)
        .tolist(),
        "com_ahead_of_rear_support_m_p10_p50_p90": np.percentile(margin, [10, 50, 90])
        .round(3)
        .tolist(),
        "com_behind_support_fraction": float(np.mean(np.array(margin) < 0)),
    }


def sole_x_range(clip: _Clip | None = None) -> tuple[float, float]:
    """Sole extent along the foot x axis in the foot frame (heel, toe)."""
    clip = clip or _Clip(k1_model())
    clip.d.qpos[:] = 0
    clip.d.qpos[3] = 1
    mujoco.mj_forward(clip.m, clip.d)
    g, b = clip.geoms[0], clip.feet[0]
    mesh = clip.m.geom_dataid[g]
    start, count = clip.m.mesh_vertadr[mesh], clip.m.mesh_vertnum[mesh]
    world = (
        clip.m.mesh_vert[start : start + count] @ clip.d.geom_xmat[g].reshape(3, 3).T
        + clip.d.geom_xpos[g]
    )
    local = (world - clip.d.xpos[b]) @ clip.d.xmat[b].reshape(3, 3)
    return float(local[:, 0].min()), float(local[:, 0].max())
