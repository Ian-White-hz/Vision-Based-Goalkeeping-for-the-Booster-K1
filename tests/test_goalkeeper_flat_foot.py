"""Flat-foot clip correction: only ankle pitch changes and the COM moves onto the feet."""

import pickle
from pathlib import Path

import numpy as np

from booster_mjlab.motion.flat_foot import (
    BACK_MARGIN,
    MAX_CORRECTION,
    MAX_LEAN,
    flatten_stance_feet,
    k1_model,
)
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER

SOURCE = Path("artifacts/goalkeeper_k1/serial")
OUTPUT = Path("artifacts/goalkeeper_k1/serial_flatfoot")
ankles = [K1_JOINT_ORDER.index(f"{s}_Ankle_Pitch") for s in ("Left", "Right")]
others = [i for i in range(len(K1_JOINT_ORDER)) if i not in ankles]
model = k1_model()
for name in ("lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"):
    with (SOURCE / f"{name}.pkl").open("rb") as file:
        source = pickle.load(file)
    dof, root_pos, root_rot, report = flatten_stance_feet(
        source["root_pos"], source["root_rot"], source["dof_pos"], model
    )
    assert dof.shape == source["dof_pos"].shape
    np.testing.assert_array_equal(dof[:, others], source["dof_pos"][:, others])
    change = np.abs(dof[:, ankles] - source["dof_pos"][:, ankles]).max()
    assert 0.05 < change <= MAX_CORRECTION + MAX_LEAN + 1e-6, change
    np.testing.assert_allclose(np.linalg.norm(root_rot, axis=1), 1.0, atol=1e-6)
    before, after = report["before"], report["after"]
    # Stance feet are level and the flat-foot COM is ahead of the support's rear edge.
    assert abs(after["stance_foot_pitch_deg_p10_p50_p90"][1]) < 0.5, after
    assert after["com_behind_support_fraction"] <= 0.05, after
    assert after["com_behind_support_fraction"] <= before["com_behind_support_fraction"]
    assert after["com_ahead_of_rear_support_m_p10_p50_p90"][1] >= BACK_MARGIN - 0.005
    # No faster than the original clip.
    speed = lambda q: np.abs(np.diff(q, axis=0)).max()  # noqa: E731
    assert speed(dof) <= speed(source["dof_pos"]) + 1e-6
    # The committed dataset is what the function produces.
    with (OUTPUT / f"{name}.pkl").open("rb") as file:
        saved = pickle.load(file)
    np.testing.assert_allclose(saved["dof_pos"], dof, atol=1e-6)
    assert saved["fps"] == source["fps"]
    print(
        f"{name}: behind support {before['com_behind_support_fraction']:.2f} -> "
        f"{after['com_behind_support_fraction']:.2f}, max ankle change {change:.2f} rad"
    )
print("flat-foot clips: ankle pitch only, level stance feet, COM ahead of rear support")
