"""K1 elbow PD stability and K1-scaled trunk randomization for the source task."""

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from booster_mjlab.tasks.goalkeeper.source.config import env_cfg
from booster_mjlab.tasks.goalkeeper.source.settings import (
    ELBOW_PITCH_DAMPING,
    TRUNK_PAYLOAD_KG,
    TRUNK_COM_OFFSET_M,
)
from booster_mjlab.tasks.goalkeeper.source.state import state

torch.set_num_threads(1)
elbow = [K1_JOINT_ORDER.index(n) for n in ("Left_Elbow_Pitch", "Right_Elbow_Pitch")]
others = [i for i in range(22) if i not in elbow]

# Deterministic standing: elbow pitch must hold its target without saturating.
e = ManagerBasedRlEnv(env_cfg(4, play=True, randomize=False), device="cpu")
try:
    e.reset(seed=0)
    s = state(e)
    assert (s.d_gains[elbow] == ELBOW_PITCH_DAMPING).all()
    yaw = [K1_JOINT_ORDER.index(n) for n in ("Left_Elbow_Yaw", "Right_Elbow_Yaw")]
    assert (s.d_gains[yaw] == 1.0).all() and (s.p_gains[elbow] == 10.0).all()
    peak = torch.zeros(2)
    with torch.no_grad():
        for _ in range(50):
            e.step(torch.zeros((4, 22)))
            peak = torch.maximum(peak, (s.torques[:, elbow].abs() / 14.0).amax(0))
    assert peak.max() < 0.2, peak
    print(f"elbow pitch kd {ELBOW_PITCH_DAMPING}: peak |tau|/limit {peak.max():.3f}")
finally:
    e.close()

# Randomized startup: trunk payload and COM stay inside the K1 ranges.
nominal = ManagerBasedRlEnv(env_cfg(64, randomize=False), device="cpu")
rand = ManagerBasedRlEnv(env_cfg(64), device="cpu")
try:
    rob = nominal.scene["robot"]
    gid = rob.indexing.body_ids[rob.find_bodies("Trunk")[0][0]]
    m0 = nominal.sim.model.body_mass[:, gid]
    c0 = nominal.sim.model.body_ipos[:, gid]
    dm = rand.sim.model.body_mass[:, gid] - m0
    dc = rand.sim.model.body_ipos[:, gid] - c0
    assert dm.min() >= TRUNK_PAYLOAD_KG[0] - 1e-5, dm.min()
    assert dm.max() <= TRUNK_PAYLOAD_KG[1] + 1e-5, dm.max()
    assert dc.abs().max() <= TRUNK_COM_OFFSET_M + 1e-6, dc.abs().max()
    assert dm.max() - dm.min() > 2.0 and dc.abs().max() > 0.01
    print(
        f"trunk payload {dm.min():+.2f}..{dm.max():+.2f} kg, "
        f"COM offset max {dc.abs().max():.3f} m"
    )
finally:
    nominal.close()
    rand.close()
