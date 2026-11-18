"""Save credit by body part, behind-phase eereach and K1 region geometry."""

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from booster_mjlab.tasks.goalkeeper.source.config import env_cfg
from booster_mjlab.tasks.goalkeeper.source.settings import (
    REGIONS,
    GOAL_WIDTH,
    BALL_RADIUS,
    BEHIND_REACH_SCALE,
    ARM_SAVE_CREDIT,
)
from booster_mjlab.tasks.goalkeeper.source.state import state

torch.set_num_threads(1)
assert not ARM_SAVE_CREDIT and BEHIND_REACH_SCALE == 2.0
# Mid/low regions start outside the body; every region fits inside the posts.
for i in (0, 1, 4, 5):
    assert min(abs(REGIONS[i][0]), abs(REGIONS[i][1])) >= 0.15, REGIONS[i]
assert (
    max(abs(v) for r in REGIONS for v in r[:2]) + BALL_RADIUS + 0.03 <= GOAL_WIDTH / 2
)

e = ManagerBasedRlEnv(env_cfg(2, play=True, randomize=False), device="cpu")
try:
    # Default (upstream): any body part earns credit. arm_only: arms only.
    for seed, arm_only in enumerate((False, True)):
        e.reset(seed=seed)
        s = state(e)
        assert not s.arm_touch.any()  # reset clears the latch
        s.arm_save_credit = arm_only
        o = e.scene.env_origins
        hand = s.robot.data.body_link_pos_w[0, s.hand_indices[0]] - o[0]
        # env 0: ball straight at the left hand; env 1: at the trunk centreline.
        targets = torch.stack((hand, torch.tensor([0.0, 0.0, 0.62])))
        ball = torch.zeros((2, 13))
        ball[:, :3] = targets + o + torch.tensor([0.35, 0.0, 0.0])
        ball[:, 3] = 1.0
        ball[:, 7] = -5.0
        e.scene["ball"].write_root_state_to_sim(ball)
        s.ball_vel[:] = -5.0
        stop = torch.zeros(2)
        with torch.no_grad():
            for _ in range(8):
                e.step(torch.zeros((2, 22)))
                stop += s.rewards["stopball"]
        deflected = s.ball_states[:, 7] - s.ball_vel > 0.5
        assert deflected.all(), s.ball_states[:, 7]
        assert s.arm_touch.tolist() == [True, False], s.arm_touch
        trunk = 0.0 if arm_only else 1.0
        assert s.success_flag.tolist() == [1.0, trunk], s.success_flag
        assert stop[0] > 0 and (stop[1] > 0) == (not arm_only), stop
        assert (s.stop_flag == 1).all()
        # After the deflection eereach is 2x the plain reach term (upstream),
        # and 0 when the scale is 0.
        behind = s._behind()
        reach = 1 - 1 / (1 + torch.exp(-s.curriculumsigma * (s.dist - 0.2)))
        upright = 1 - s.projected_gravity[:, :2].square().sum(1).clip(0, 1)
        torch.testing.assert_close(
            s._reward_eereach()[behind], (2 * reach * upright)[behind]
        )
        s.behind_reach_scale = 0.0
        assert (s._reward_eereach()[behind] == 0).all()
        s.behind_reach_scale = BEHIND_REACH_SCALE
    print(
        "any-part deflection credited by default; arm_only credits the hand hit "
        "only; behind-phase eereach is 2x reach"
    )
finally:
    e.close()
