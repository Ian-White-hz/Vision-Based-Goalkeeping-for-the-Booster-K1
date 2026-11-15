"""K1 rescaling of upstream ball-phase thresholds.

Upstream hardcodes thresholds sized for G1 shots (start x=3..5 m, vx ~ -3..-12
m/s): far phase while ball x > 1.5 m, deflection when vx rises > 2 m/s, and
"behind" once ball x < 0. K1 shots start at x=1.5..2.5 m with vx ~ -1.7..-3.6
m/s, so the far phase almost vanished and a deflection had to nearly stop the
ball. Formulas are unchanged; only the thresholds are attributes. Setting
phase1_x=1.5, deflect_ratio=0, deflect_min=2, behind_x=0, arm_save_credit=False
and behind_reach_scale=2 restores upstream exactly (used by the parity test).
"""

import torch
from .vendor.rewards import SourceRewards
from .settings import (
    PHASE1_X,
    DEFLECT_RATIO,
    DEFLECT_MIN,
    BEHIND_X,
    ARM_SAVE_CREDIT,
    BEHIND_REACH_SCALE,
)


class K1Rewards(SourceRewards):
    phase1_x = PHASE1_X
    deflect_ratio = DEFLECT_RATIO
    deflect_min = DEFLECT_MIN
    behind_x = BEHIND_X
    arm_save_credit = ARM_SAVE_CREDIT
    behind_reach_scale = BEHIND_REACH_SCALE

    def _ball_x(self):
        return self.ball_states[:, 0] - self.env_origins[:, 0]

    def _deflected(self):
        """Ball x velocity rose by a speed-relative amount since launch."""
        threshold = (self.deflect_ratio * self.ball_vel.abs()).clamp(
            min=self.deflect_min
        )
        return self.ball_states[:, 7] - self.ball_vel > threshold

    def _behind(self):
        return (self._ball_x() < self.behind_x) | self._deflected()

    def _reward_eereach(self):
        x = self._ball_x()
        deflected = self._deflected()
        phase1 = (x > self.phase1_x) & ~deflected
        end_target_local = self.end_target - self.torso_pos
        asidegoal = torch.clip(end_target_local[:, 1], -1.0, 1.0)
        asidegoal[torch.abs(asidegoal) < 0.3] = 0.0
        verticalgoal = torch.clip(
            self.torso_pos[:, 2] - torch.clip(self.end_target[:, 2], 0.3, 1.2),
            0.0,
            1.0,
        )
        phase1_rew = 1.0 - (verticalgoal + torch.abs(asidegoal)) / 2.0

        jump_scale = 3.0 + 3.0 * self.curriculumupdate
        body_vel = self.rigid_body_states[:, self.upper_body_index, 7:10]
        vy, vz = body_vel[:, 1], body_vel[:, 2]
        left = (self.end_regions == 0) | (self.end_regions == 4)
        right = (self.end_regions == 1) | (self.end_regions == 5)
        jump = (self.end_regions == 2) | (self.end_regions == 3)
        vel_sigma = torch.zeros_like(x)
        vel_sigma = torch.where(left, 1 + 3.0 * vy.clip(0.0, 3.0), vel_sigma)
        vel_sigma = torch.where(right, 1 - 3.0 * vy.clip(-3.0, 0.0), vel_sigma)
        vel_sigma = torch.where(jump, 1 + jump_scale * vz.clip(0.0, 3.0), vel_sigma)
        vel_sigma = torch.where(self._behind(), self.behind_reach_scale, vel_sigma)

        taskrew = 1 - 1 / (
            1
            + torch.exp(-self.curriculumsigma * (self.dist - self.cfg.rewards.reach_th))
        )
        taskrew = torch.where(phase1, phase1_rew, taskrew * vel_sigma)
        upright = 1 - torch.clip(
            torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1), 0.0, 1.0
        )
        return taskrew * upright

    def _reward_stopball(self):
        x = self._ball_x()
        changevel = self._deflected() & (x > self.behind_x)
        stopped = changevel | (x < self.behind_x)
        credited = changevel
        if self.arm_save_credit:
            # A deflection by trunk, head or legs still ends the save window.
            credited = changevel & self.arm_touch
        rew_stop = 1.0 * (self.stop_flag == 0) * credited
        self.success_flag[(self.stop_flag == 0) & credited] = 1.0
        self.stop_flag[stopped] = 1.0
        return rew_stop

    def _reward_postorientation(self):
        g = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(g * -3) * self._behind()

    def _reward_postangvel(self):
        w = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return torch.exp(w * -3) * self._behind()

    def _reward_postlinvel(self):
        v = torch.sum(torch.square(self.base_lin_vel[:, 0:1]), dim=1)
        return torch.exp(v * -3) * self._behind()

    def _reward_postupperdofpos(self):
        j = self.upper_body_joint_indices
        mse = torch.sum(
            torch.square(self.dof_pos[:, j] - self.default_dof_pos[:, j]), dim=-1
        )
        return torch.exp(mse * -1) * self._behind()
