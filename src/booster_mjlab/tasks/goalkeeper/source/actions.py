"""K1 PD targets with source action scale, startup hold, and substep delay."""

from dataclasses import dataclass
import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from .state import state
from mjlab.actuator import BuiltinMotorActuatorCfg
from booster_mjlab.robots.booster_k1.actuators import ActuatorConfig


@dataclass(kw_only=True)
class SourceMotorCfg(BuiltinMotorActuatorCfg):
    stiffness: float
    damping: float
    motor: ActuatorConfig


@dataclass(kw_only=True)
class SourceActionCfg(JointPositionActionCfg):
    def build(self, env):
        return SourceAction(self, env)


class SourceAction(JointPositionAction):
    def process_actions(self, actions):
        self.previous = self._raw_actions.clone()
        super().process_actions(actions.clamp(-100, 100))
        self.substep = 0
        self.delay = torch.randint(0, 4, (self.num_envs, 1), device=self.device)
        self.injection = torch.empty_like(actions).uniform_(-0.01, 0.01)
        for i, name in enumerate(self.target_names):
            if name in ("Left_Shoulder_Roll", "Right_Shoulder_Roll"):
                self.injection[:, i] = 0.0

    def apply_actions(self):
        s = state(self._env)
        action = (
            torch.where(self.substep >= self.delay, self._raw_actions, self.previous)
            if s.settings.randomize
            else self._raw_actions
        )
        target = self._offset + action * 0.25
        target = torch.where(
            (s.catchstep > s.startstep)[:, None], s.init_dof_pos, target
        )
        q = self._entity.data.joint_pos[:, self._target_ids]
        dq = self._entity.data.joint_vel[:, self._target_ids]
        torque = s.p_gains * s.kp_factor * (target - q) - s.d_gains * s.kd_factor * dq
        if s.settings.randomize:
            torque += s.offset + self.injection * s.torque_limits
        torque = torque.clamp(-s.torque_limits, s.torque_limits)
        self._entity.set_joint_effort_target(torque, joint_ids=self._target_ids)
        self.substep += 1
