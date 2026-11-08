"""Evaluation-only trajectory checks; values never enter controller observations."""
import torch
from goalkeeper_sim.full_goal.settings import FullGoalSettings


def audited_outcome(original, measure):
    class AuditedOutcome(original):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.audit_peak_z = torch.zeros(env.num_envs, device=env.device)
            self.audit_contact_peak_z = torch.zeros_like(self.audit_peak_z)
            self.audit_lift_frames = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            self.audit_hold_clearance = torch.full_like(self.audit_peak_z, float('inf'))
            self.audit_last = {}

        def __call__(self, env):
            fresh = self.stamp != env.common_step_counter
            success = super().__call__(env)
            m = measure(env)
            z = m['ball_pos'][:, 2]
            contact = m['contacts'].any(1)
            self.audit_peak_z[fresh] = torch.maximum(self.audit_peak_z, z)[fresh]
            selected = fresh & contact
            self.audit_contact_peak_z[selected] = torch.maximum(self.audit_contact_peak_z, z)[selected]
            self.audit_lift_frames += (fresh & contact & (z > .14)).long()
            holding = fresh & (self.hold > 0)
            clearance = m['ball_pos'][:, 0] - .11 - FullGoalSettings().goal_x
            self.audit_hold_clearance[holding] = torch.minimum(self.audit_hold_clearance, clearance)[holding]
            return success

        def reset(self, env_ids=None):
            if hasattr(self, 'audit_peak_z'):
                ids = slice(None) if env_ids is None else env_ids
                values = dict(peak_ball_height_m=self.audit_peak_z,
                    peak_ball_height_while_hand_contact_m=self.audit_contact_peak_z,
                    hand_contact_lift_frames=self.audit_lift_frames,
                    minimum_entire_ball_goal_clearance_during_hold_m=self.audit_hold_clearance)
                for name, value in values.items():
                    if name not in self.audit_last:
                        self.audit_last[name] = torch.zeros_like(value)
                    self.audit_last[name][ids] = value[ids]
                self.audit_peak_z[ids] = self.audit_contact_peak_z[ids] = 0
                self.audit_lift_frames[ids] = 0
                self.audit_hold_clearance[ids] = float('inf')
            return super().reset(env_ids)

    return AuditedOutcome
