"""Experimental RGB-timed handoff; original support and strict hold gates retained."""
import torch
from coherent_control import CoherentCaptureEnv
from bridge import phase

class FluidCaptureEnv(CoherentCaptureEnv):
    def collection_ready(self, v, feet, visible):
        ready = super().collection_ready(v, feet, visible)
        # Only measured RGB arrival and existing actor output affect timing.
        urgency = ((3.5-self.arrival_time)/2.).clamp(0, 1)
        urgency = torch.where(self.prediction_stable & self.lane_ready, urgency, 0.)
        self.handoff_duration = (1.2-.25*urgency+.05*self.preparation_action[:,0]).clamp(.90,1.25)
        self.nominal_descent_duration = 3.0
        return ready

    def compose_motor_targets(self, gait, collection, residual):
        target = super().compose_motor_targets(gait, collection, residual)
        # Prepare arms during the final braking step without changing leg targets.
        e = self.env
        robot = e.scene['robot'].data
        speed = robot.root_link_lin_vel_w[:,:2].norm(dim=1)
        active = (e._bridge_stage==1) & self.prediction_stable & (speed<.12)
        opened = self.open_table.at(torch.full_like(self.arrival_time,.08))
        arm_target = (opened[:,3:25]+opened[:,25:47])[:,self.path_order]
        arm = torch.tensor(['Shoulder' in n or 'Elbow' in n for n in self.action_names],device=self.device)
        weight = (.15*(self.soft_support_time/.12).clamp(0,1))[:,None]
        delta = (arm_target-target).clamp(-.12,.12)*weight
        return target + torch.where(active[:,None]&arm[None],delta,0.)
