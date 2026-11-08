"""RGB arrival scheduling; measured-stop and strict outcome gates are unchanged."""
import torch
from random_capture_task import RandomCaptureEnv
from bridge import HANDOFF_SETTLE, DURATION


class TimedCaptureEnv(RandomCaptureEnv):
    handoff_duration = 1.0
    nominal_descent_duration = 3.0
    deadline_enabled = False
    impact_enabled = False

    def __init__(self,env):
        super().__init__(env)
        self.selected_handoff=torch.full((self.num_envs,),HANDOFF_SETTLE,device=self.device)
        self.handoff_selected=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)

    def collection_ready(self,v,feet,visible):
        ready = super().collection_ready(v,feet,visible)
        if not self.deadline_enabled:
            return ready
        e = self.env
        blend_left = (HANDOFF_SETTLE-e._bridge_clock).clamp_min(0)*self.handoff_duration/HANDOFF_SETTLE
        # Schedule readiness from the measured motor time still needed. If the
        # ball is already intercepted, retain the original tactile recovery.
        due = self.arrival_time <= blend_left+self.nominal_descent_duration+.35
        return ready & (due | self.preparation_contact | e._incoming_ready)

    def descent_clock_increment(self):
        e = self.env
        clock = e._bridge_clock
        # A smooth shorter motor blend removes idle time after the measured stop.
        # Split the crossing timestep so no phase jump occurs at the boundary.
        to_boundary = (HANDOFF_SETTLE-clock).clamp_min(0)
        if self.impact_enabled:
            self.handoff_selected[e._bridge_stage<2]=False
            select=(e._bridge_stage==2)&~self.handoff_selected
            # Preserve the stable body-assisted receive when there is ample
            # time. Short arrivals need the hands to begin descending sooner,
            # rather than waiting out the full standing-to-reference blend.
            # Smooth interpolation avoids a discontinuous timing threshold.
            u=((self.arrival_time-2.8)/.7).clamp(0,1)
            u=u*u*(3-2*u)
            budget=1.+1.5*u
            self.selected_handoff[select]=budget[select]
            self.handoff_selected[select]=True
            duration=self.selected_handoff
        else:
            duration=self.handoff_duration
        blend_rate = HANDOFF_SETTLE/duration
        blend_time = torch.minimum(to_boundary/blend_rate, torch.full_like(clock,e.step_dt))
        descent_time = e.step_dt-blend_time
        rate = torch.full_like(clock,DURATION/self.nominal_descent_duration)
        if self.deadline_enabled:
            # Adapt reference speed only within a separately screened range.
            # Ball input is exclusively the RGB track. Never rush a late ball
            # beyond the maximum rate; late cases remain recorded failures.
            remaining = (DURATION-(clock-HANDOFF_SETTLE).clamp_min(0)).clamp_min(0)
            budget = (self.arrival_time-.35).clamp_min(.1)
            usable = self.lane_ready & self.vision.fit_ready & (self.arrival_time>0) & ~self.preparation_contact
            needed = remaining/budget
            rate = torch.where(usable,needed.clamp(DURATION/4.,DURATION/2.5),rate)
        remaining = (DURATION-(clock-HANDOFF_SETTLE).clamp_min(0)).clamp_min(0)
        traversing = torch.minimum(descent_time,remaining/rate)
        # Endpoint dwell uses real time, preserving the original post-descent
        # settling buffer as well as the real two-second hold requirement.
        return blend_time*blend_rate+traversing*rate+(descent_time-traversing)
