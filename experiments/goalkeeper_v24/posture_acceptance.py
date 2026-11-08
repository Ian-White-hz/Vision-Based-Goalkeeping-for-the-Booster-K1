"""Pose-aware success only. Motor controllers and policy inputs stay unchanged."""
import math
import torch
from synchronized_task import StabilityOutcome, make_cfg as controller_cfg, measurements
from bridge import table, phase
from goalkeeper_sim.full_goal.settings import FullGoalSettings

RULE_ID='posture_grounded_hold_v2'

def hold_mask(m,relative_speed,ground,lean_error,clearance,overflow):
    # Existing physical thresholds are retained. The historical upright brake
    # bit is deliberately absent; m['stable'] measures the CURRENT support,
    # COM margin, foot drift and root velocities, and profile checks knees/lean.
    return (m['ready']&m['contacts'].all(1)&ground&m['stable']&m['profile']
        &~m['failed']&~m['goal']&(lean_error<math.radians(10))
        &(m['speed']<.08)&(relative_speed<.12)
        &(m['ball_pos'][:,2]>.10)&(m['ball_pos'][:,2]<.14)
        &(m['hand_ball_distance']<.15)&(clearance>0)&(overflow==0))

class PostureOutcome(StabilityOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.current_hold=torch.zeros_like(self.hold)
        self.current_peak=torch.zeros_like(self.hold)
        self.legacy_hold=torch.zeros_like(self.hold)
        self.legacy_peak=torch.zeros_like(self.hold)
        for name in ('pose_stable_now','posture_hold_success','legacy_success'):
            setattr(self,name,torch.zeros_like(self.success))
            self.last[name]=getattr(self,name).clone()
        for name in ('legacy_peak_hold_s','max_current_pose_hold_s'):
            self.timing[name]=torch.zeros(env.num_envs,device=env.device)
            self.timing_last[name]=self.timing[name].clone()

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        # Reuse truthful contact/timing/failure bookkeeping. Its old success
        # counters are overwritten below before the result reaches termination,
        # reward reporting or the outer trajectory audit.
        super().__call__(env)
        m=measurements(env);d=env.scene['robot'].data;ball=env.scene['ball'].data
        relative=(ball.root_link_lin_vel_w-d.body_link_lin_vel_w[:,env._collect_geometry[0]].mean(1)).norm(dim=1)
        ground=env.scene['ball_ground_contact'].data.found.flatten(1).any(1)
        lean_error=(m['tilt']-table(env).at(phase(env))[:,2]).abs()
        m=dict(m,hand_ball_distance=(ball.root_link_pos_w-m['hands'].mean(1)).norm(dim=1))
        clearance=m['ball_pos'][:,0]-.11-FullGoalSettings().goal_x
        physical=hold_mask(m,relative,ground,lean_error,clearance,env.sim.data.overflow.to(torch.long))
        eligibility=~self.failed&~env._bridge_unsafe&self.motion_seen&self.travelled&self.intercepted&self.descent_before_interception
        if not getattr(env,'_incoming_allow_stationary',False):eligibility&=env._bridge_distance>.08
        valid=physical&eligibility
        self.pose_stable_now[fresh]=(m['stable']&m['profile']&(lean_error<math.radians(10)))[fresh]
        self.current_hold[fresh]=torch.where(valid,self.current_hold+1,0)[fresh]
        self.current_peak[fresh]=torch.maximum(self.current_peak,self.current_hold)[fresh]
        legacy=m['valid']&~self.failed&self.motion_seen&self.travelled&self.intercepted&self.descent_before_interception
        self.legacy_hold[fresh]=torch.where(legacy,self.legacy_hold+1,0)[fresh]
        self.legacy_peak[fresh]=torch.maximum(self.legacy_peak,self.legacy_hold)[fresh]
        limit=math.ceil(2./env.step_dt-1e-9)
        self.posture_hold_success|=fresh&(self.current_hold>=limit)
        self.legacy_success|=fresh&(self.legacy_hold>=limit)
        self.hold.copy_(self.current_hold);self.peak_hold.copy_(self.current_peak)
        self.success.copy_(self.posture_hold_success&~self.failed)
        self.synchronized_success.copy_(self.success&self.synchronized_contact)
        self.body_then_close_success.copy_(self.success&self.body_then_close_contact)
        self.visual_strict_success.copy_(self.success&self.visual_closure_seen)
        self.timing['legacy_peak_hold_s']=self.legacy_peak*env.step_dt
        self.timing['max_current_pose_hold_s']=self.current_peak*env.step_dt
        return self.success&~self.failed

    def reset(self,env_ids=None):
        result=super().reset(env_ids)
        if hasattr(self,'current_hold'):
            ids=slice(None) if env_ids is None else env_ids
            for name in ('current_hold','current_peak','legacy_hold','legacy_peak'):
                getattr(self,name)[ids]=0
        return result

def make_cfg(num_envs=16):
    cfg=controller_cfg(num_envs)
    cfg.terminations['pose_success'].func=PostureOutcome
    return cfg
