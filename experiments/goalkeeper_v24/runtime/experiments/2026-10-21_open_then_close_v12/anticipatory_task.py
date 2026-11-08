"""Predictive preparation after a measured stop, before body contact.

Ball inputs to control are RGB estimates only. Physical truth is used below
only for reward/outcome diagnostics. Live state writes occur only in resets.
"""
import torch
from tensordict import TensorDict
from mjlab.managers import TerminationTermCfg
from coordinated_capture_task import CoordinatedCaptureEnv,raw_class,measurements
from rolling_task import IncomingOutcome
from bridge import phase,book,HANDOFF_SETTLE
from random_goal_task import make_cfg as range_cfg
from shot_curriculum import RANGES,ShotRange,specification

STAGE='frontal_preparation_v1'
RANGES[STAGE]=ShotRange(.020,.020,.30,.40,3.0,3.4)


class PreparationOutcome(IncomingOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        for name in ('descent_before_contact','ready_before_contact','intercepted',
                     'descent_before_interception','ready_before_interception'):
            value=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
            setattr(self,name,value); self.last[name]=value.clone()
        self.timing=dict(first_descent_s=torch.full((env.num_envs,),float('inf'),device=env.device),
            first_body_contact_s=torch.full((env.num_envs,),float('inf'),device=env.device),
            first_interception_s=torch.full((env.num_envs,),float('inf'),device=env.device),
            max_outgoing_x_speed_after_body_m_s=torch.zeros(env.num_envs,device=env.device),
            max_outgoing_x_speed_after_interception_m_s=torch.zeros(env.num_envs,device=env.device))
        for name in ('first_measured_stop_s','first_pose_ready_s','first_bilateral_s'):
            self.timing[name]=torch.full((env.num_envs,),float('inf'),device=env.device)
        self.timing_last={k:v.clone() for k,v in self.timing.items()}

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        body=env.scene['body_ball_contact'].data.found.flatten(1).any(1)
        contact=body|measurements(env)['contacts'].any(1)
        self.intercepted|=fresh&contact
        t=env.episode_length_buf*env.step_dt
        before_body=torch.isinf(self.timing['first_body_contact_s'])&~body
        descent=phase(env)>.001
        self.descent_before_contact|=fresh&descent&before_body
        self.ready_before_contact|=fresh&(phase(env)>.98)&before_body
        before_any=torch.isinf(self.timing['first_interception_s'])&~contact
        self.descent_before_interception|=fresh&descent&before_any
        self.ready_before_interception|=fresh&(phase(env)>.98)&before_any
        for name,selected in (('first_measured_stop_s',env._bridge_braked),
                              ('first_pose_ready_s',phase(env)>.98),
                              ('first_bilateral_s',measurements(env)['contacts'].all(1)),
                              ('first_descent_s',descent),('first_body_contact_s',body),
                              ('first_interception_s',contact)):
            ids=fresh&selected&torch.isinf(self.timing[name])
            self.timing[name][ids]=t[ids]
        after=fresh&~torch.isinf(self.timing['first_body_contact_s'])
        # Outcome/reward-only measurement; never used to choose a motor target.
        outgoing=env.scene['ball'].data.root_link_lin_vel_w[:,0].clamp_min(0)
        self.timing['max_outgoing_x_speed_after_body_m_s'][after]=torch.maximum(
            self.timing['max_outgoing_x_speed_after_body_m_s'],outgoing)[after]
        after_any=fresh&~torch.isinf(self.timing['first_interception_s'])
        self.timing['max_outgoing_x_speed_after_interception_m_s'][after_any]=torch.maximum(
            self.timing['max_outgoing_x_speed_after_interception_m_s'],outgoing)[after_any]
        return super().__call__(env)

    def reset(self,env_ids=None):
        if hasattr(self,'timing'):
            ids=slice(None) if env_ids is None else env_ids
            for name,value in self.timing.items():
                self.timing_last[name][ids]=value[ids]
                value[ids]=0. if name.startswith('max_') else float('inf')
        return super().reset(env_ids)


def make_cfg(num_envs=16):
    cfg=range_cfg(num_envs,stage=STAGE)
    cfg.terminations['pose_success']=TerminationTermCfg(func=PreparationOutcome)
    return cfg


class AnticipatoryEnv(CoordinatedCaptureEnv):
    contact_reference_fix=True
    def __init__(self,env):
        super().__init__(env)
        env._anticipatory_receive=True
        self.arrival_time=torch.full((self.num_envs,),12.,device=self.device)
        self.preparation_action=torch.zeros((self.num_envs,2),device=self.device)
        self.preparation_contact=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        # Train bounded arm/stance adjustments, preserving the existing teachers.
        self.bridge_amplitude=torch.tensor([.08 if 'Shoulder' in name or 'Elbow' in name else .015
            for name in self.action_names],device=self.device)

    def get_observations(self):
        base=super().get_observations()
        if not hasattr(self,'arrival_time'): return base
        e=self.env
        extra=torch.stack(((self.arrival_time/10).clamp(0,1),e._incoming_ready.float(),
                           e._incoming_touched.float()),1)
        actor=torch.cat((base['actor'],extra),1)
        return TensorDict({'actor':actor,'critic':torch.cat((actor,base['critic'][:,-6:]),1)},
            batch_size=[self.num_envs])

    def approach_lateral_command(self,error,velocity,actions):
        # First drill: already aligned frontal shots; no unnecessary sideways step.
        return torch.zeros(self.num_envs,device=self.device)

    def collection_ready(self,v,feet,visible):
        e=self.env
        speed=-self.lane_velocity[:,0]
        receive_x=feet[:,0]+.20+.015*self.preparation_action[:,1]
        tta=((v.point[:,0]-receive_x)/speed.clamp_min(.05)).clamp(-1.,12.)
        self.arrival_time=tta
        predicted_y=v.point[:,1]+self.lane_velocity[:,1]*tta.clamp_min(0)
        horizon=6.5+.6*self.preparation_action[:,0]
        preparing=(e._bridge_stage==2)&e._bridge_braked&visible&v.fit_ready&self.lane_ready
        preparing&=(speed>.08)&(tta>0)&(tta<horizon)&((predicted_y-feet[:,1]).abs()<.05)
        # Preserve contact-driven recovery if a ball has already slowed nearby.
        reactive=super().collection_ready(v,feet,visible)
        # A genuine hand-first catch ends preparation too. The old body-only
        # flag overwrote both RGB following and the tactile-established anchor
        # on every step until a shin happened to touch the ball.
        reset=e.episode_length_buf==0
        self.preparation_contact[reset]=False
        hand_contact=torch.stack([e.scene[side+'_ball_contact'].data.force.reshape(
            self.num_envs,-1,3).norm(dim=-1).sum(1)>.15 for side in ('left','right')],1).any(1)
        self.preparation_contact|=hand_contact|e._incoming_touched
        before_contact=~self.preparation_contact&~self.grip_hold
        if not self.contact_reference_fix:
            before_contact=~e._incoming_touched
        e._incoming_reach_x=torch.where(before_contact,
            torch.full_like(e._incoming_reach_x,.20)+.015*self.preparation_action[:,1],e._incoming_reach_x)
        return preparing|reactive

    def step(self,actions):
        self.preparation_action=actions[:,22:24].clamp(-1,1).clone()
        obs,reward,done,extras=super().step(actions)
        active=~done.bool(); e=self.env; m=measurements(e)
        in_time=(self.arrival_time>0)&(self.arrival_time<1.5)
        # Privileged physical labels supervise reward only, never actor inputs.
        reward+=e.step_dt*(1.0*in_time*phase(e)*m['stable'].float()
            -2.*e._pose_outcome.intercepted*(m['speed']/.15).clamp(max=3))*active
        extras['log']['Prepare/predicted_arrival_s']=self.arrival_time
        extras['log']['Prepare/ready_before_contact']=e._pose_outcome.ready_before_contact.float()
        self.preparation_action[done.bool()]=0
        self.arrival_time[done.bool()]=12.
        self.preparation_contact[done.bool()]=False
        return self.get_observations(),reward,done,extras


env_class=AnticipatoryEnv
