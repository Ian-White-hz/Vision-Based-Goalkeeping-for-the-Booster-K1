"""RGB-region closure trigger. Ball truth is confined to outcome supervision."""
import torch
from mjlab.managers import TerminationTermCfg
from v12_reference import SynchronizedCaptureEnv as PreviousEnv
from v12_reference import SynchronizedOutcome,make_cfg as previous_cfg,raw_class,measurements
from bridge import table

def region_masks(point,feet,yaw):
    centre=feet.mean(1)
    delta=point-centre
    c,s=torch.cos(yaw),torch.sin(yaw)
    relative=torch.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]),1)
    footdelta=feet[:,0]-feet[:,1]
    half=(.5*(-s*footdelta[:,0]+c*footdelta[:,1]).abs()-.01).clamp(.07,.17)
    inside=(relative[:,0]>=-.03)&(relative[:,0]<=.20)&(relative[:,1].abs()<=half)
    expanded=(relative[:,0]>=-.06)&(relative[:,0]<=.24)&(relative[:,1].abs()<=half+.025)
    return inside,expanded,relative,half

class RegionOutcome(SynchronizedOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        for name in ('visual_closure_seen','visual_strict_success','physical_hold_2s_without_stop'):
            setattr(self,name,torch.zeros(env.num_envs,device=env.device,dtype=torch.bool))
            self.last[name]=getattr(self,name).clone()
        self.physical_clock=torch.zeros(env.num_envs,device=env.device)
        self.timing['max_physical_hold_without_stop_s']=self.physical_clock.clone()
        self.timing_last['max_physical_hold_without_stop_s']=self.physical_clock.clone()
        self.timing['first_visual_close_s']=torch.full_like(self.physical_clock,float('inf'))
        self.timing_last['first_visual_close_s']=self.timing['first_visual_close_s'].clone()

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        trigger=getattr(env,'_visual_region_triggered',torch.zeros_like(self.visual_closure_seen))
        self.visual_closure_seen|=fresh&trigger
        first=fresh&trigger&torch.isinf(self.timing['first_visual_close_s'])
        self.timing['first_visual_close_s'][first]=(env.episode_length_buf*env.step_dt)[first]
        m=measurements(env);robot=env.scene['robot'].data;ball=env.scene['ball'].data
        relative=(ball.root_link_lin_vel_w-robot.body_link_lin_vel_w[:,env._collect_geometry[0]].mean(1)).norm(dim=1)
        ground=env.scene['ball_ground_contact'].data.found.flatten(1).any(1)
        # Diagnostic only: reconstruct the original physical hold conditions,
        # omitting ONLY the historical braked flag. Never terminate on this.
        valid=(m['ready']&m['contacts'].all(1)&ground&m['stable']&m['profile']&~m['failed']
            &(m['speed']<.08)&(relative<.12)&(m['ball_pos'][:,2]>.10)&(m['ball_pos'][:,2]<.14)
            &((ball.root_link_pos_w-m['hands'].mean(1)).norm(dim=1)<.15)&~env._bridge_unsafe)
        if not getattr(env,'_incoming_allow_stationary',False):valid&=env._bridge_distance>.08
        self.physical_clock[fresh]=torch.where(valid,self.physical_clock+env.step_dt,0.)[fresh]
        self.timing['max_physical_hold_without_stop_s']=torch.maximum(self.timing['max_physical_hold_without_stop_s'],self.physical_clock)
        self.physical_hold_2s_without_stop|=fresh&(self.physical_clock>=2.-1e-5)
        result=super().__call__(env)
        self.visual_strict_success|=fresh&self.success&self.visual_closure_seen
        return result

    def reset(self,env_ids=None):
        if hasattr(self,'physical_clock'):
            self.physical_clock[slice(None) if env_ids is None else env_ids]=0
        return super().reset(env_ids)

def make_cfg(num_envs=16):
    cfg=previous_cfg(num_envs)
    cfg.terminations['pose_success'].func=RegionOutcome
    return cfg

class SynchronizedCaptureEnv(PreviousEnv):
    def __init__(self,env):
        super().__init__(env)
        self.region_dwell=torch.zeros(self.num_envs,device=self.device)
        self.last_region_time=torch.full_like(self.region_dwell,-1000.)
        self.visual_triggered=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        env._visual_region_triggered=self.visual_triggered.clone()
        self.region_relative=torch.zeros((self.num_envs,2),device=self.device)

    def collection_reference(self,body_phase):
        e=self.env;v=self.vision;d=e.scene['robot'].data;m=measurements(e)
        reset=e._bridge_stage<2
        self.closure[reset]=0;self.visual_triggered[reset]=False
        self.region_dwell[reset]=0;self.last_region_time[reset]=-1000.
        feet=m['feet'][:,:,:2]-e.scene.env_origins[:,None,:2]
        q=d.root_link_quat_w
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        inside,expanded,self.region_relative,half=region_masks(v.point,feet,yaw)
        fresh=v.visible&v.fit_ready&(v.age<.061)&torch.isfinite(v.point).all(1)
        observed=fresh&inside&(e._bridge_stage==2)
        t=e.episode_length_buf*e.step_dt
        self.region_dwell=torch.where(observed,self.region_dwell+e.step_dt,torch.zeros_like(t))
        self.last_region_time=torch.where(observed,t,self.last_region_time)
        hand_touch=(m['forces']>.15).any(1)
        body_touch=e.scene['body_ball_contact'].data.found.flatten(1).any(1)
        recent=(t-self.last_region_time>=0)&(t-self.last_region_time<=.20)
        assisted=(~fresh)&recent&expanded&(hand_touch|body_touch)
        width=(m['hands'][:,0,1]-m['hands'][:,1,1]).abs()
        prepared=(body_phase>.65)&(e._bridge_stage==2)
        start=prepared&(width>=.235)&((self.region_dwell>=.059)|assisted)
        self.visual_triggered|=start
        # Visible escape with no actual hand contact cancels the old region
        # authorization. Body-contact history alone never releases closure.
        escape=fresh&~expanded&~hand_touch&~self.grip_hold
        self.visual_triggered[escape]=False
        can_continue=(fresh&expanded)|(t-self.last_region_time<=.25)|hand_touch|self.grip_hold
        duration=.65+.15*self.preparation_action[:,0]
        closing=prepared&self.visual_triggered&can_continue
        self.closure=torch.where(closing,(self.closure+e.step_dt/duration).clamp(max=1),self.closure)
        self.closure=torch.where(escape,(self.closure-2*e.step_dt).clamp_min(0),self.closure)
        # Compatibility with the inherited arm-bound adapter: this flag now
        # authorizes closure from visual evidence, not body-contact history.
        self.body_received=self.visual_triggered.clone()
        e._visual_region_triggered=self.visual_triggered.clone()
        self.hand_phase=body_phase.clone()
        e._sync_hand_ready=(self.closure>.9)&self.visual_triggered
        reference=table(e).at(body_phase);opened=self.open_table.at(body_phase)
        u=self.closure*self.closure*(3-2*self.closure)
        for offset in (3,25):
            ids=self.hand_reference_ids+offset
            reference[:,ids]=(1-u[:,None])*opened[:,ids]+u[:,None]*reference[:,ids]
        self.open_arm_target=(reference[:,3:25]+reference[:,25:47])[:,self.path_order]
        return reference

    def step(self,actions):
        obs,reward,done,extras=super().step(actions)
        out=self.env._pose_outcome
        # Replace the old body-first terminal bonus with visually released
        # strict success. All original physical success conditions stay intact.
        reward-=60.*(done.bool()&out.last['body_then_close_success']).float()
        reward+=60.*(done.bool()&out.last['visual_strict_success']).float()
        extras['log']['VisionRegion/triggered']=self.visual_triggered.float()
        extras['log']['VisionRegion/dwell']=self.region_dwell
        self.visual_triggered[done.bool()]=False;self.region_dwell[done.bool()]=0
        self.last_region_time[done.bool()]=-1000.
        return obs,reward,done,extras

env_class=SynchronizedCaptureEnv
