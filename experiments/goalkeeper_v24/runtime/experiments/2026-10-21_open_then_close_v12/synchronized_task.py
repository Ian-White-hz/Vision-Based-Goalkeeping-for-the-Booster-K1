"""Sequential vision, timing and light-force variants using only RGB/tactile control."""
import numpy as np
import torch
from v13_reference import SynchronizedCaptureEnv as V13Env,RegionOutcome,region_masks,make_cfg as old_cfg,raw_class,measurements
from bridge import table

class StabilityOutcome(RegionOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.gross_false_visual_start=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.last['gross_false_visual_start']=self.gross_false_visual_start.clone()

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        trigger=getattr(env,'_visual_region_triggered',torch.zeros_like(self.visual_closure_seen))
        first=fresh&trigger&~self.visual_closure_seen
        # Privileged truth is an offline-outcome label only, never returned to
        # the actor and never consulted by any motor or closure decision.
        m=measurements(env);d=env.scene['robot'].data;q=d.root_link_quat_w
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        feet=m['feet'][:,:,:2]-env.scene.env_origins[:,None,:2]
        _,_,relative,half=region_masks(m['ball_pos'][:,:2],feet,yaw)
        gross=(relative[:,0]<-.12)|(relative[:,0]>.35)|(relative[:,1].abs()>half+.08)
        self.gross_false_visual_start|=first&gross
        return super().__call__(env)

def make_cfg(num_envs=16):
    cfg=old_cfg(num_envs);cfg.terminations['pose_success'].func=StabilityOutcome
    return cfg

class VisionGuardEnv(V13Env):
    adaptive_closure=False
    gentle_hold=False
    def __init__(self,env):
        super().__init__(env)
        self.previous_rgb=torch.zeros((self.num_envs,2),device=self.device)
        self.previous_rgb_valid=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.jump_quarantine_until=torch.zeros(self.num_envs,device=self.device)
        self.quality_ok=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.closure_rate=torch.zeros(self.num_envs,device=self.device)

    def collection_reference(self,body_phase):
        e=self.env;v=self.vision;d=e.scene['robot'].data;m=measurements(e)
        reset=e._bridge_stage<2
        self.closure[reset]=0;self.visual_triggered[reset]=False
        self.region_dwell[reset]=0;self.last_region_time[reset]=-1000.
        self.previous_rgb_valid[reset]=False;self.jump_quarantine_until[reset]=0
        feet=m['feet'][:,:,:2]-e.scene.env_origins[:,None,:2]
        q=d.root_link_quat_w
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        inside,expanded,self.region_relative,half=region_masks(v.point,feet,yaw)
        t=e.episode_length_buf*e.step_dt;global_t=e.common_step_counter*e.step_dt
        visible=v.visible&v.fit_ready&(v.age<.061)&torch.isfinite(v.point).all(1)
        jump=visible&self.previous_rgb_valid&((v.point-self.previous_rgb).norm(dim=1)>.04)
        self.jump_quarantine_until=torch.where(jump,t+.12,self.jump_quarantine_until)
        self.previous_rgb=torch.where(visible[:,None],v.point,self.previous_rgb)
        self.previous_rgb_valid=visible&~reset
        fit_age=global_t-v.corrected_at
        self.quality_ok=visible&(v.fit_loss<.03)&(fit_age>=0)&(fit_age<=.16)&(t>=self.jump_quarantine_until)
        observed=self.quality_ok&inside&(e._bridge_stage==2)
        self.region_dwell=torch.where(observed,self.region_dwell+e.step_dt,torch.zeros_like(t))
        self.last_region_time=torch.where(observed,t,self.last_region_time)
        forces=m['forces'];hand_touch=(forces>.15).any(1)
        body_touch=e.scene['body_ball_contact'].data.found.flatten(1).any(1)
        recent=(t-self.last_region_time>=0)&(t-self.last_region_time<=.20)
        assisted=(~visible)&recent&expanded&(hand_touch|body_touch)
        width=(m['hands'][:,0,1]-m['hands'][:,1,1]).abs()
        prepared=(body_phase>.65)&(e._bridge_stage==2)
        start=prepared&(width>=.235)&((self.region_dwell>=.059)|assisted)
        self.visual_triggered|=start
        escape=visible&~expanded&~hand_touch&~self.grip_hold
        self.visual_triggered[escape]=False
        can_continue=(self.quality_ok&expanded)|(t-self.last_region_time<=.25)|hand_touch|self.grip_hold
        duration=.65+.15*self.preparation_action[:,0]
        rate=1./duration
        if self.adaptive_closure:
            c,s=torch.cos(yaw),torch.sin(yaw)
            approach=-(c*v.velocity[:,0]+s*v.velocity[:,1])
            tta=(self.region_relative[:,0]-.10).clamp_min(0)/approach.clamp(.05,1.2)
            scheduled=((1-self.closure)/(tta+.25)).clamp(.6,2.2)
            rate=torch.where(self.quality_ok,scheduled,rate)
            rate=torch.where(hand_touch,torch.minimum(rate,torch.full_like(rate,.8)),rate)
        if self.gentle_hold:
            # After bilateral contact, maintain the existing opening and let
            # independent tactile feedback regulate pressure, rather than push.
            bilateral=(forces>.25).all(1)&(self.closure>.55)
            rate=torch.where(bilateral,torch.zeros_like(rate),rate)
        self.closure_rate=rate
        closing=prepared&self.visual_triggered&can_continue
        if hasattr(self,"closing_permitted"): closing &= self.closing_permitted()
        self.closure=torch.where(closing,(self.closure+e.step_dt*rate).clamp(max=1),self.closure)
        self.closure=torch.where(escape,(self.closure-2*e.step_dt).clamp_min(0),self.closure)
        self.body_received=self.visual_triggered.clone();e._visual_region_triggered=self.visual_triggered.clone()
        self.hand_phase=body_phase.clone()
        contact_feedback=hand_touch&(body_phase>.98)&(self.closure>.55)&self.gentle_hold
        e._sync_hand_ready=((self.closure>.9)|contact_feedback)&self.visual_triggered
        reference=table(e).at(body_phase);opened=self.open_table.at(body_phase)
        u=self.closure*self.closure*(3-2*self.closure)
        for offset in (3,25):
            ids=self.hand_reference_ids+offset
            reference[:,ids]=(1-u[:,None])*opened[:,ids]+u[:,None]*reference[:,ids]
        self.open_arm_target=(reference[:,3:25]+reference[:,25:47])[:,self.path_order]
        return reference

    def update_hand_radius(self,index,forces):
        if not self.gentle_hold:return super().update_hand_radius(index,forces)
        old=self.hand_radius[index].copy()
        # Retain the established real-contact dwell and grip-release logic.
        super().update_hand_radius(index,forces)
        if (forces>.15).any():
            change=np.where(forces<.35,-.000025,np.where(forces>1.2,.00003,0.))
            change=np.where(forces>3.,.00012,change)
            self.hand_radius[index]=np.clip(old+change,.132,.137)
        else:
            self.hand_radius[index]=np.clip(self.hand_radius[index],.132,.137)

class AdaptiveReceiveEnv(VisionGuardEnv):adaptive_closure=True
class GentleHoldEnv(AdaptiveReceiveEnv):gentle_hold=True
SynchronizedCaptureEnv=GentleHoldEnv
env_class=GentleHoldEnv
