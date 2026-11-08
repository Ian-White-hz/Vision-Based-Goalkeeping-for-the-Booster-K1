"""Credible perception controls correction; confirmed receiving takes priority."""
import torch
from continuous_interception import ContinuousInterceptEnv
from reliable_vision import ReliableVision
from synchronized_task import region_masks
from bridge import phase
from v12_reference import SynchronizedCaptureEnv as SafeReceiveBase

def priority_mask(quality,inside,dwell,prepared,recent,expanded,hand,body,grip):
    confirmed=quality&inside&(dwell>=.059)&prepared
    assisted=recent&expanded&(hand|body)&prepared
    return confirmed|assisted|hand|grip

class PriorityCaptureEnv(ContinuousInterceptEnv):
    def __init__(self,e):
        super().__init__(e)
        # Free the old renderer while its own context is current. Creating the
        # replacement first can delete the replacement's framebuffer objects.
        old=self.vision
        old.close()
        self.vision=ReliableVision(e)
        self.receive_dwell=torch.zeros(self.num_envs,device=self.device)
        self.receive_seen_at=torch.full_like(self.receive_dwell,-1000.)
        self.receive_priority=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.recovered_clock=torch.zeros_like(self.receive_dwell)

    def prepare_interception_update(self,feet,visible):
        e=self.env;d=e.scene['robot'].data;v=self.vision
        reset=e.episode_length_buf==0
        self.receive_dwell[reset]=self.recovered_clock[reset]=0;self.receive_seen_at[reset]=-1000.
        q=d.root_link_quat_w
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        foot_positions=d.body_link_pos_w[:,e._collect_geometry[1],:2]-e.scene.env_origins[:,None,:2]
        inside,expanded,_,_=region_masks(v.point,foot_positions,yaw)
        quality=v.control_quality&v.visible&(v.age<.061)
        observed=quality&inside&(e._bridge_stage==2)
        self.receive_dwell=torch.where(observed,self.receive_dwell+e.step_dt,0.)
        t=e.common_step_counter*e.step_dt
        self.receive_seen_at=torch.where(observed,torch.full_like(self.receive_seen_at,t),self.receive_seen_at)
        forces=torch.stack([e.scene[s+'_ball_contact'].data.force.reshape(self.num_envs,-1,3).norm(dim=-1).sum(1) for s in ('left','right')],1)
        hand=(forces>.15).any(1)
        body=e.scene['body_ball_contact'].data.found.flatten(1).any(1)
        recent=(t-self.receive_seen_at>=0)&(t-self.receive_seen_at<=.25)
        prepared=(phase(e)>.55)&(e._bridge_stage==2)
        self.receive_priority=priority_mask(quality,inside,self.receive_dwell,prepared,recent,expanded,hand,body,self.grip_hold)
        error=(self.intercept_target[:,1]-feet[:,1]).abs()
        recovered=quality&self.prediction_stable&(error<.055)
        self.recovered_clock=torch.where(recovered,self.recovered_clock+e.step_dt,0.)
        mismatch=(e._bridge_stage==2)&self.prediction_stable&quality&(error>.08)&~self.receive_priority
        self.mismatch_clock=torch.where(mismatch,self.mismatch_clock+e.step_dt,0.)
        start=(self.mismatch_clock>=.12)&~self.realigning
        self.realign_count+=start.long();self.realigning|=start
        cancel=self.receive_priority|(self.recovered_clock>=.10)|hand|self.grip_hold
        self.realigning[cancel]=False;self.mismatch_clock[cancel]=0
        returning=self.realigning&(e._bridge_stage==2)
        e._incoming_ready[returning]=True
        tilt=torch.acos((1-2*(q[:,1].square()+q[:,2].square())).clamp(-1,1))
        finished=returning&(e._bridge_clock<=e.step_dt)&(tilt<.20)
        e._bridge_stage[finished]=0;e._bridge_clock[finished]=0;e._bridge_quiet[finished]=0
        e._incoming_ready[finished]=False;self.soft_support_time[finished]=0
        self.realigning[finished]=False;self.mismatch_clock[finished]=0

    def closing_permitted(self):
        return ~self.realigning

    def descent_clock_increment(self):
        normal=super().descent_clock_increment()
        # The inherited support/tilt/speed protection remains active when a
        # verified near ball takes priority over the old long-horizon lane.
        safe=SafeReceiveBase.descent_clock_increment(self)
        result=torch.where(self.receive_priority,safe,normal)
        blind_rewind=self.realigning&~self.vision.control_quality&~self.receive_priority
        return torch.where(blind_rewind,torch.zeros_like(result),result)

    def step(self,a):
        result=super().step(a)
        ids=result[2].bool();self.receive_priority[ids]=False
        self.receive_dwell[ids]=self.recovered_clock[ids]=0;self.receive_seen_at[ids]=-1000.
        return result

env_class=PriorityCaptureEnv
