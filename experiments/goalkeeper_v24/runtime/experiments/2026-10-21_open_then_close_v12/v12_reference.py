"""Stop safely, prepare the torso, then deploy hands around RGB body-arrival.

Only RGB, measured robot configuration, and genuine tactile state select motor
references. Physical ball truth below is restricted to rewards and evaluation.
"""
import torch
from reach_task import ReachTable
from tensordict import TensorDict
from mjlab.managers import TerminationTermCfg
from adaptive_capture_task import AdaptiveCaptureEnv
from random_capture_task import make_cfg as original_cfg,raw_class,measurements,STAGE
from anticipatory_task import PreparationOutcome
from bridge import HANDOFF_SETTLE,DURATION,table,phase

class SynchronizedOutcome(PreparationOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        for name in ('synchronized_contact','synchronized_success','body_then_close_contact','body_then_close_success'):
            setattr(self,name,torch.zeros(env.num_envs,device=env.device,dtype=torch.bool))
            self.last[name]=getattr(self,name).clone()
        self.timing['first_hand_contact_s']=torch.full((env.num_envs,),float('inf'),device=env.device)
        self.timing_last['first_hand_contact_s']=self.timing['first_hand_contact_s'].clone()

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        selected=fresh&m['contacts'].any(1)&torch.isinf(self.timing['first_hand_contact_s'])
        self.timing['first_hand_contact_s'][selected]=(env.episode_length_buf*env.step_dt)[selected]
        result=super().__call__(env)
        body=self.timing['first_body_contact_s'];hand=self.timing['first_hand_contact_s']
        both=self.timing['first_bilateral_s']
        finite=torch.isfinite(body)&torch.isfinite(hand)&torch.isfinite(both)
        synchronized=finite&((hand-body).abs()<=.4)&((both-body).abs()<=.6)
        self.synchronized_contact|=fresh&synchronized
        self.synchronized_success|=fresh&self.success&self.synchronized_contact
        ordered=finite&(hand>=body-.021)&(both>=body)&(both-body<=1.2)
        self.body_then_close_contact|=fresh&ordered
        self.body_then_close_success|=fresh&self.success&self.body_then_close_contact
        return result

def make_cfg(num_envs=16):
    cfg=original_cfg(num_envs)
    cfg.terminations['pose_success']=TerminationTermCfg(func=SynchronizedOutcome)
    return cfg

class SynchronizedCaptureEnv(AdaptiveCaptureEnv):
    servo_requires_braked=False
    impact_enabled=False
    deadline_enabled=False
    handoff_duration=1.2
    nominal_descent_duration=3.

    def __init__(self,env):
        super().__init__(env)
        self.open_table=ReachTable(env,'open_reach_paths.json')
        self.closure=torch.zeros(self.num_envs,device=self.device)
        self.body_received=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.open_arm_target=None
        self.hand_phase=torch.zeros(self.num_envs,device=self.device)
        env._sync_hand_ready=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.hand_reference_ids=torch.tensor([i for i,n in enumerate(table(env).names)
            if 'Shoulder' in n or 'Elbow' in n],device=self.device)
        self.soft_support_time=torch.zeros(self.num_envs,device=self.device)
        self.sync_event_paid=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)

    def get_observations(self):
        base=super().get_observations()
        if not hasattr(self,'hand_phase'):return base
        actor=base['actor'].clone()
        # Same 128/134-dimensional interface as parent; replace the redundant
        # ready bit (also encoded by stage/phase) with observable hand progress.
        actor[:,-2]=self.closure
        return TensorDict({'actor':actor,'critic':torch.cat((actor,base['critic'][:,-6:]),1)},batch_size=[self.num_envs])

    def collection_ready(self,v,feet,visible):
        inherited=super().collection_ready(v,feet,visible)
        e=self.env
        # Body/shin arrival plane estimated from robot feet plus known geometry;
        # no sampled launch target or physical ball position is read here.
        speed=(-self.lane_velocity[:,0]).clamp_min(.05)
        self.arrival_time=((v.point[:,0]-feet[:,0]-.10)/speed).clamp(-1.,12.)
        # Match the hand reference to the robot's body-block geometry, rather
        # than the old x=.20 frontal-ball target. The private reference screen
        # places shin interception near x=.095. Real RGB/tactile following takes
        # over after contact; no physical ball truth enters this reference.
        before_contact=~self.preparation_contact&~self.grip_hold
        receive_x=.10+.015*self.preparation_action[:,1]
        e._incoming_reach_x=torch.where(before_contact,receive_x,e._incoming_reach_x)
        stopped=e._bridge_stage==2
        forward=(v.point[:,0]-feet[:,0]>-.08)&(v.point[:,0]-feet[:,0]<5.)
        reliable=visible&v.fit_ready&self.lane_ready&forward
        # Start torso preparation once physically stopped. Hand deployment is
        # scheduled separately, so this does not require incoming speed/TTC.
        recovery=self.preparation_contact&visible&forward
        return inherited|(stopped&(reliable|recovery))

    def soft_brake_transition(self,m,d,visible,error,strict_trigger):
        # Arriving switches to braking immediately; actual quiet-stop telemetry
        # remains truthful and independent of permission to prepare the torso.
        e=self.env
        feet_speed=d.body_link_lin_vel_w[:,e._collect_geometry[1],:2].norm(dim=-1).max(1).values
        supported=m['support']&(m['margin']>.008)&(m['foot_tilt']<.15)&(m['tilt']<.20)
        supported&=(d.root_link_lin_vel_w[:,:2].norm(dim=1)<.10)&(feet_speed<.08)
        supported&=d.root_link_ang_vel_w.norm(dim=1)<.45
        self.soft_support_time=torch.where((e._bridge_stage==1)&supported,
            self.soft_support_time+e.step_dt,torch.zeros_like(self.soft_support_time))
        start=(e._bridge_stage==1)&(self.soft_support_time>=.12)&visible&(error[:,1].abs()<.055)&(error[:,0].abs()<.05)
        return strict_trigger|start

    def descent_clock_increment(self):
        increment=super().descent_clock_increment()
        e=self.env;m=measurements(e);d=e.scene['robot'].data
        speed=d.root_link_lin_vel_w[:,:2].norm(dim=1)
        angular=d.root_link_ang_vel_w.norm(dim=1)
        # Continuous proprioceptive protection: allow shallow preparation while
        # braking, deeper descent only as motion and body lean subside.
        reference_tilt=table(e).at(self.phase())[:,2]
        tilt_error=(m['tilt']-reference_tilt).abs()
        calm=torch.minimum(((.22-speed)/.16).clamp(0,1),((.28-tilt_error)/.16).clamp(0,1))
        calm=torch.minimum(calm,((1.2-angular)/.8).clamp(0,1))
        cap=.08+.92*calm
        supported=m['support']&(m['foot_tilt']<.25)&(m['margin']>-.015)
        limit=HANDOFF_SETTLE+DURATION*cap
        allowed=(limit-e._bridge_clock).clamp_min(0)
        # Once the full pose is reached, continue the ordinary real-time dwell.
        allowed=torch.where((calm>=.999)&(e._bridge_clock>=HANDOFF_SETTLE+DURATION-1e-5),increment,allowed)
        return torch.where(supported,torch.minimum(increment,allowed),torch.zeros_like(increment))

    def collection_reference(self,body_phase):
        e=self.env;reset=e._bridge_stage<2
        self.closure[reset]=0;self.body_received[reset]=False
        # Only the dedicated actual non-hand body contact sensor releases
        # closing. A visual arrival prediction or hand-first contact cannot.
        body=e.scene['body_ball_contact'].data.found.flatten(1).any(1)
        self.body_received|=body&(e._bridge_stage==2)
        duration=.65+.15*self.preparation_action[:,0]
        close=self.body_received&(body_phase>.65)
        self.closure=torch.where(close,(self.closure+e.step_dt/duration).clamp(max=1),self.closure)
        self.hand_phase=body_phase.clone()
        e._sync_hand_ready=(self.closure>.9)&self.body_received
        reference=table(e).at(body_phase)
        opened=self.open_table.at(body_phase)
        u=self.closure*self.closure*(3-2*self.closure)
        for offset in (3,25):
            ids=self.hand_reference_ids+offset
            reference[:,ids]=(1-u[:,None])*opened[:,ids]+u[:,None]*reference[:,ids]
        self.open_arm_target=(reference[:,3:25]+reference[:,25:47])[:,self.path_order]
        return reference

    def adjust_motor_targets(self,target):
        result=super().adjust_motor_targets(target)
        if self.open_arm_target is not None:
            # Keep the preparatory opening from being cancelled by actor or
            # endpoint residuals before real body interception.
            active=(self.env._bridge_stage==2)&~self.body_received
            arm=[i for i,n in enumerate(self.action_names) if 'Shoulder' in n or 'Elbow' in n]
            for i in arm:
                constrained=self.open_arm_target[:,i]+(result[:,i]-self.open_arm_target[:,i]).clamp(-.02,.02)
                # Use the same gait-to-reference blend during initial handoff.
                u=(self.env._bridge_clock/HANDOFF_SETTLE).clamp(0,1)
                u=u*u*(3-2*u)
                result[:,i]=torch.where(active,(1-u)*result[:,i]+u*constrained,result[:,i])
        return result

    def step(self,actions):
        previous=self.sync_event_paid.clone()
        obs,reward,done,extras=super().step(actions)
        e=self.env;out=e._pose_outcome;m=measurements(e);active=~done.bool()
        # Supervision only: actual body-contact age and real hand distances.
        age=e.episode_length_buf*e.step_dt-out.timing['first_body_contact_s']
        near_body=(age>=0)&(age<=.8)
        reaching=torch.exp(-((m['distances']-.133)/.06).square()).mean(1)
        reward+=e.step_dt*(3.*near_body*reaching+4.*near_body*m['contacts'].all(1)
            +3.*m['valid'].float())*active
        new=out.body_then_close_contact&~previous&active
        reward+=12.*new.float()
        self.sync_event_paid|=out.body_then_close_contact
        reward+=60.*(done.bool()&out.last['body_then_close_success']).float()
        self.sync_event_paid[done.bool()]=False
        self.closure[done.bool()]=0
        self.body_received[done.bool()]=False
        self.hand_phase[done.bool()]=0
        e._sync_hand_ready[done.bool()]=False
        extras['log']['Receive/closure']=self.closure
        extras['log']['Receive/body_received']=self.body_received.float()
        extras['log']['Sync/contact_window']=out.synchronized_contact.float()
        return self.get_observations(),reward,done,extras

env_class=SynchronizedCaptureEnv
