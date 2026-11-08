"""Continuous RGB interception, gated arrival and reversible preparation.

No launch metadata or true ball state is read by the control methods. Rewards
and the unchanged posture acceptance may use privileged measurements.
"""
import torch
from synchronized_task import GentleHoldEnv, measurements
from bridge import book, phase, HANDOFF_SETTLE
from goalkeeper_sim.full_goal.settings import FullGoalSettings


def arrival_mask(stable, predicted, filtered, feet, visible):
    return (stable & visible & ((predicted[:, 1]-feet[:, 1]).abs()<.035)
            & ((filtered[:, 1]-feet[:, 1]).abs()<.025))


class ContinuousInterceptEnv(GentleHoldEnv):
    def __init__(self, env):
        super().__init__(env)
        self.prediction_history=torch.zeros((self.num_envs,8),device=self.device)
        self.prediction_valid=torch.zeros((self.num_envs,8),device=self.device,dtype=torch.bool)
        self.prediction_stable=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.prediction_fresh=self.prediction_stable.clone()
        self.intercept_target=torch.zeros((self.num_envs,2),device=self.device)
        self.prediction_rms=torch.full((self.num_envs,),1.,device=self.device)
        self.realigning=self.prediction_stable.clone()
        self.mismatch_clock=torch.zeros(self.num_envs,device=self.device)
        self.realign_count=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.last_prediction_time=torch.full_like(self.mismatch_clock,-1000.)
        self.last_prediction_stamp=-1
        self.goal_settings=FullGoalSettings()
        self.goal_clear_miss=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.goal_clear_miss_ever=self.goal_clear_miss.clone()
        self.goal_route_decided=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.goal_predicted_y=torch.zeros(self.num_envs,device=self.device)
        self.goal_predicted_time=torch.zeros(self.num_envs,device=self.device)

    def target_point(self,v,feet,visible):
        # Parent updates the robust 64-frame RGB-only lane fit exactly once.
        super().target_point(v,feet,visible)
        e=self.env
        if self.last_prediction_stamp==e.common_step_counter:return self.intercept_target
        self.last_prediction_stamp=e.common_step_counter
        reset=e.episode_length_buf==0
        self.prediction_valid[reset]=False
        self.prediction_stable[reset]=False
        self.realigning[reset]=False
        self.mismatch_clock[reset]=0
        self.realign_count[reset]=0
        self.last_prediction_time[reset]=-1000.
        self.intercept_target[reset]=feet[reset,:2]
        self.goal_clear_miss[reset]=False;self.goal_clear_miss_ever[reset]=False;self.goal_route_decided[reset]=False
        self.goal_predicted_y[reset]=0;self.goal_predicted_time[reset]=0
        t=e.common_step_counter*e.step_dt
        w=self.track_valid.float()
        n=w.sum(1).clamp_min(1)
        mt=(self.track_times*w).sum(1)/n
        mp=(self.track_positions*w[:,:,None]).sum(1)/n[:,None]
        fit=mp[:,None]+self.lane_velocity[:,None]*(self.track_times-mt[:,None])[:,:,None]
        residual=(self.track_positions-fit).norm(dim=2)
        robust=w*(.04/residual.clamp_min(.04))
        nr=robust.sum(1).clamp_min(1)
        mt=(self.track_times*robust).sum(1)/nr
        mp=(self.track_positions*robust[:,:,None]).sum(1)/nr[:,None]
        fit=mp[:,None]+self.lane_velocity[:,None]*(self.track_times-mt[:,None])[:,:,None]
        self.prediction_rms=(((self.track_positions-fit).square().sum(2)*robust).sum(1)/nr).sqrt()
        current=mp+self.lane_velocity*(t-mt)[:,None]
        # Predict the robot-relative receiving plane, rather than the ball's
        # present lateral coordinate. Frozen guarding x remains unchanged.
        receive_x=feet[:,0]+.10
        tta=((current[:,0]-receive_x)/(-self.lane_velocity[:,0]).clamp_min(.04)).clamp(0,12)
        lane=current[:,1]+self.lane_velocity[:,1]*tta
        fresh=v.visible & visible & (v.age<.061) & torch.isfinite(v.point).all(1)
        approaching=fresh & self.lane_ready & (self.lane_velocity[:,0]<-.04) & (self.prediction_rms<.035)
        relative=v.point-feet[:,:2]
        stopped=fresh & v.fit_ready & (v.velocity.norm(dim=1)<.07)
        stopped &= (relative[:,0]>-.06)&(relative[:,0]<.35)&(relative[:,1].abs()<.20)
        self.prediction_fresh=approaching|stopped
        lane=torch.where(approaching,lane,v.point[:,1]).clamp(-1.07,1.07)
        desired=torch.stack((torch.full_like(lane,float(book(e)['feet'][:,0].mean())),lane),1)
        self.intercept_target=torch.where(self.prediction_fresh[:,None],desired,self.intercept_target)
        self.last_prediction_time=torch.where(self.prediction_fresh,torch.full_like(lane,t),self.last_prediction_time)
        slot=e.common_step_counter%8
        self.prediction_history[:,slot]=lane
        self.prediction_valid[:,slot]=self.prediction_fresh
        span=self.prediction_history.max(1).values-self.prediction_history.min(1).values
        self.prediction_stable=self.prediction_valid.all(1)&(span<.035)&self.prediction_fresh
        # Suppress lateral defense only when a stable RGB trajectory clearly misses
        # beyond both post and ball radius; uncertain/near-post routes remain defended.
        goal=self.goal_settings
        raw_goal_tta=(current[:,0]-goal.goal_x)/(-self.lane_velocity[:,0]).clamp_min(.04)
        goal_tta=raw_goal_tta.clamp(0,12)
        goal_y=current[:,1]+self.lane_velocity[:,1]*goal_tta
        uncertainty=(.08+2*self.prediction_rms).clamp(max=.18)
        outside=goal_y.abs()>(goal.goal_width/2+goal.ball_radius+uncertainty)
        inside_limit=(goal.goal_width/2-goal.ball_radius+.02-self.prediction_rms).clamp_min(0)
        confidently_inside=goal_y.abs()<inside_limit
        candidate_miss=approaching&self.prediction_stable&(raw_goal_tta>0)&(raw_goal_tta<12)&outside
        certain_inside=approaching&self.prediction_stable&(raw_goal_tta>0)&(raw_goal_tta<12)&confidently_inside
        self.goal_route_decided|=candidate_miss|certain_inside
        self.goal_clear_miss=torch.where(certain_inside,torch.zeros_like(self.goal_clear_miss),
            self.goal_clear_miss|candidate_miss)
        self.goal_clear_miss_ever|=self.goal_clear_miss
        self.goal_predicted_y=goal_y;self.goal_predicted_time=raw_goal_tta
        home=torch.zeros_like(self.intercept_target)
        home[:,0]=float(book(e)['feet'][:,0].mean())
        self.intercept_target=torch.where(self.goal_clear_miss[:,None],home,self.intercept_target)
        return self.intercept_target

    def approach_lateral_command(self,error,velocity,actions):
        command=super().approach_lateral_command(error,velocity,actions)
        defend=self.goal_route_decided&~self.goal_clear_miss
        return torch.where(defend,command,torch.zeros_like(command))

    def arrival_gate(self,feet,visible):
        return self.goal_route_decided&~self.goal_clear_miss

    def collection_ready(self,v,feet,visible):
        return super().collection_ready(v,feet,visible)&self.goal_route_decided&~self.goal_clear_miss

    def target_tracking_mask(self,visible):
        # Continue updating the target even during descent. Only upright gait
        # may step; mismatch in a bent pose requests a smooth return first.
        return self.prediction_fresh

    def prepare_interception_update(self,feet,visible):
        e=self.env
        robot=e.scene['robot'].data
        forces=torch.stack([e.scene[side+'_ball_contact'].data.force.reshape(self.num_envs,-1,3).norm(dim=-1).sum(1)
                            for side in ('left','right')],1)
        touch=(forces>.15).any(1)|self.grip_hold
        # Use measured joints/root only for stance safety; no true ball inputs.
        q=robot.root_link_quat_w
        tilt=torch.acos((1-2*(q[:,1].square()+q[:,2].square())).clamp(-1,1))
        error=(self.intercept_target[:,1]-feet[:,1]).abs()
        mismatch=(e._bridge_stage==2)&self.prediction_stable&(error>.08)&~touch
        self.mismatch_clock=torch.where(mismatch,self.mismatch_clock+e.step_dt,0.)
        start=(self.mismatch_clock>=.12)&~self.realigning
        self.realign_count+=start.long()
        self.realigning|=start
        self.realigning[touch]=False
        returning=self.realigning&(e._bridge_stage==2)
        e._incoming_ready[returning]=True  # permits reversible clock, never success
        finished=returning&(e._bridge_clock<=e.step_dt)&(tilt<.20)
        e._bridge_stage[finished]=0
        e._bridge_clock[finished]=0
        e._bridge_quiet[finished]=0
        e._incoming_ready[finished]=False
        self.soft_support_time[finished]=0
        self.realigning[finished]=False
        self.mismatch_clock[finished]=0

    def arrival_gate(self,feet,visible):
        return (arrival_mask(self.prediction_stable,self.intercept_target,self.env._bridge_target,feet,visible)
                &self.goal_route_decided&~self.goal_clear_miss)

    def reacquire_mask(self,error,visible):
        # No stale quiet-stop prerequisite for correcting a newly changed lane.
        return (self.env._bridge_stage==1)&self.prediction_stable&(error[:,1].abs()>.055)

    def soft_brake_transition(self,m,d,visible,error,strict_trigger):
        trigger=super().soft_brake_transition(m,d,visible,error,strict_trigger)
        feet=m['feet'].mean(1)-self.env.scene.env_origins
        # Allow the measured few-centimetre braking drift already supported by
        # the parent handoff; the tighter arrival band is for moving->braking.
        aligned=(self.intercept_target[:,1]-feet[:,1]).abs()<.055
        return trigger & self.prediction_stable & aligned & visible & ~self.realigning

    def collection_ready(self,v,feet,visible):
        ready=super().collection_ready(v,feet,visible)
        # Recent validated prediction survives short visual occlusion. Actual
        # contact recovery is retained; body history alone never closes hands.
        t=self.env.common_step_counter*self.env.step_dt
        recent=(t-self.last_prediction_time<=.25)
        aligned=(self.intercept_target[:,1]-feet[:,1]).abs()<.065
        recovery=self.preparation_contact|self.grip_hold
        return ready & ((recent&aligned)|recovery) & ~self.realigning & self.goal_route_decided & ~self.goal_clear_miss

    def descent_clock_increment(self):
        increment=super().descent_clock_increment()
        e=self.env
        robot=e.scene['robot'].data
        forces=torch.stack([e.scene[s+'_ball_contact'].data.force.reshape(self.num_envs,-1,3).norm(dim=-1).sum(1)
                            for s in ('left','right')],1)
        hands=(forces>.15).any(1)|self.grip_hold
        feet=robot.body_link_pos_w[:,e._collect_geometry[1],:2].mean(1)-e.scene.env_origins[:,:2]
        mismatch=(self.intercept_target[:,1]-feet[:,1]).abs()>.065
        pause=(e._bridge_stage==2)&self.prediction_stable&mismatch&~hands
        increment=torch.where(pause,torch.zeros_like(increment),increment)
        # Rewind the same smooth reference at <=normal descent clock rate.
        # Walking commands remain zero until the reference is upright again.
        rewind=-torch.minimum(e._bridge_clock,torch.full_like(increment,e.step_dt))
        return torch.where(self.realigning,rewind,increment)

    def step(self,actions):
        obs,reward,done,extras=super().step(actions)
        e=self.env
        # Privileged supervision only: current posture-grounded validity, not
        # the obsolete upright-braked gate, earns the continuous hold bonus.
        active=~done.bool()
        reward+=e.step_dt*5.*(e._pose_outcome.current_hold>0)*active
        extras['log']['Intercept/prediction_stable']=self.prediction_stable.float()
        extras['log']['Intercept/realigning']=self.realigning.float()
        extras['log']['Intercept/fit_rms_m']=self.prediction_rms
        ids=done.bool()
        self.prediction_valid[ids]=False
        self.prediction_stable[ids]=False
        self.prediction_fresh[ids]=False
        self.realigning[ids]=False
        self.mismatch_clock[ids]=0
        self.last_prediction_time[ids]=-1000.
        return obs,reward,done,extras


env_class=ContinuousInterceptEnv
