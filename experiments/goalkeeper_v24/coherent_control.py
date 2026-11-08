"""Pure learned locomotion, measured handoff, and calibrated ball-channel stance.

Only normal motor targets are changed. No root/foot/ball state is overwritten.
"""
import torch,copy
from channel_task import ChannelCaptureEnv
from bridge import HANDOFF_SETTLE,phase,table
from controller import measurements

class BrakingGait(torch.nn.Module):
    """Keep the proven moving expert; blend only into the trained stop expert."""
    def __init__(self,original,env,checkpoint):
        super().__init__();self.original=original;self.env=env
        candidates=[m for m in original.modules() if hasattr(m,'mlp') and hasattr(m,'obs_normalizer')]
        assert len(candidates)==1
        self.stop=copy.deepcopy(candidates[0])
        expected=torch.load(checkpoint,map_location=env.device,weights_only=False)['actor_state_dict']
        self.stop.load_state_dict(expected);self.stop.eval().requires_grad_(False)
        assert all(torch.equal(self.stop.state_dict()[k],v) for k,v in expected.items())
        self.blend=torch.zeros(env.num_envs,device=env.device)
    def forward(self,obs):
        raw=self.env.env
        braking=raw._bridge_stage==1
        desired=braking.float()
        self.blend+=(desired-self.blend).clamp(-raw.step_dt/.25,raw.step_dt/.25)
        self.blend[raw.episode_length_buf==0]=0
        a=self.original(obs);b=self.stop(obs)
        return a+(b-a)*self.blend[:,None]

def lane_command(error,velocity,defend):
    # The new gait passed 47/48 drills at .20-.40 m/s; do not request .50 yet.
    desired=(2.6*error-.45*velocity).clamp(-.40,.40)
    # The learned shuffle needs a real stepping command, not an unsupported
    # persistent sub-step command. The arrival/braking gate handles the stop.
    desired=torch.sign(desired)*desired.abs().clamp_min(.18)
    return torch.where(defend,desired,torch.zeros_like(desired))

class CoherentCaptureEnv(ChannelCaptureEnv):
    def __init__(self,e):
        super().__init__(e)
        self.handoff_latched=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.handoff_pose=torch.zeros((self.num_envs,22),device=self.device)
        self.gait_residual_max=torch.zeros(self.num_envs,device=self.device)
        self.receive_width=torch.full((self.num_envs,),.305,device=self.device)
        self.last_width_stamp=-1
        self.measured_stop_time=torch.zeros(self.num_envs,device=self.device)
        self.motor_joint_ids=e.scene['robot'].find_joints(tuple(self.action_names),preserve_order=True)[0]
        self.leg_mask=torch.tensor([not ('Shoulder' in n or 'Elbow' in n or 'Head' in n) for n in self.action_names],device=self.device)

    def install_stop_gait(self,checkpoint):
        self.gait_teacher=BrakingGait(self.gait_teacher,self,checkpoint)

    def approach_lateral_command(self,error,velocity,actions):
        return lane_command(error[:,1],velocity[:,1],self.goal_route_decided&~self.goal_clear_miss)

    def reference_width(self):
        # No inherited measured_width-.042 floor: that could retain a leaky stance.
        # Called during superclass construction before the receive_width buffer exists.
        if not hasattr(self,'receive_width'):
            return torch.full((self.num_envs,),.305,device=self.device)
        return self.receive_width

    def prepare_interception_update(self,feet,visible):
        super().prepare_interception_update(feet,visible)
        if not hasattr(self,'receive_width'):return
        e=self.env;reset=e.episode_length_buf==0;self.receive_width[reset]=.305;self.handoff_latched[reset]=False
        # A return to walking must also clear the old raw adapter's decaying
        # corrections; removing the upper residual alone is not sufficient.
        walking=e._bridge_stage<2
        for name in ('_balance_delta','_width_delta'):
            if hasattr(e,name):getattr(e,name)[walking]=0
        m=measurements(e);d=e.scene['robot'].data
        foot_speed=d.body_link_lin_vel_w[:,e._collect_geometry[1],:2].norm(dim=-1).max(1).values
        # A truthful stop marker must remain measurable in the receiving pose;
        # the old upright-only tilt/width test cannot identify a bent quiet robot.
        quiet=m['support']&(m['margin']>.012)&(m['foot_tilt']<.15)
        quiet&=(m['tilt']-table(e).at(phase(e))[:,2]).abs()<.15
        quiet&=(d.root_link_lin_vel_w[:,:2].norm(dim=1)<.045)&(d.root_link_ang_vel_w.norm(dim=1)<.30)&(foot_speed<.04)
        quiet&=(e._bridge_cmd[:,:2].norm(dim=1)<.025)&(e._bridge_stage>=1)&~reset
        self.measured_stop_time=torch.where(quiet,self.measured_stop_time+e.step_dt,0.)
        e._bridge_braked|=self.measured_stop_time>=.6
        if self.channel_stamp==self.last_width_stamp:return
        self.last_width_stamp=self.channel_stamp
        # FK sweep: .32 leaves a positive rear passage; .305 preserves both
        # entry clearance and rear blockage throughout the reference trajectory.
        self.receive_width.fill_(.305)

    def soft_brake_transition(self,m,d,visible,error,strict_trigger):
        trigger=super().soft_brake_transition(m,d,visible,error,strict_trigger)
        # Establish the entry and rear channel BEFORE the receiving anchor is
        # captured. Walking steps themselves are not held to this stance width.
        stance=self.channel_measured&(self.channel_front>=.008)&(self.channel_rear<=-.002)
        # Collision geometry, not a fixed foot-centre band, determines whether
        # the ball can enter without leaving a rear escape route.
        return trigger&stance

    def adjust_motor_targets(self,target):
        adjusted=super().adjust_motor_targets(target)
        return torch.where((self.env._bridge_stage<2)[:,None],target,adjusted)

    def compose_motor_targets(self,gait,collection,residual):
        e=self.env;collecting=e._bridge_stage==2
        entering=collecting&~self.handoff_latched
        measured=e.scene['robot'].data.joint_pos[:,self.motor_joint_ids]
        self.handoff_pose[entering]=measured[entering]
        self.handoff_latched=collecting.clone()
        u=(e._bridge_clock/HANDOFF_SETTLE).clamp(0,1);u=u*u*(3-2*u)
        # Old high-level leg residuals cannot undo a collision-calibrated stance.
        bounded=torch.where(self.leg_mask[None],residual.clamp(-.006,.006),residual)
        receiving=(1-u[:,None])*self.handoff_pose+u[:,None]*(collection+bounded)
        target=torch.where(collecting[:,None],receiving,gait)
        self.gait_residual_max=torch.where(~collecting,(target-gait).abs().max(1).values,0.)
        assert (self.gait_residual_max<1e-7).all()
        return target

    def step(self,a):
        result=super().step(a)
        done=result[2].bool();self.handoff_latched[done]=False;self.receive_width[done]=.305
        return result
