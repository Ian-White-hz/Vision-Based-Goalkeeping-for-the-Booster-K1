"""RGB lane prediction, upright body block, then grounded collection.

This first rolling drill deliberately uses narrow parallel shots at .30-.40m/s.
Full-width independent start/target sampling is a later, separately tested gate.
"""
import math
import sys
sys.dont_write_bytecode=True
import torch
from tensordict import TensorDict
from mjlab.managers import EventTermCfg,TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from controller import BalancedRawEnv,BalancedKeeperEnv,make_cfg as static_cfg
from bridge import BridgeOutcome,book,table,phase,measurements as static_measurements,HANDOFF_SETTLE
from goalkeeper_sim.full_goal.outcomes import goal_crossed
from goalkeeper_sim.full_goal.settings import FullGoalSettings


def launch(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    if not hasattr(env,'_incoming_ready'):
        env._incoming_ready=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        env._incoming_touched=env._incoming_ready.clone()
        env._incoming_initial_x=torch.zeros(env.num_envs,device=env.device)
        env._incoming_initial_speed=env._incoming_initial_x.clone()
    sign=torch.where(torch.rand(len(ids),device=env.device)<.5,-1.,1.)
    if getattr(env,'_balanced_eval',False): sign=torch.where(ids%2==0,-1.,1.)
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0]=1.8+.8*torch.rand(len(ids),device=env.device)
    state[:,1]=sign*(.18+.04*torch.rand(len(ids),device=env.device))
    state[:,2],state[:,3]=.112,1.
    speed=.30+.10*torch.rand(len(ids),device=env.device)
    state[:,7]=-speed; state[:,11]=-speed/.11
    env._bridge_left[ids]=sign>0
    env._incoming_initial_x[ids]=state[:,0]
    env._incoming_initial_speed[ids]=speed
    env._incoming_ready[ids]=env._incoming_touched[ids]=False
    state[:,:3]+=env.scene.env_origins[ids]
    # Only ordinary episode resets write ball state; subsequent motion is physics.
    env.scene['ball'].write_root_state_to_sim(state,env_ids=ids)


def measurements(env):
    m=static_measurements(env)
    if getattr(env,'_incoming_allow_stationary',False):
        # A centered ball need not cause pointless lateral travel. Reconstruct
        # the same strict physical hold gates, omitting ONLY minimum travel.
        robot=env.scene['robot'].data; ball=env.scene['ball'].data
        hands=env._collect_geometry[0]
        relative=(ball.root_link_lin_vel_w-robot.body_link_lin_vel_w[:,hands].mean(1)).norm(dim=1)
        ground=env.scene['ball_ground_contact'].data.found.flatten(1).any(1)
        m['valid']=(m['ready']&m['contacts'].all(1)&ground&m['stable']&m['profile']&~m['failed']
            &(m['speed']<.08)&(relative<.12)&(m['ball_pos'][:,2]>.10)&(m['ball_pos'][:,2]<.14)
            &((ball.root_link_pos_w-m['hands'].mean(1)).norm(dim=1)<.15)
            &env._bridge_braked&~env._bridge_unsafe)
    m['goal']=goal_crossed(m['ball_pos'],FullGoalSettings())
    m['failed']=m['failed']|m['goal']
    m['valid']=m['valid']&~m['goal']
    return m


class IncomingOutcome(BridgeOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        for name in ('motion_seen','travelled','body_blocked','goal'):
            setattr(self,name,self.success.clone()); self.last[name]=self.success.clone()
        env._incoming_outcome=self

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        self.pose_id[fresh]=env._pose_ids[fresh]
        self.failed[fresh]|=m['failed'][fresh]; self.fallen[fresh]|=m['fallen'][fresh]
        self.left_seen[fresh]|=m['contacts'][fresh,0]; self.right_seen[fresh]|=m['contacts'][fresh,1]
        self.both_seen[fresh]|=m['contacts'][fresh].all(1)
        self.moved[fresh]|=env._bridge_distance[fresh]>.08
        self.braked[fresh]|=env._bridge_braked[fresh]
        self.unsafe[fresh]|=env._bridge_unsafe[fresh]
        self.descent[fresh]|=phase(env)[fresh]>.9999
        self.left[fresh]=env._bridge_left[fresh]
        self.motion_seen[fresh]|=((m['speed']>.15)&(m['ball_pos'][:,0]>.6))[fresh]
        self.travelled[fresh]|=((env._incoming_initial_x-m['ball_pos'][:,0])>.6)[fresh]
        tactile=env.scene['body_ball_contact'].data.found.flatten(1).any(1)
        # A body contact must precede the hands; a ball that stops short is not a save.
        self.body_blocked[fresh]|=(tactile&~self.both_seen)[fresh]
        self.goal[fresh]|=m['goal'][fresh]
        interception=self.body_blocked
        if getattr(env,'_anticipatory_receive',False):
            # User-authorized predictive receiving may intercept with hands
            # before a shin impact. Keep actual body contacts separate and true.
            interception=self.intercepted&self.descent_before_interception
        valid=m['valid']&~self.failed&self.motion_seen&self.travelled&interception
        self.hold[fresh]=torch.where(valid[fresh],self.hold[fresh]+1,0)
        self.peak_hold[fresh]=torch.maximum(self.peak_hold[fresh],self.hold[fresh])
        self.success[fresh]|=self.hold[fresh]>=math.ceil(2./env.step_dt-1e-9)
        self.stamp[fresh]=env.common_step_counter
        return self.success&~self.failed


def failure(env): return measurements(env)['failed']


def make_cfg(num_envs=24):
    cfg=static_cfg(num_envs)
    cfg.events['ball_reset']=EventTermCfg(func=launch,mode='reset')
    cfg.episode_length_s=30.
    cfg.terminations['pose_success']=TerminationTermCfg(func=IncomingOutcome)
    cfg.terminations['physical_failure']=TerminationTermCfg(func=failure)
    return cfg


class IncomingEnv(BalancedKeeperEnv):
    """Uses actual RGB for perception and only its predicted lane for walking."""
    descent_rate=1.
    def target_point(self,v,feet,visible):
        e=self.env
        approaching=visible&v.fit_ready&(v.velocity[:,0]<-.04)
        tta=((v.point[:,0]-.20)/(-v.velocity[:,0]).clamp_min(.04)).clamp(0,12)
        predicted_y=torch.where(approaching,v.point[:,1]+v.velocity[:,1]*tta,v.point[:,1])
        close=visible&v.fit_ready&(v.velocity[:,:2].norm(dim=1)<.06)&(v.point[:,0]<.40)
        foot_x=float(book(e)['feet'][:,0].mean())
        target_x=torch.where(close,(v.point[:,0]-.20+foot_x).clamp(-.08,.08),foot_x)
        return torch.stack((target_x,predicted_y.clamp(-1.07,1.07)),1)

    def collection_ready(self,v,feet,visible):
        relative=v.point-feet[:,:2]
        ready=visible&v.fit_ready&(v.velocity[:,:2].norm(dim=1)<.06)
        return ready&(relative[:,0]>.15)&(relative[:,0]<.25)&(relative[:,1].abs()<.04)

    def endpoint_weight(self):
        return torch.ones(self.num_envs,device=self.device)

    def forward_command(self,error,velocity):
        return (1.2*error[:,0]-.3*velocity[:,0]).clamp(-.08,.08)

    def adjust_motor_targets(self,target):
        return target

    def approach_lateral_command(self,error,velocity,actions):
        speed=.24+.04*actions[:,22]
        command=(2.6*(error[:,1]+.008*actions[:,23])-.45*velocity[:,1]).clamp(-speed,speed)
        return torch.sign(command)*torch.maximum(command.abs(),torch.full_like(command,.18))

    def step(self,actions):
        a=actions.clamp(-1,1); e=self.env
        base=self._base_observations(); d=e.scene['robot'].data; m=measurements(e)
        feet=m['feet'].mean(1)-e.scene.env_origins
        own=d.root_link_pos_w-e.scene.env_origins
        e._bridge_distance=torch.maximum(e._bridge_distance,(own[:,1]-e._bridge_start_y).abs())
        v=self.vision; visible=v.usable&(v.age<1.)
        # No shot reset parameters or ball truth enter the target or timing logic.
        # Hold the original guarding x-plane while a distant ball approaches.
        # A nearby stopped ball may request a small RGB-only fore/aft alignment.
        desired_point=self.target_point(v,feet,visible)
        tracking=self.target_tracking_mask(visible) if hasattr(self,"target_tracking_mask") else (e._bridge_stage<2)&visible
        e._bridge_target=torch.where(tracking[:,None],.75*e._bridge_target+.25*desired_point,e._bridge_target)
        if hasattr(self,"prepare_interception_update"): self.prepare_interception_update(feet,visible)
        error=e._bridge_target-feet[:,:2]; elapsed=e.episode_length_buf*e.step_dt
        moving=e._bridge_stage==0
        motion_ready=(e._bridge_distance>.08)|getattr(e,'_incoming_allow_stationary',False)
        arrived=moving&(elapsed>.8)&visible&(error[:,1].abs()<.025)&(error[:,0].abs()<.035)&motion_ready
        if hasattr(self,"arrival_gate"): arrived &= self.arrival_gate(feet,visible)
        e._bridge_stage[arrived]=1
        reacquire=(e._bridge_stage==1)&(e._bridge_quiet>.15)&visible&(error[:,1].abs()>.055)
        if hasattr(self,"reacquire_mask"): reacquire=self.reacquire_mask(error,visible)
        e._bridge_stage[reacquire]=0; e._bridge_quiet[reacquire]=0
        moving=e._bridge_stage==0
        lateral=self.approach_lateral_command(error,d.root_link_lin_vel_w,a)
        lateral=torch.where(moving&visible&(elapsed>.8),lateral,0.)
        forward=self.forward_command(error,d.root_link_lin_vel_w)
        forward=torch.where((e._bridge_stage<2)&visible,forward,0.)
        q=d.root_link_quat_w
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        cmd=torch.stack((torch.cos(yaw)*forward+torch.sin(yaw)*lateral,
            -torch.sin(yaw)*forward+torch.cos(yaw)*lateral,(-1.5*yaw).clamp(-.4,.4)),1)
        cmd=torch.where((e._bridge_stage<2)[:,None],cmd,0.)
        e._bridge_cmd+=(cmd-e._bridge_cmd).clamp(-.8*e.step_dt,.8*e.step_dt)
        foot_sep=(m['feet'][:,0,1]-m['feet'][:,1,1]).abs()
        fore_aft=(m['feet'][:,0,0]-m['feet'][:,1,0]).abs()
        foot_speed=d.body_link_lin_vel_w[:,e._collect_geometry[1],:2].norm(dim=-1).max(1).values
        quiet=(m['support']&(m['margin']>.012)&(m['foot_tilt']<.15)&(m['tilt']<.15)
            &(d.root_link_lin_vel_w[:,:2].norm(dim=1)<.045)&(d.root_link_ang_vel_w.norm(dim=1)<.30)
            &(e._bridge_cmd[:,:2].norm(dim=1)<.025)&(yaw.abs()<.08)
            &(foot_sep>.17)&(foot_sep<.30)&(fore_aft<.03)&(foot_speed<.04))
        braking=(e._bridge_stage>=1) if hasattr(self,"soft_brake_transition") else (e._bridge_stage==1)
        e._bridge_quiet=torch.where(braking&quiet,(e._bridge_quiet+e.step_dt).clamp(max=.6),
            torch.where(e._bridge_stage==2,e._bridge_quiet,0.))
        trigger=braking&(e._bridge_quiet>=.599)&visible&(error[:,1].abs()<.04)&(error[:,0].abs()<.035)
        e._bridge_width[trigger]=foot_sep[trigger]; e._bridge_anchor[trigger]=m['feet'][trigger]
        e._bridge_braked|=trigger
        if hasattr(self,"soft_brake_transition"):
            trigger=self.soft_brake_transition(m,d,visible,error,trigger)
            e._bridge_width[trigger]=foot_sep[trigger]; e._bridge_anchor[trigger]=m["feet"][trigger]
        e._bridge_stage[trigger]=2
        tactile=e.scene['body_ball_contact'].data.found.flatten(1).any(1)
        e._incoming_touched|=tactile
        # Measured robot stop comes first. Remain upright while body/feet block.
        # Begin descent only after tactile block AND a nearby slow RGB track.
        collect_ready=self.collection_ready(v,feet,visible)
        contact_gate=e._incoming_touched|getattr(e,'_anticipatory_receive',False)
        e._incoming_ready|=(e._bridge_stage==2)&contact_gate&collect_ready
        increment=e.step_dt
        if hasattr(self,'descent_clock_increment'):
            increment=self.descent_clock_increment()
        elif self.descent_rate!=1.:
            # Change only the motor-reference transition duration. Actual
            # physics time and the real 2-second hold counter remain unchanged.
            speeding=e._incoming_ready&(phase(e)<.9999)&(e._bridge_clock>=HANDOFF_SETTLE)
            increment=torch.where(speeding,e.step_dt*self.descent_rate,e.step_dt)
        next_clock=e._bridge_clock+increment
        next_clock=torch.where(e._incoming_ready,next_clock,next_clock.clamp(max=HANDOFF_SETTLE))
        e._bridge_clock=torch.where(e._bridge_stage==2,next_clock,0.)
        s=self.phase()
        ref=self.collection_reference(s) if hasattr(self,'collection_reference') else table(e).at(s)
        reference=ref[:,3:25][:,self.path_order]; ff=ref[:,25:47][:,self.path_order]
        entry_obs=self._base_observations()['actor'].clone()
        gait_obs=base['actor'][:,:75].clone(); gait_obs[:,72:75]=e._bridge_cmd
        with torch.no_grad():
            gaita=self.gait_teacher(TensorDict({'actor':gait_obs},batch_size=[self.num_envs]))
            endpoint_obs=entry_obs.clone(); b=book(e); ids=e._pose_ids
            endpoint_obs[:,83]=1.
            endpoint_obs[:,-3:]=torch.stack((b['knee'][ids],b['pitch'][ids],b['z'][ids]),1)
            enda=self.endpoint(TensorDict({'actor':endpoint_obs},batch_size=[self.num_envs])).clamp(-1,1)
        self.teacher_filtered=.8*self.teacher_filtered+.2*enda
        self.bridge_filter=.8*self.bridge_filter+.2*a[:,:22]
        collection=reference+ff+self.endpoint_weight()[:,None]*s[:,None]**10*self.amplitude*self.teacher_filtered
        gait=self.home+self.scale*gaita
        blend=(e._bridge_clock/HANDOFF_SETTLE).clamp(0,1); blend=blend*blend*(3-2*blend)
        target=(1-blend[:,None])*gait+blend[:,None]*collection+self.bridge_amplitude*self.bridge_filter
        if hasattr(self,'compose_motor_targets'):
            target=self.compose_motor_targets(gait,collection,self.bridge_amplitude*self.bridge_filter)
        target=self.adjust_motor_targets(target)
        _,_,done,extras=RslRlVecEnvWrapper.step(self,(target.clamp(self.low,self.high)-self.home)/self.scale)
        done=done.bool(); m=measurements(e); out=e._pose_outcome
        # Perfect ball data only below: reward, critic, termination and evaluation.
        aligned=torch.exp(-((m['ball_pos'][:,1]-own[:,1])/.08).square())
        late=phase(e)>.9; hand=m['contacts'].all(1)
        reward=e.step_dt*(-.4+aligned+.4*m['support'].float()+.5*(e._bridge_stage==2).float()
            +.4*(e._bridge_stage==2)*(m['margin']/.04).clamp(0,1)
            +late*m['contacts'].float().sum(1)+2*late*hand+4*m['valid'].float()
            -2*e._incoming_touched*(m['speed']/.15).clamp(max=3))*(~done)
        reward-=.008*(a-self.previous_bridge).square().mean(1)+.001*a.square().mean(1)
        reward+=4*trigger.float()*(~done)
        reward+=75*(done&out.last['success']).float()-30*(done&out.last['failed']).float()
        reward-=10*(done&e.termination_manager.get_term('time_out')&~out.last['success']).float()
        self.previous_bridge=a.clone(); self.previous_bridge[done]=0
        self.bridge_filter[done]=self.teacher_entry_filter[done]=self.teacher_filtered[done]=0
        ids=done.nonzero().flatten()
        if len(ids): self.vision.reset(ids)
        extras['log']={'Incoming/braked':e._bridge_braked.float(),'Incoming/phase':phase(e),
            'Incoming/ground_hold':m['valid'].float(),'Incoming/success_event':(done&out.last['success']).float()}
        return self.get_observations(),reward,done.long(),extras
