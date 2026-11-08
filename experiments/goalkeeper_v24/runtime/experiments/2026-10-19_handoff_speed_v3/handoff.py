"""Ground collection after lateral stop, with a gated faster-shot curriculum."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=(BUNDLE_ROOT / 'runtime')
OLD=PROJECT/'experiments/2026-10-19_lateral_collect_v3'
sys.path.insert(0,str(OLD))
import torch
from tensordict import TensorDict
from mjlab.managers import EventTermCfg,TerminationTermCfg
from lateral import LateralEnv,LateralOutcome,make_cfg as old_cfg,measurements,sha
from lateral import preserved as old_preserved
from transition import TransitionEnv

OLD_MODEL=OLD/'runs/2026-10-19_14-05-38/model_final.pt'
SHOTS={'bridge':dict(x=(2.4,2.6),speed=(.22,.26),lateral=(.20,.20)),
       'faster':dict(x=(3.,3.4),speed=(.30,.40),lateral=(.18,.26))}


def launch(env,env_ids,mode):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    env._shot_mode=mode
    if not hasattr(env,'_roll_phase'):
        for name in ('_roll_phase','_roll_clock','_roll_initial_x','_lat_target','_lat_quiet',
                     '_lat_distance','_bridge_vx','_bridge_ready_time'):
            setattr(env,name,torch.zeros(env.num_envs,device=env.device))
        env._roll_triggered=env._roll_phase.bool()
        env._lat_stage=env._roll_phase.long()
        env._lat_command=torch.zeros((env.num_envs,3),device=env.device)
        env._lat_handoff=env._roll_phase.bool()
        env._lat_shot_left=env._roll_phase.bool()
        env._bridge_motor=torch.zeros((env.num_envs,22),device=env.device)
    settings=SHOTS[mode]
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0]=settings['x'][0]+(settings['x'][1]-settings['x'][0])*torch.rand(len(ids),device=env.device)
    sign=torch.where(torch.rand(len(ids),device=env.device)<.5,-1.,1.)
    if getattr(env,'_balanced_eval',False): sign=torch.where(ids%2==0,-1.,1.)
    lo,hi=settings['lateral']
    state[:,1]=sign*(lo+(hi-lo)*torch.rand(len(ids),device=env.device))
    state[:,2],state[:,3]=.112,1.
    lo,hi=settings['speed']
    speed=lo+(hi-lo)*torch.rand(len(ids),device=env.device)
    state[:,7]=-speed
    state[:,11]=-speed/.11
    env._roll_initial_x[ids]=state[:,0]
    env._lat_shot_left[ids]=sign>0
    state[:,:3]+=env.scene.env_origins[ids]
    env.scene['ball'].write_root_state_to_sim(state,env_ids=ids)
    for name in ('_roll_phase','_roll_clock','_lat_target','_lat_quiet','_lat_distance',
                 '_lat_stage','_bridge_vx','_bridge_ready_time'):
        getattr(env,name)[ids]=0
    env._lat_command[ids]=env._bridge_motor[ids]=0
    env._roll_triggered[ids]=env._lat_handoff[ids]=False
    env._bridge_est_stamp=-1


class HandoffOutcome(LateralOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        for name in ('crouch_started','crouch_finished'):
            setattr(self,name,self.success.clone())
            self.last[name]=self.success.clone()

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        result=super().__call__(env)
        self.crouch_started[fresh]|=env._roll_triggered[fresh]
        self.crouch_finished[fresh]|=env._roll_phase[fresh]>.9999
        return result


def make_cfg(num_envs=16,mode='bridge'):
    cfg=old_cfg(num_envs)
    cfg.events['rolling_ball']=EventTermCfg(func=launch,mode='reset',params={'mode':mode})
    cfg.episode_length_s=20.
    cfg.terminations['collection_success']=TerminationTermCfg(func=HandoffOutcome)
    return cfg


class HandoffEnv(LateralEnv):
    def estimated_tta(self):
        e=self.env
        v=self.vision
        if e._bridge_est_stamp!=e.common_step_counter:
            reliable=v.usable&v.fit_ready&(v.velocity[:,0]<-.06)
            # Cache only image-derived approaching velocity. A brief fit dropout
            # must not replace a valid approach estimate with the constant 12s.
            first=reliable&(e._bridge_vx>=-.03)
            e._bridge_vx=torch.where(first,v.velocity[:,0],e._bridge_vx)
            e._bridge_vx=torch.where(reliable,.94*e._bridge_vx+.06*v.velocity[:,0],e._bridge_vx)
            e._bridge_est_stamp=e.common_step_counter
        approaching=v.usable&(e._bridge_vx<-.04)
        tta=torch.where(approaching,(v.point[:,0]-.20)/(-e._bridge_vx).clamp_min(.04),12.)
        return tta.clamp(0,12),approaching

    def step(self,actions):
        a=actions.clamp(-1,1)
        e=self.env
        d=e.scene['robot'].data
        own=d.root_link_pos_w-e.scene.env_origins
        v=self.vision
        m=measurements(e)
        out=e._rolling
        flags=('braked','crouch_finished','left_seen','right_seen','both_seen')
        before={name:getattr(out,name).clone() for name in flags}
        old_phase=self.phase().clone()
        tta,approaching=self.estimated_tta()
        elapsed=e.episode_length_buf*e.step_dt
        observed=v.point[:,1].clamp(-.38,.38)
        tracking=(e._lat_stage<2)&v.usable
        e._lat_target=torch.where(tracking,.85*e._lat_target+.15*observed,e._lat_target)
        feet=m['feet'].mean(1)-e.scene.env_origins
        error=e._lat_target+.02*a[:,22]-feet[:,1]
        body_error=e._lat_target-own[:,1]
        moving=e._lat_stage==0
        arrive=moving&(elapsed>.6)&v.usable&(error.abs()<.030)
        e._lat_stage[arrive]=1
        # Only re-enter walking after the braking transient. Never chase the
        # torso lean while the supporting feet are already close to the lane.
        reacquire=(e._lat_stage==1)&(e._lat_quiet>.10)&(body_error.abs()>.055)&v.usable&(tta>4.8)
        e._lat_stage[reacquire]=0
        moving=e._lat_stage==0
        speed=.30+.06*a[:,23]
        lateral=(2.4*error-.45*d.root_link_lin_vel_w[:,1]).clamp(-speed,speed)
        # The accepted gait was trained at about .18-.38m/s. Tiny commands
        # caused a standing lean without finishing the last step in the old FSM.
        lateral=torch.sign(error)*torch.maximum(lateral.abs(),torch.full_like(error,.18))
        lateral=torch.where(moving&v.usable&(elapsed>.6),lateral,0.)
        q=d.root_link_quat_w
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        forward=(-.8*own[:,0]-.3*d.root_link_lin_vel_w[:,0]).clamp(-.10,.10)
        desired=torch.stack((torch.cos(yaw)*forward+torch.sin(yaw)*lateral,
            -torch.sin(yaw)*forward+torch.cos(yaw)*lateral,(-1.5*yaw).clamp(-.4,.4)),1)
        desired=torch.where((e._lat_stage<2)[:,None],desired,0.)
        e._lat_command+=(desired-e._lat_command).clamp(-.8*e.step_dt,.8*e.step_dt)
        quiet=(m['support']&(m['com_error']<.05)&(m['tilt']<.15)
            &(d.root_link_lin_vel_w[:,:2].norm(dim=1)<.05)
            &(d.root_link_ang_vel_w.norm(dim=1)<.35)
            &(e._lat_command[:,:2].norm(dim=1)<.03))
        braking=e._lat_stage==1
        e._lat_quiet=torch.where(braking&quiet,(e._lat_quiet+e.step_dt).clamp(max=.4),
            torch.where(e._lat_stage>=2,e._lat_quiet,0.))
        ready=braking&(e._lat_quiet>=.399)&(body_error.abs()<.060)
        e._lat_stage[ready]=2
        e._lat_handoff|=ready
        e._bridge_ready_time[ready]=elapsed[ready]
        drift=(e._lat_stage==2)&~quiet
        e._lat_stage[drift]=1
        e._lat_handoff[drift]=False
        e._lat_quiet[drift]=0
        base=self.roll_obs['actor'].clone()
        base[:,72:75]=0.
        with torch.no_grad():
            old=self.rolling_teacher(TensorDict({'actor':base},batch_size=[self.num_envs])).clamp(-1,1)
        threshold=4.8+.3*a[:,24]
        # A visible near-ball fallback uses RGB distance rather than a perfect
        # velocity fit. Descent is still gated by measured physical stability.
        near=v.usable&(v.point[:,0]>.05)&(v.point[:,0]<2.0)
        timing=(approaching&(tta<threshold))|near
        trigger=(e._lat_stage==2)&quiet&v.usable&timing
        e._lat_stage[trigger]=3
        e._roll_triggered|=trigger
        advance=e.step_dt/4*(1+.10*old[:,23]+.10*a[:,25])
        e._roll_clock=(e._roll_clock+advance*e._roll_triggered).clamp(0,1)
        c=e._roll_clock
        e._roll_phase=c**3*(10-15*c+6*c.square())
        prior=base[:,:92].clone()
        prior[:,83]=e._roll_phase
        prior[:,75]=.20-own[:,0]
        prior[:,76:80]=0.
        prior[:,80]=1.
        prior[:,72:75]=e._lat_command
        with torch.no_grad():
            posture=self.standing_teacher(TensorDict({'actor':prior},batch_size=[self.num_envs])).clamp(-1,1)
        blend=(e._roll_phase/.05).clamp(0,1)[:,None]
        stopped=((e._lat_stage>=1)&(d.root_link_lin_vel_w[:,:2].norm(dim=1)<.08)).float()[:,None]
        correction=(.060*stopped+.090*blend)*a[:,:22]
        e._bridge_motor=.8*e._bridge_motor+.2*correction
        combined=(blend*posture+blend*(.10/.15)*old[:,:22]+e._bridge_motor/.15).clamp(-1,1)
        self.current_obs=TensorDict({'actor':prior,'critic':self.roll_obs['critic'][:,:98]},batch_size=[self.num_envs])
        obs,_,done,extras=TransitionEnv.step(self,combined)
        done=done.bool()
        m=measurements(e)
        d=e.scene['robot'].data
        own=d.root_link_pos_w-e.scene.env_origins
        aligned=torch.exp(-((m['ball_pos'][:,1]-own[:,1])/.06).square())
        center=torch.exp(-(m['com_error']/.06).square())
        values=self.path.at(self.phase())
        angles=d.joint_pos[:,self.term.target_ids]
        track=torch.exp(-((angles-values[:,3:25][:,self.path_order])/.25).square().mean(1))
        tilt=torch.exp(-((m['tilt']-values[:,2])/.18).square())
        support=m['support'].float()
        late=self.phase()>.90
        both=m['contacts'].all(1)
        reach=torch.exp(-(m['distances']-.135).abs().sum(1)/.12)*late
        # Completion is worth more than any entire timeout's living reward.
        # One-time milestones cannot be farmed by switching states repeatedly.
        reward=e.step_dt*(-.6+.3*aligned+.2*center+.2*support+.2*track+.1*tilt
            +1.5*reach+2*both+4*m['valid'].float()
            -1.*out.touched*(m['speed']/.15).clamp(max=4))*(~done)
        bonuses={'braked':3.,'crouch_finished':5.,'left_seen':2.,'right_seen':2.,'both_seen':5.}
        for name in flags:
            now=torch.where(done,out.last[name],getattr(out,name))
            reward+=bonuses[name]*(now&~before[name]).float()
        reward+=2*(self.phase()-old_phase).clamp(min=0)*(~done)
        reward-=.005*(a-self.previous_lateral).square().mean(1)+.001*a.square().mean(1)
        reward+=75*(done&out.last['success']).float()-30*(done&out.last['failed']).float()
        timeout=e.termination_manager.get_term('time_out')
        reward-=10*(done&timeout&~out.last['success']).float()
        self.previous_lateral=a.clone()
        self.previous_lateral[done]=0
        extras['log']={'Handoff/alignment':aligned,'Handoff/stopped':e._lat_handoff.float(),
            'Handoff/phase':self.phase(),'Handoff/bilateral_contact':both.float(),
            'Handoff/valid_hold':m['valid'].float(),'Handoff/success_event':(done&out.last['success']).float()}
        return obs,reward,done.long(),extras


def preserved():
    return bundle_integrity()
