"""RGB lateral alignment, measured braking, then physical rolling-ball collection."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
from dataclasses import asdict
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=(BUNDLE_ROOT / 'runtime')
ROLL=PROJECT/'experiments/2026-10-19_slow_roll_collect_v3'
sys.path.insert(0,str(ROLL))
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.managers import EventTermCfg,TerminationTermCfg
from rolling import RollingEnv,RollingOutcome,make_cfg as rolling_cfg,measurements,HOLD_SECONDS,sha
from rolling import preserved as rolling_preserved
from train_rolling import config as rolling_config
from transition import TransitionEnv
from goalkeeper_sim.full_goal.outcomes import goal_crossed
from goalkeeper_sim.full_goal.settings import FullGoalSettings

ROLL_MODEL=ROLL/'runs/2026-10-19_02-37-23/model_final.pt'


def launch(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    if not hasattr(env,'_roll_phase'):
        for name in ('_roll_phase','_roll_clock','_roll_initial_x','_lat_target','_lat_quiet','_lat_distance'):
            setattr(env,name,torch.zeros(env.num_envs,device=env.device))
        env._roll_triggered=env._roll_phase.bool()
        env._lat_stage=env._roll_phase.long()
        env._lat_command=torch.zeros((env.num_envs,3),device=env.device)
        env._lat_handoff=env._roll_phase.bool()
        env._lat_shot_left=env._roll_phase.bool()
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0]=2.1+.3*torch.rand(len(ids),device=env.device)
    # Randomized starts, with balanced left/right evaluation lanes. These
    # labels are used for test accounting only, never by the controller.
    sign=torch.where(torch.rand(len(ids),device=env.device)<.5,-1.,1.)
    if getattr(env,'_balanced_eval',False): sign=torch.where(ids%2==0,-1.,1.)
    state[:,1]=sign*(.18+.08*torch.rand(len(ids),device=env.device))
    state[:,2],state[:,3]=.112,1.
    speed=.17+.04*torch.rand(len(ids),device=env.device)
    state[:,7]=-speed
    state[:,11]=-speed/.11
    env._roll_initial_x[ids]=state[:,0]
    env._lat_shot_left[ids]=sign>0
    state[:,:3]+=env.scene.env_origins[ids]
    env.scene['ball'].write_root_state_to_sim(state,env_ids=ids)
    for name in ('_roll_phase','_roll_clock','_lat_target','_lat_quiet','_lat_distance','_lat_stage'):
        getattr(env,name)[ids]=0
    env._lat_command[ids]=0
    env._roll_triggered[ids]=False
    env._lat_handoff[ids]=False


class LateralOutcome(RollingOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        for name in ('moved','braked','unsafe_handoff','left_shot','right_shot','fallen'):
            setattr(self,name,self.success.clone())
            self.last[name]=self.success.clone()

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        d=env.scene['robot'].data
        own=d.root_link_pos_w-env.scene.env_origins
        self.full_start[fresh]=env._start_fraction[fresh]<1e-6
        self.motion_seen[fresh]|=((m['speed']>.1)&(m['ball_pos'][:,0]>.5))[fresh]
        self.travelled[fresh]|=(env._roll_initial_x-m['ball_pos'][:,0]>.6)[fresh]
        self.moved[fresh]|=(own[:,1].abs()>.08)[fresh]
        self.braked[fresh]|=env._lat_handoff[fresh]
        self.unsafe_handoff[fresh]|=((env._roll_phase>0)&~env._lat_handoff)[fresh]
        self.left_shot[fresh]=env._lat_shot_left[fresh]
        self.right_shot[fresh]=~env._lat_shot_left[fresh]
        self.fallen[fresh]|=((m['height']<.25)|(d.projected_gravity_b[:,2]>-.56))[fresh]
        body=env.scene['body_ball_contact'].data.found.flatten(1).any(1)
        self.touched[fresh]|=(body|m['contacts'].any(1))[fresh]
        self.goal[fresh]|=goal_crossed(m['ball_pos'],FullGoalSettings())[fresh]
        self.escaped[fresh]|=(m['ball_pos'][:,0]<-.35)[fresh]
        self.failed[fresh]|=(m['failed']|self.goal|self.escaped)[fresh]
        self.left_seen[fresh]|=m['contacts'][fresh,0]
        self.right_seen[fresh]|=m['contacts'][fresh,1]
        self.both_seen[fresh]|=m['contacts'][fresh].all(1)
        valid=(m['valid']&~self.failed&self.motion_seen&self.travelled&self.touched
            &self.moved&self.braked&~self.unsafe_handoff&(env._roll_phase>.9999))
        self.hold[fresh]=torch.where(valid[fresh],self.hold[fresh]+1,0)
        self.peak_hold[fresh]=torch.maximum(self.peak_hold[fresh],self.hold[fresh])
        self.success[fresh]|=self.hold[fresh]>=math.ceil(HOLD_SECONDS/env.step_dt-1e-9)
        self.stamp[fresh]=env.common_step_counter
        return self.success&~self.failed


def make_cfg(num_envs=16):
    cfg=rolling_cfg(num_envs)
    cfg.events['rolling_ball']=EventTermCfg(func=launch,mode='reset')
    cfg.episode_length_s=22.
    cfg.terminations['collection_success']=TerminationTermCfg(func=LateralOutcome)
    return cfg


class LateralEnv(RollingEnv):
    def __init__(self,env):
        self.lateral_ready=False
        super().__init__(env)
        c=asdict(rolling_config())['actor']
        args={k:c[k] for k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
        self.rolling_teacher=MLPModel(self.roll_obs,{'actor':['actor']},'actor',24,**args).to(env.device)
        self.rolling_teacher.load_state_dict(torch.load(ROLL_MODEL,map_location=env.device,weights_only=False)['actor_state_dict'])
        self.rolling_teacher.eval().requires_grad_(False)
        self.previous_lateral=torch.zeros((env.num_envs,26),device=env.device)
        self.num_actions=26
        self.lateral_ready=True
        self.get_observations()

    def get_observations(self):
        base=super().get_observations()
        if not self.lateral_ready: return base
        d=self.env.scene['robot'].data
        own=d.root_link_pos_w-self.env.scene.env_origins
        feet=measurements(self.env)['feet'].mean(1)-self.env.scene.env_origins
        base['actor'][:,72:75]=self.env._lat_command
        stages=torch.nn.functional.one_hot(self.env._lat_stage,4).float()
        extra=torch.cat((self.env._lat_command,stages,
            (self.env._lat_target-feet[:,1])[:,None],self.env._lat_quiet[:,None]/.4,
            d.root_link_lin_vel_w[:,:2]),1)
        actor=torch.cat((base['actor'],extra),1)
        self.lat_obs=TensorDict({'actor':actor,'critic':torch.cat((actor,base['critic'][:,-6:]),1)},batch_size=[self.num_envs])
        return self.lat_obs

    def reset(self):
        result=super().reset()
        if hasattr(self,'previous_lateral'): self.previous_lateral.zero_()
        return result

    def step(self,actions):
        a=actions.clamp(-1,1)
        e=self.env
        d=e.scene['robot'].data
        own=d.root_link_pos_w-e.scene.env_origins
        v=self.vision
        tta,approaching=self.estimated_tta()
        m=measurements(e)
        # RGB-ground-plane trajectory fit; proprioceptive pose supplies the
        # camera transform. No ball truth enters this target or state machine.
        # This curriculum explicitly uses straight negative-X shots. Their
        # lateral lane is measured from RGB directly: extrapolating tiny noisy
        # vy estimates over 10 seconds otherwise creates false target swings.
        # Diagonal shots require a separately validated trajectory estimator.
        observed=v.point[:,1].clamp(-.38,.38)
        tracking=(e._lat_stage<2)&v.usable
        e._lat_target=torch.where(tracking,.85*e._lat_target+.15*observed,e._lat_target)
        # A lateral gait leans its torso ahead of the support center. Stopping
        # when the leaning root reaches the lane makes the settled body fall
        # short. Align the support midpoint instead, using foot kinematics.
        feet_mid=m['feet'].mean(1)-e.scene.env_origins
        error=e._lat_target+.015*a[:,22]-feet_mid[:,1]
        elapsed=e.episode_length_buf*e.step_dt
        moving=e._lat_stage==0
        arrive=moving&(elapsed>.6)&v.fit_ready&(error.abs()<.018)
        e._lat_stage[arrive]=1
        reacquire=(e._lat_stage==1)&(error.abs()>.045)&v.usable
        e._lat_stage[reacquire]=0
        moving=e._lat_stage==0
        speed=(.28+.08*a[:,23])
        lateral=(2.5*error-.5*d.root_link_lin_vel_w[:,1]).clamp(-speed,speed)
        lateral=torch.where(moving&v.usable&(elapsed>.6),lateral,0.)
        q=d.root_link_quat_w
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        forward=(-.8*own[:,0]-.3*d.root_link_lin_vel_w[:,0]).clamp(-.10,.10)
        omega=(-1.5*yaw).clamp(-.4,.4)
        desired=torch.stack((torch.cos(yaw)*forward+torch.sin(yaw)*lateral,
            -torch.sin(yaw)*forward+torch.cos(yaw)*lateral,omega),1)
        desired=torch.where((e._lat_stage<2)[:,None],desired,0.)
        e._lat_command+=(desired-e._lat_command).clamp(-.8*e.step_dt,.8*e.step_dt)
        # Measured stop gate, continuously satisfied for 0.4 seconds. The
        # actuator command is not accepted as proof that the robot has stopped.
        quiet=(m['support']&(m['com_error']<.05)&(m['tilt']<.15)
            &(d.root_link_lin_vel_w[:,:2].norm(dim=1)<.045)
            &(d.root_link_ang_vel_w.norm(dim=1)<.3)
            &(e._lat_command[:,:2].norm(dim=1)<.03))
        braking=e._lat_stage==1
        e._lat_quiet=torch.where(braking&quiet,e._lat_quiet+e.step_dt,0.)
        ready=braking&(e._lat_quiet>=.4)&(error.abs()<.045)
        e._lat_stage[ready]=2
        e._lat_handoff|=ready
        # If the standing robot drifts before descent, brake again; do not
        # inherit an old quiet flag as permission to bend while moving.
        drift=(e._lat_stage==2)&~quiet
        e._lat_stage[drift]=1
        e._lat_handoff[drift]=False
        base=self.roll_obs['actor'].clone()
        base[:,72:75]=0.
        with torch.no_grad():
            old=self.rolling_teacher(TensorDict({'actor':base},batch_size=[self.num_envs])).clamp(-1,1)
        threshold=4.1+.4*old[:,22]+.3*a[:,24]
        trigger=(e._lat_stage==2)&quiet&v.usable&approaching&(tta<threshold)
        e._lat_stage[trigger]=3
        e._roll_triggered|=trigger
        advance=e.step_dt/4*(1+.12*old[:,23]+.08*a[:,25])
        e._roll_clock=(e._roll_clock+advance*e._roll_triggered).clamp(0,1)
        c=e._roll_clock
        e._roll_phase=c**3*(10-15*c+6*c.square())
        prior=base[:,:92].clone()
        prior[:,83]=e._roll_phase
        # Translation-invariant context for the frozen posture primitive:
        # the local collection center follows the settled robot's lateral lane.
        prior[:,75]=.20-own[:,0]
        prior[:,76:80]=0.
        prior[:,80]=1.
        prior[:,72:75]=e._lat_command
        with torch.no_grad():
            posture=self.standing_teacher(TensorDict({'actor':prior},batch_size=[self.num_envs])).clamp(-1,1)
        blend=(e._roll_phase/.05).clamp(0,1)[:,None]
        combined=(blend*posture+blend*(.10/.15)*old[:,:22]
            +blend*(.04/.15)*a[:,:22]).clamp(-1,1)
        self.current_obs=TensorDict({'actor':prior,'critic':self.roll_obs['critic'][:,:98]},batch_size=[self.num_envs])
        obs,_,done,extras=TransitionEnv.step(self,combined)
        done=done.bool()
        m=measurements(e)
        out=e._rolling
        own2=e.scene['robot'].data.root_link_pos_w-e.scene.env_origins
        # Truth is used only to supervise alignment and evaluate physical saves.
        true_error=m['ball_pos'][:,1]-own2[:,1]
        aligned=torch.exp(-(true_error/.07).square())
        center=torch.exp(-(m['com_error']/.07).square())
        late=self.phase()>.9
        reach=torch.exp(-(m['distances']-.135).abs().sum(1)/.15)*late
        both=m['contacts'].all(1)
        values=self.path.at(self.phase())
        angle=e.scene['robot'].data.joint_pos[:,self.term.target_ids]
        track=torch.exp(-((angle-values[:,3:25][:,self.path_order])/.25).square().mean(1))
        height=torch.exp(-((m['height']-values[:,1]-.005)/.07).square())
        tilt=torch.exp(-((m['tilt']-values[:,2])/.18).square())
        reward=e.step_dt*(1.+m['support'].float()+center+3*aligned+track+height+tilt
            +2*reach+4*both+6*m['valid'].float()
            -.7*out.touched*(m['speed']/.15).clamp(max=4)
            -2*((m['ball_pos'][:,2]-.14)/.05).clamp(0,4))*(~done)
        reward-=.008*(a-self.previous_lateral).square().mean(1)+.001*a.square().mean(1)
        reward+=25*(done&out.last['success']).float()-12*(done&out.last['failed']).float()
        self.previous_lateral=a.clone()
        self.previous_lateral[done]=0
        extras['log']={'Lateral/alignment':aligned,'Lateral/stopped_before_crouch':e._lat_handoff.float(),
            'Lateral/collect_phase':self.phase(),'Lateral/valid_hold':m['valid'].float(),
            'Lateral/success_event':(done&out.last['success']).float()}
        return obs,reward,done.long(),extras


def preserved():
    return bundle_integrity()
