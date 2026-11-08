"""Vision-timed frontal slow rolling-ball collection; old motor skills frozen."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=(BUNDLE_ROOT / 'runtime')
STAND=PROJECT/'experiments/2026-10-19_standing_collect_v3'
sys.path.insert(0,str(STAND))
import torch
from tensordict import TensorDict
from mjlab.managers import EventTermCfg,TerminationTermCfg
from transition import TransitionEnv,CollectOutcome,make_cfg as standing_cfg,measurements,model,sha
from transition import preserved as standing_preserved
from train_transition import config as standing_config
from goalkeeper_sim.full_goal.outcomes import goal_crossed
from goalkeeper_sim.full_goal.settings import FullGoalSettings

STAND_MODEL=STAND/'runs/2026-10-19_01-29-17/model_final.pt'
HOLD_SECONDS=2.


def launch_rolling_ball(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    if not hasattr(env,'_roll_phase'):
        env._roll_phase=torch.zeros(env.num_envs,device=env.device)
        env._roll_clock=env._roll_phase.clone()
        env._roll_triggered=env._roll_phase.bool()
        env._roll_initial_x=env._roll_phase.clone()
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0]=1.4+.3*torch.rand(len(ids),device=env.device)
    state[:,1]=(torch.rand(len(ids),device=env.device)-.5)*.012
    state[:,2],state[:,3]=.112,1.
    speed=.18+.06*torch.rand(len(ids),device=env.device)
    state[:,7]=-speed
    state[:,11]=-speed/.11
    env._roll_initial_x[ids]=state[:,0]
    state[:,:3]+=env.scene.env_origins[ids]
    env.scene['ball'].write_root_state_to_sim(state,env_ids=ids)
    env._roll_phase[ids]=env._roll_clock[ids]=0.
    env._roll_triggered[ids]=False


class RollingOutcome(CollectOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        for n in ('full_start','motion_seen','travelled','touched','goal','escaped'):
            setattr(self,n,self.success.clone())
            self.last[n]=self.success.clone()
        self.peak_hold=self.hold.clone()
        self.last_peak_hold=self.hold.clone()
        env._rolling=self

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.last_peak_hold[ids]=self.peak_hold[ids]
        self.peak_hold[ids]=0
        super().reset(env_ids)

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        self.full_start[fresh]=env._start_fraction[fresh]<1e-6
        self.motion_seen[fresh]|=((m['speed']>.1)&(m['ball_pos'][:,0]>.5))[fresh]
        self.travelled[fresh]|=(env._roll_initial_x-m['ball_pos'][:,0]>.6)[fresh]
        body=env.scene['body_ball_contact'].data.found.flatten(1).any(1)
        self.touched[fresh]|=(body|m['contacts'].any(1))[fresh]
        self.goal[fresh]|=goal_crossed(m['ball_pos'],FullGoalSettings())[fresh]
        self.escaped[fresh]|=(m['ball_pos'][:,0]<-.35)[fresh]
        self.failed[fresh]|=(m['failed']|self.goal|self.escaped)[fresh]
        self.left_seen[fresh]|=m['contacts'][fresh,0]
        self.right_seen[fresh]|=m['contacts'][fresh,1]
        self.both_seen[fresh]|=m['contacts'][fresh].all(1)
        valid=(m['valid']&~self.failed&self.motion_seen&self.travelled&self.touched
            &(env._roll_phase>.9999))
        self.hold[fresh]=torch.where(valid[fresh],self.hold[fresh]+1,0)
        self.peak_hold[fresh]=torch.maximum(self.peak_hold[fresh],self.hold[fresh])
        self.success[fresh]|=self.hold[fresh]>=math.ceil(HOLD_SECONDS/env.step_dt-1e-9)
        self.stamp[fresh]=env.common_step_counter
        return self.success&~self.failed


def escaped(env): return env._rolling.escaped|env._rolling.goal


def make_cfg(num_envs=16):
    cfg=standing_cfg(num_envs)
    cfg.events.pop('static_ball')
    cfg.events['rolling_ball']=EventTermCfg(func=launch_rolling_ball,mode='reset')
    cfg.episode_length_s=16.
    cfg.terminations['collection_success']=TerminationTermCfg(func=RollingOutcome)
    cfg.terminations['escaped_ball']=TerminationTermCfg(func=escaped)
    return cfg


class RollingEnv(TransitionEnv):
    def __init__(self,env):
        self.rolling_ready=False
        super().__init__(env)
        self.standing_teacher=model(self.current_obs,STAND_MODEL,asdict(standing_config())['actor'],env.device)
        self.previous_new=torch.zeros((env.num_envs,24),device=env.device)
        self.num_actions=24
        self.rolling_ready=True
        self.roll_obs=self.get_observations()

    def phase(self): return self.env._roll_phase

    def estimated_tta(self):
        v=self.vision
        approaching=v.usable&v.fit_ready&(v.velocity[:,0]<-.03)
        tta=torch.where(approaching,(v.point[:,0]-.20)/(-v.velocity[:,0]).clamp_min(.03),12.)
        return tta.clamp(0,12),approaching

    def get_observations(self):
        base=super().get_observations()
        if not self.rolling_ready: return base
        tta,_=self.estimated_tta()
        extra=torch.stack((tta/8,self.env._roll_triggered.float()),1)
        actor=torch.cat((base['actor'],extra),1)
        critic=torch.cat((actor,base['critic'][:,-6:]),1)
        self.roll_obs=TensorDict({'actor':actor,'critic':critic},batch_size=[self.num_envs])
        return self.roll_obs

    def reset(self):
        result=super().reset()
        if hasattr(self,'previous_new'): self.previous_new.zero_()
        return result

    def step(self,actions):
        a=actions.clamp(-1,1)
        tta,approaching=self.estimated_tta()
        v=self.vision
        settled=self.env.episode_length_buf*self.env.step_dt>.6
        threshold=4.1+.4*a[:,22]
        # Both trigger branches use only RGB-derived estimates. Shot launch
        # speed/location and ball truth are not available to this controller.
        trigger=settled&v.usable&((approaching&(tta<threshold))|(v.point[:,0]<.65))
        self.env._roll_triggered|=trigger
        advance=self.env.step_dt/4.*(1.+.12*a[:,23])
        self.env._roll_clock=(self.env._roll_clock+advance*self.env._roll_triggered).clamp(0,1)
        c=self.env._roll_clock
        self.env._roll_phase=c**3*(10-15*c+6*c.square())
        baseactor=self.roll_obs['actor'][:,:92].clone()
        baseactor[:,83]=self.env._roll_phase
        # Frozen standing-to-crouch motor primitive is conditioned on its fixed
        # collection target, not a fabricated measurement of the moving ball.
        # Actual RGB remains in the new learner and the trigger above.
        own=self.env.scene['robot'].data.root_link_pos_w-self.env.scene.env_origins
        prior=baseactor.clone()
        prior[:,75:77]=prior.new_tensor((.20,0.))-own[:,:2]
        prior[:,77:80]=0.
        prior[:,80]=1.
        with torch.no_grad():
            old=self.standing_teacher(TensorDict({'actor':prior},batch_size=[self.num_envs])).clamp(-1,1)
        primitive_weight=(self.env._roll_phase/.05).clamp(0,1)[:,None]
        combined=(primitive_weight*old+(.10/.15)*a[:,:22]).clamp(-1,1)
        self.current_obs=TensorDict({'actor':prior,'critic':self.roll_obs['critic'][:,:98]},batch_size=[self.num_envs])
        obs,_,done,extras=super().step(combined)
        done=done.bool()
        m=measurements(self.env)
        out=self.env._rolling
        values=self.path.at(self.phase())
        root=self.env.scene['robot'].data
        angle=root.joint_pos[:,self.term.target_ids]
        ref=values[:,3:25][:,self.path_order]
        track=torch.exp(-((angle-ref)/.25).square().mean(1))
        center=torch.exp(-(m['com_error']/.07).square())
        height=torch.exp(-((m['height']-values[:,1]-.005)/.07).square())
        tilt=torch.exp(-((m['tilt']-values[:,2])/.18).square())
        late=self.phase()>.9
        reach=torch.exp(-(m['distances']-.135).abs().sum(1)/.15)*late
        both=m['contacts'].all(1)
        reward=self.env.step_dt*(1.+m['support'].float()+center+track+height+tilt
            +2*reach+4*both+6*m['valid'].float()
            -.7*out.touched*(m['speed']/.15).clamp(max=4)
            -2*((m['ball_pos'][:,2]-.14)/.05).clamp(0,4))*(~done)
        reward-=.015*(a-self.previous_new).square().mean(1)+.002*a.square().mean(1)
        reward+=20*(done&out.last['success']).float()-12*(done&out.last['failed']).float()
        self.previous_new=a.clone()
        self.previous_new[done]=0
        extras['log']={'Rolling/supported':m['support'].float(),'Rolling/triggered':self.env._roll_triggered.float(),
            'Rolling/bilateral_contact':both.float(),'Rolling/valid_hold':m['valid'].float(),
            'Rolling/hold_seconds':out.hold*self.env.step_dt,
            'Rolling/success_event':(done&out.last['success']).float()}
        return obs,reward,done.long(),extras


def preserved():
    return bundle_integrity()
