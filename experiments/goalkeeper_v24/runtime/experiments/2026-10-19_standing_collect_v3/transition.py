"""Standing-to-collection curriculum with frozen endpoint skills and learned residuals."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=(BUNDLE_ROOT / 'runtime')
LOW=PROJECT/'experiments/2026-10-19_ground_collect_v3'
V2=PROJECT/'experiments/2026-10-18_natural_lateral_v2'
sys.path.insert(0,str(LOW))
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.managers import EventTermCfg,TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.models import MLPModel
from booster_mjlab.robots.booster_k1.k1_constants import HOME_KEYFRAME
from collection import CollectionEnv,CollectOutcome,make_cfg as low_cfg,measurements
from learn import config as low_config,preserved as original_preserved,sha

LOW_MODEL=LOW/'runs/2026-10-19_00-56-18/model_final.pt'
GAIT_MODEL=V2/'runs/2026-10-18_15-55-55/model_final.pt'
DURATION=4.
SETTLE=.6


class PathTable:
    def __init__(self,env):
        source=json.loads((HERE/'path.json').read_text())
        self.names=source['joint_names']
        self.table=torch.tensor([[f['root_x'],f['root_z'],f['root_pitch'],
            *f['joint_pos'],*f['gravity_offset']] for f in source['frames']],device=env.device)
        self.xml=[self.names.index(n) for n in env.scene['robot'].joint_names]
        self.feet=torch.tensor(source['feet_reference'],device=env.device)
        self.feet[:,2]+=.005

    def at(self,fraction):
        v=fraction.clamp(0,1)*(len(self.table)-1)
        i=v.long().clamp(max=len(self.table)-2)
        w=(v-i)[:,None]
        return self.table[i]*(1-w)+self.table[i+1]*w


def table(env):
    if not hasattr(env,'_transition_table'): env._transition_table=PathTable(env)
    return env._transition_table


def reset_transition(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    if not hasattr(env,'_start_fraction'):
        env._start_fraction=torch.zeros(env.num_envs,device=env.device)
    f=torch.zeros(len(ids),device=env.device)
    if getattr(env,'_allow_partial_starts',False):
        partial=torch.rand(len(ids),device=env.device)<.25
        f[partial]=.3+.55*torch.rand(int(partial.sum()),device=env.device)
    env._start_fraction[ids]=f
    p=table(env).at(f)
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0],state[:,2]=p[:,0],p[:,1]+.005
    state[:,3],state[:,5]=torch.cos(p[:,2]/2),torch.sin(p[:,2]/2)
    state[:,:3]+=env.scene.env_origins[ids]
    robot=env.scene['robot']
    robot.write_root_state_to_sim(state,env_ids=ids)
    q=p[:,3:25][:,table(env).xml]
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)


def phase(env):
    t=env.episode_length_buf*env.step_dt
    linear=((t-SETTLE)/DURATION).clamp(0,1)
    smooth=linear**3*(10-15*linear+6*linear.square())
    return env._start_fraction+(1-env._start_fraction)*smooth


class TransitionOutcome(CollectOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.full_start=self.success.clone()
        self.last['full_start']=self.success.clone()

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        self.full_start[fresh]=env._start_fraction[fresh]<1e-6
        self.failed[fresh]|=m['failed'][fresh]
        self.left_seen[fresh]|=m['contacts'][fresh,0]
        self.right_seen[fresh]|=m['contacts'][fresh,1]
        self.both_seen[fresh]|=m['contacts'][fresh].all(1)
        # Every evaluation starts standing; neither early contact nor partial
        # crouch initialization can count as a completed standing transition.
        ready=(phase(env)>.9999)&(env.episode_length_buf*env.step_dt>SETTLE+DURATION+.2)
        valid=m['valid']&~self.failed&ready
        self.hold[fresh]=torch.where(valid[fresh],self.hold[fresh]+1,0)
        self.success[fresh]|=self.hold[fresh]>=math.ceil(.6/env.step_dt-1e-9)
        self.stamp[fresh]=env.common_step_counter
        return self.success&~self.failed


def make_cfg(num_envs=16):
    cfg=low_cfg(num_envs)
    cfg.scene.entities['robot'].init_state=copy.deepcopy(HOME_KEYFRAME)
    cfg.events.pop('reset_base')
    cfg.events.pop('reset_robot_joints')
    # Keep ball/goal assets and all collision/sensor settings from low skill.
    cfg.events={'transition_start':EventTermCfg(func=reset_transition,mode='reset'),**cfg.events}
    cfg.episode_length_s=10.
    cfg.terminations['collection_success']=TerminationTermCfg(func=TransitionOutcome)
    return cfg


def model(obs,checkpoint,actor_cfg,device):
    args={k:actor_cfg[k] for k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
    actor=MLPModel(obs,{'actor':['actor']},'actor',22,**args).to(device)
    actor.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['actor_state_dict'])
    return actor.eval().requires_grad_(False)


class TransitionEnv(CollectionEnv):
    def __init__(self,env):
        super().__init__(env)
        self.path=table(env)
        self.path_order=[self.path.names.index(n) for n in self.action_names]
        p=json.loads((LOW/'poses.json').read_text())
        pose=p['poses'][p['selected']]['joint_pos']
        self.low_xml=torch.tensor([pose[n] for n in env.scene['robot'].joint_names],device=env.device)
        self.low_action=torch.tensor([pose[n] for n in self.action_names],device=env.device)
        obs=self.get_observations()
        lowc=asdict(low_config())['actor']
        gaitc=json.loads((BUNDLE_ROOT/'models/lateral_gait/legacy_setup.json').read_text())['algorithm']['actor']
        self.low_teacher=model(obs,LOW_MODEL,lowc,env.device)
        gait_obs=TensorDict({'actor':obs['actor'][:,:75]},batch_size=[env.num_envs])
        self.gait_teacher=model(gait_obs,GAIT_MODEL,gaitc,env.device)
        self.current_obs=obs

    def phase(self): return phase(self.env)

    def get_observations(self):
        obs=super().get_observations()
        self.current_obs=obs
        return obs

    def step(self,actions):
        a=actions.clamp(-1,1)
        s=self.phase()
        values=self.path.at(s)
        reference=values[:,3:25][:,self.path_order]
        feedforward=values[:,25:47][:,self.path_order]
        with torch.no_grad():
            lowobs=self.current_obs['actor'].clone()
            # The frozen low teacher retains its original feature convention:
            # joint angles and previous actions relative to its low home pose.
            lowobs[:,6:28]+=self.env.scene['robot'].data.default_joint_pos-self.low_xml
            lowobs[:,50:72]+=(self.home-self.low_action)/self.scale
            lowobs[:,83]=1.
            lowa=self.low_teacher(TensorDict({'actor':lowobs.clamp(-20,20)},batch_size=[self.num_envs])).clamp(-1,1)
            gaita=self.gait_teacher(TensorDict({'actor':self.current_obs['actor'][:,:75]},batch_size=[self.num_envs]))
        standing=(1-s[:,None])**6
        lowweight=s[:,None]**6
        middle=(1-standing-lowweight).clamp(min=0)
        target=reference+standing*self.scale*gaita+middle*feedforward+lowweight*.15*lowa+.15*a
        target=target.clamp(self.low,self.high)
        raw_action=(target-self.home)/self.scale
        self.low_obs,_,done,extras=RslRlVecEnvWrapper.step(self,raw_action)
        done=done.bool()
        m=measurements(self.env)
        out=self.env._collect
        s2=self.phase()
        nextref=self.path.at(s2)
        root=self.env.scene['robot'].data
        angle=root.joint_pos[:,self.term.target_ids]
        refangle=nextref[:,3:25][:,self.path_order]
        track=torch.exp(-((angle-refangle)/.25).square().mean(1))
        targettilt=nextref[:,2]
        tilt=torch.exp(-((m['tilt']-targettilt)/.18).square())
        height=torch.exp(-((m['height']-nextref[:,1]-.005)/.07).square())
        center=torch.exp(-(m['com_error']/.07).square())
        late=s2>.90
        reach=torch.exp(-(m['distances']-.135).abs().sum(1)/.15)*late
        both=m['contacts'].all(1)
        moved=(m['ball_pos'][:,:2]-m['ball_pos'].new_tensor((.20,0))).norm(dim=1)
        reward=self.env.step_dt*(1.+m['support'].float()+center+2*track+tilt+height
            +2*reach+2*late*both+3*late*m['valid']
            -.5*(m['speed']/.15).clamp(max=4)-.5*(moved/.15).clamp(max=4))*(~done)
        reward-=.015*(a-self.previous).square().mean(1)+.002*a.square().mean(1)
        reward+=15*(done&out.last['success']).float()-10*(done&out.last['failed']).float()
        ids=done.nonzero().flatten()
        if len(ids): self.vision.reset(ids)
        self.previous=a.clone()
        self.previous[done]=0
        extras['log']={'Transition/supported':m['support'].float(),
            'Transition/reference_tracking':track,'Transition/late_bilateral_contact':(late&both).float(),
            'Transition/success_event':(done&out.last['success']).float(),
            'Transition/full_standing_start':(self.env._start_fraction<1e-6).float()}
        return self.get_observations(),reward,done.long(),extras


def preserved():
    return bundle_integrity()
