"""Free-base standing entry, frozen endpoint skill and learned joint residuals."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=(BUNDLE_ROOT / 'runtime')
SOURCE=PROJECT/'experiments/2026-10-19_pose_balance_v3'
sys.path.insert(0,str(SOURCE))
import torch
from tensordict import TensorDict
from mjlab.managers import EventTermCfg,TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.models import MLPModel
from pose_task import PoseEnv,PoseOutcome,book,measurements as pose_measurements,make_cfg as pose_cfg
from pose_task import preserved as pose_preserved,sha
from train_poses import config as pose_config

SETTLE=.8
DURATION=6.
HOLD=2.
ENDPOINT=model_path('pose_endpoint')


class PathTable:
    def __init__(self,env):
        source=json.loads((HERE/'path.json').read_text())
        self.names=source['joint_names']
        self.values=torch.tensor([[f['root_x'],f['root_z'],f['root_pitch'],
            *f['joint_pos'],*f['gravity_offset']] for f in source['frames']],device=env.device)
        self.robot_order=[self.names.index(n) for n in env.scene['robot'].joint_names]
        self.knees=[self.names.index(n) for n in ('Left_Knee_Pitch','Right_Knee_Pitch')]

    def at(self,f):
        point=f.clamp(0,1)*(len(self.values)-1)
        i=point.long().clamp(max=len(self.values)-2)
        w=(point-i)[:,None]
        return self.values[i]*(1-w)+self.values[i+1]*w


def table(env):
    if not hasattr(env,'_entry_table'): env._entry_table=PathTable(env)
    return env._entry_table


def phase(env):
    t=env.episode_length_buf*env.step_dt
    u=((t-SETTLE)/DURATION).clamp(0,1)
    return u**3*(10-15*u+6*u.square())


def reset_standing(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    b=book(env)
    if not hasattr(env,'_pose_ids'): env._pose_ids=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    env._pose_ids[ids]=b['names'].index('knee25_lean80')
    p=table(env).at(torch.zeros(len(ids),device=env.device))
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0],state[:,2]=p[:,0],p[:,1]+.005
    state[:,3],state[:,5]=torch.cos(p[:,2]/2),torch.sin(p[:,2]/2)
    state[:,:3]+=env.scene.env_origins[ids]
    robot=env.scene['robot']
    robot.write_root_state_to_sim(state,env_ids=ids)
    q=p[:,3:25][:,table(env).robot_order]
    q=q+(torch.rand_like(q)-.5)*.004
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)


def measurements(env):
    m=pose_measurements(env)
    ref=table(env).at(phase(env))
    d=env.scene['robot'].data
    illegal=env.scene['non_foot_ground_contact'].data.found.flatten(1).any(1)
    settled=env.episode_length_buf*env.step_dt>.4
    fallen=(m['height']<.25)|illegal|(settled&((m['margin']<-.035)|(m['foot_tilt']>.45)
        |((m['height']-ref[:,1]-.005).abs()>.11)|((m['tilt']-ref[:,2]).abs()>.35)))
    from goalkeeper_sim.full_goal.outcomes import invalid_ball,out_of_guard_region
    from goalkeeper_sim.full_goal.settings import FullGoalSettings
    failed=fallen|invalid_ball(env)|out_of_guard_region(env,FullGoalSettings())
    ball=env.scene['ball'].data
    hi=env._collect_geometry[0]
    relative=(ball.root_link_lin_vel_w-d.body_link_lin_vel_w[:,hi].mean(1)).norm(dim=1)
    ground=env.scene['ball_ground_contact'].data.found.flatten(1).any(1)
    ready=(phase(env)>.9999)&(env.episode_length_buf*env.step_dt>SETTLE+DURATION+.2)
    valid=(ready&m['contacts'].all(1)&ground&m['stable']&m['profile']&~failed
        &(m['speed']<.08)&(relative<.12)&(m['ball_pos'][:,2]>.10)&(m['ball_pos'][:,2]<.14)
        &((ball.root_link_pos_w-m['hands'].mean(1)).norm(dim=1)<.15))
    m.update(failed=failed,fallen=fallen,valid=valid,ready=ready,reference=ref)
    return m


class EntryOutcome(PoseOutcome):
    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        self.pose_id[fresh]=env._pose_ids[fresh]
        self.failed[fresh]|=m['failed'][fresh]
        self.fallen[fresh]|=m['fallen'][fresh]
        self.left_seen[fresh]|=m['contacts'][fresh,0]
        self.right_seen[fresh]|=m['contacts'][fresh,1]
        self.both_seen[fresh]|=m['contacts'][fresh].all(1)
        valid=m['valid']&~self.failed
        self.hold[fresh]=torch.where(valid[fresh],self.hold[fresh]+1,0)
        self.peak_hold[fresh]=torch.maximum(self.peak_hold[fresh],self.hold[fresh])
        self.success[fresh]|=self.hold[fresh]>=math.ceil(HOLD/env.step_dt-1e-9)
        self.stamp[fresh]=env.common_step_counter
        return self.success&~self.failed


def failure(env): return measurements(env)['failed']


def make_cfg(num_envs=16):
    cfg=pose_cfg(num_envs)
    cfg.events['pose_reset']=EventTermCfg(func=reset_standing,mode='reset')
    cfg.episode_length_s=12.
    cfg.terminations['pose_success']=TerminationTermCfg(func=EntryOutcome)
    cfg.terminations['physical_failure']=TerminationTermCfg(func=failure)
    return cfg


class EntryEnv(PoseEnv):
    def __init__(self,env):
        super().__init__(env)
        self.path_order=[table(env).names.index(n) for n in self.action_names]
        obs=self.get_observations()
        c=asdict(pose_config())['actor']
        args={k:c[k] for k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
        self.endpoint=MLPModel(obs,{'actor':['actor']},'actor',22,**args).to(env.device)
        self.endpoint.load_state_dict(torch.load(ENDPOINT,map_location=env.device,weights_only=False)['actor_state_dict'])
        self.endpoint.eval().requires_grad_(False)
        self.teacher_filtered=torch.zeros_like(self.filtered)

    def phase(self): return phase(self.env)

    def get_observations(self):
        # The existing 109-feature convention is retained. The actor receives
        # RGB estimates and robot/contact state; only critic adds ball truth.
        obs=super().get_observations()
        ref=table(self.env).at(self.phase())
        descriptors=torch.stack((ref[:,3:25][:,table(self.env).knees].mean(1),ref[:,2],ref[:,1]),1)
        obs['actor'][:,-3:]=descriptors
        obs['critic'][:,-9:-6]=descriptors
        return obs

    def reset(self):
        obs,extras=super().reset()
        if hasattr(self,'teacher_filtered'): self.teacher_filtered.zero_()
        return obs,extras

    def step(self,actions):
        a=actions.clamp(-1,1)
        s=self.phase()
        ref=table(self.env).at(s)
        q=ref[:,3:25][:,self.path_order]
        ff=ref[:,25:47][:,self.path_order]
        with torch.no_grad():
            frozen_obs=self.get_observations()['actor'].clone()
            b=book(self.env); ids=self.env._pose_ids
            frozen_obs[:,83]=1.
            frozen_obs[:,-3:]=torch.stack((b['knee'][ids],b['pitch'][ids],b['z'][ids]),1)
            teacher=self.endpoint(TensorDict({'actor':frozen_obs},batch_size=[self.num_envs])).clamp(-1,1)
        self.teacher_filtered=.8*self.teacher_filtered+.2*teacher
        self.filtered=.8*self.filtered+.2*a
        endpoint_weight=s[:,None]**10
        target=q+ff+endpoint_weight*self.amplitude*self.teacher_filtered+self.amplitude*self.filtered
        target=target.clamp(self.low,self.high)
        _,_,done,extras=RslRlVecEnvWrapper.step(self,(target-self.home)/self.scale)
        done=done.bool()
        m=measurements(self.env); out=self.env._pose_outcome
        after=m['reference']; angles=self.env.scene['robot'].data.joint_pos[:,self.term.target_ids]
        track=torch.exp(-((angles-after[:,3:25][:,self.path_order])/.18).square().mean(1))
        lean=torch.exp(-((m['tilt']-after[:,2])/.12).square())
        height=torch.exp(-((m['height']-after[:,1]-.005)/.04).square())
        late=self.phase()>.9
        reach=torch.exp(-(m['distances']-.13).abs().sum(1)/.10)*late
        both=m['contacts'].all(1)
        reward=self.env.step_dt*(-.5+.5*track+.4*lean+.4*height+.5*m['support'].float()
            +.5*(m['margin']/.04).clamp(0,1)+.5*reach+late*m['contacts'].float().sum(1)
            +2*late*both+4*m['valid'].float()-2*(m['speed']/.15).clamp(max=3)
            -3*((m['ball_pos'][:,2]-.14)/.05).clamp(0,3))*(~done)
        reward-=.01*(a-self.previous).square().mean(1)+.001*a.square().mean(1)
        reward+=50*(done&out.last['success']).float()-25*(done&out.last['failed']).float()
        reward-=8*(done&self.env.termination_manager.get_term('time_out')&~out.last['success']).float()
        self.previous=a.clone()
        self.previous[done]=self.filtered[done]=self.teacher_filtered[done]=0
        reset_ids=done.nonzero().flatten()
        if len(reset_ids): self.vision.reset(reset_ids)
        extras['log']={'Entry/phase':self.phase(),'Entry/reference_tracking':track,
            'Entry/support_margin':m['margin'],'Entry/support':m['support'].float(),
            'Entry/valid_hold':m['valid'].float(),'Entry/success_event':(done&out.last['success']).float()}
        return self.get_observations(),reward,done.long(),extras


def preserved():
    return bundle_integrity()
