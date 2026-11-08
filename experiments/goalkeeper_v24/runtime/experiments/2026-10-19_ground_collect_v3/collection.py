"""Physical ground collection: IK reference + learned joint residuals, no attachments."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PROJECT=(BUNDLE_ROOT / 'runtime')
V2=PROJECT/'experiments/2026-10-18_natural_lateral_v2'
V3=PROJECT/'experiments/2026-10-18_vision_block_v3'
sys.dont_write_bytecode=True
sys.path.insert(0,str(V2/'src'))
sys.path.insert(0,str(V3))
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.managers import EventTermCfg,TerminationTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch,ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import booster_k1_flat_env_cfg
from goalkeeper_sim.full_goal.env_cfg import full_goal_env_cfg
from goalkeeper_sim.full_goal.outcomes import invalid_ball,out_of_guard_region,goal_crossed
from goalkeeper_sim.full_goal.settings import FullGoalSettings
from vision import VisionEstimate


def launch_static(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0]=.20+(torch.rand(len(ids),device=env.device)-.5)*.02
    state[:,1]=(torch.rand(len(ids),device=env.device)-.5)*.015
    state[:,2],state[:,3]=.112,1.
    state[:,:3]+=env.scene.env_origins[ids]
    env.scene['ball'].write_root_state_to_sim(state,env_ids=ids)


def fallen(env):
    d=env.scene['robot'].data
    return ((d.root_link_pos_w[:,2]-env.scene.env_origins[:,2]<.25)
        | (d.projected_gravity_b[:,2]>-.56))


def measurements(env):
    robot=env.scene['robot']
    if not hasattr(env,'_collect_geometry'):
        hands,_=robot.find_bodies(('left_hand_end_ball','right_hand_end_ball'),preserve_order=True)
        feet,_=robot.find_bodies(('left_foot_link','right_foot_link'),preserve_order=True)
        mass=torch.tensor(np.array([env.sim.mj_model.body('robot/'+n).mass[0] for n in robot.body_names]),device=env.device,dtype=torch.float32)
        env._collect_geometry=(hands,feet,mass/mass.sum())
    hi,fi,mass=env._collect_geometry
    d=robot.data
    ball=env.scene['ball'].data
    hands=d.body_link_pos_w[:,hi]
    feet=d.body_link_pos_w[:,fi]
    contact=[]
    forces=[]
    for side in ('left','right'):
        s=env.scene[side+'_ball_contact'].data
        force=s.force.reshape(env.num_envs,-1,3).norm(dim=-1).sum(1)
        forces.append(force)
        contact.append(s.found.flatten(1).any(1)&(force>.15))
    both=torch.stack(contact,1)
    com=(d.body_com_pos_w*mass[None,:,None]).sum(1)
    center_error=(com[:,:2]-feet[:,:,:2].mean(1)).norm(dim=1)
    support=env.scene['feet_ground_contact'].data.found.flatten(1).all(1)
    height=d.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
    upright=(d.projected_gravity_b[:,2]<-.70)&(height>.27)&support&(center_error<.085)
    pos=ball.root_link_pos_w-env.scene.env_origins
    illegal=env.scene['non_foot_ground_contact'].data.found.flatten(1).any(1)
    failed=fallen(env)|illegal|invalid_ball(env)|out_of_guard_region(env,FullGoalSettings())
    speed=ball.root_link_lin_vel_w.norm(dim=1)
    ground=env.scene['ball_ground_contact'].data.found.flatten(1).any(1)
    relative=(ball.root_link_lin_vel_w-d.body_link_lin_vel_w[:,hi].mean(1)).norm(dim=1)
    valid=(both.all(1)&ground&upright&~failed&(speed<.08)&(relative<.12)
        &(pos[:,2]<.14)&(pos[:,2]>.10)&(pos[:,0]-.11>-.8)
        &((ball.root_link_pos_w-hands.mean(1)).norm(dim=1)<.15))
    return dict(hands=hands,feet=feet,contacts=both,forces=torch.stack(forces,1),
        com_error=center_error,support=support,upright=upright,failed=failed,
        ball_pos=pos,speed=speed,valid=valid,height=height,
        distances=(hands-ball.root_link_pos_w[:,None]).norm(dim=-1),
        tilt=torch.acos((-d.projected_gravity_b[:,2]).clamp(-1,1)))


class CollectOutcome(ManagerTermBase):
    def __init__(self,cfg,env):
        super().__init__(env)
        self.stamp=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
        self.hold=torch.zeros_like(self.stamp)
        self.success=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.failed=self.success.clone()
        self.left_seen=self.success.clone()
        self.right_seen=self.success.clone()
        self.both_seen=self.success.clone()
        self.last={n:self.success.clone() for n in ('success','failed','left_seen','right_seen','both_seen')}
        self.last_hold=self.hold.clone()
        env._collect=self

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.last_hold[ids]=self.hold[ids]
        for name in self.last:
            self.last[name][ids]=getattr(self,name)[ids]
            getattr(self,name)[ids]=False
        self.hold[ids],self.stamp[ids]=0,-1

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        self.failed[fresh]|=m['failed'][fresh]
        self.left_seen[fresh]|=m['contacts'][fresh,0]
        self.right_seen[fresh]|=m['contacts'][fresh,1]
        self.both_seen[fresh]|=m['contacts'][fresh].all(1)
        valid=m['valid']&~self.failed&(env.episode_length_buf*env.step_dt>.8)
        self.hold[fresh]=torch.where(valid[fresh],self.hold[fresh]+1,0)
        self.success[fresh]|=self.hold[fresh]>=math.ceil(.6/env.step_dt-1e-9)
        self.stamp[fresh]=env.common_step_counter
        return self.success&~self.failed


def make_cfg(num_envs=16):
    base=booster_k1_flat_env_cfg(play=False)
    c=base.commands['twist']
    c.rel_standing_envs=1.
    c.rel_heading_envs=c.rel_forward_envs=0.
    c.heading_command=False
    c.ranges.lin_vel_x=c.ranges.lin_vel_y=c.ranges.ang_vel_z=(0.,0.)
    c.ranges.heading=None
    if hasattr(c,'init_velocity_prob'): c.init_velocity_prob=0.
    base.curriculum={}
    cfg=full_goal_env_cfg(policy_env_cfg=base,num_envs=num_envs,play=True,enable_camera=True)
    # Curriculum 1 starts at the feasible low posture. Root state is set ONLY by
    # the normal reset; gravity, balance and hand-ball interactions remain free.
    poses=json.loads((HERE/'poses.json').read_text())
    pose=poses['poses'][poses['selected']]
    initial=cfg.scene.entities['robot'].init_state
    initial.pos=(pose['root_x'],0.,pose['root_z'])
    initial.rot=(math.cos(pose['root_pitch']/2),0.,math.sin(pose['root_pitch']/2),0.)
    initial.joint_pos=pose['joint_pos']
    articulation=cfg.scene.entities['robot'].articulation
    cfg.scene.entities['robot'].articulation=replace(articulation,actuators=tuple(
        replace(a,delay_min_lag=0,delay_max_lag=0) for a in articulation.actuators))
    cfg.scene.env_spacing=6.
    cfg.episode_length_s=8.
    cfg.events.pop('full_goal_launch')
    cfg.events['static_ball']=EventTermCfg(func=launch_static,mode='reset')
    cfg.scene.entities['robot'].collisions=(*cfg.scene.entities['robot'].collisions,
        CollisionCfg(geom_names_expr=('left_hand_collision','left_hand_end_collision',
            'right_hand_collision','right_hand_end_collision'),contype=1,conaffinity=1,
            friction=(.9,.005,.0001),condim=6,priority=0,disable_other_geoms=False,
            solref=(.01,1.),solmix=.001))
    sensors=list(cfg.scene.sensors)
    for side in ('left','right'):
        sensors.append(ContactSensorCfg(name=side+'_ball_contact',
            primary=ContactMatch(mode='subtree',entity='robot',pattern=side+'_hand_link'),
            secondary=ContactMatch(mode='geom',entity='ball',pattern='football_geom'),
            fields=('found','force'),reduce='netforce',num_slots=1))
    cfg.scene.sensors=tuple(sensors)
    cfg.rewards={}
    cfg.terminations={
        'collection_success':TerminationTermCfg(func=CollectOutcome),
        'time_out':cfg.terminations['time_out'],
        'fallen':TerminationTermCfg(func=fallen),
        'illegal_contact':cfg.terminations['illegal_contact'],
        'out_of_guard_region':cfg.terminations['out_of_guard_region'],
        'invalid_ball':cfg.terminations['invalid_ball']}
    return cfg


class CollectionEnv(RslRlVecEnvWrapper):
    def __init__(self,env):
        super().__init__(env,clip_actions=None)
        self.vision=VisionEstimate(env)
        self.previous=torch.zeros((env.num_envs,22),device=env.device)
        poses=json.loads((HERE/'poses.json').read_text())
        p=poses['poses'][poses['selected']]
        term=env.action_manager.get_term('joint_pos')
        self.action_names=term.target_names
        self.target=torch.tensor([p['joint_pos'][n] for n in self.action_names],device=env.device)
        self.home=term.offset.clone()
        self.scale=term.scale
        self.term=term
        ids=term.target_ids
        limits=env.scene['robot'].data.joint_pos_limits
        self.low=limits[:,ids,0]+.005
        self.high=limits[:,ids,1]-.005
        self.residual_scale=.15
        self.low_obs=super().get_observations()

    def phase(self):
        return torch.ones(self.num_envs,device=self.device)

    def get_observations(self):
        low=super().get_observations()['actor']
        self.vision.update(self.env)
        v=self.vision
        d=self.env.scene['robot'].data
        own=d.root_link_pos_w-self.env.scene.env_origins
        m=measurements(self.env)
        remembered=torch.where((v.age<1.)[:,None],v.point-own[:,:2],0.)
        extra=torch.cat((remembered,v.velocity,v.age.clamp(0,2)[:,None],v.usable.float()[:,None],
            m['contacts'].float(),self.phase()[:,None],
            (m['hands']-d.root_link_pos_w[:,None]).flatten(1),m['forces'].clamp(0,50)/50),1)
        actor=torch.cat((low,extra),1).clamp(-20,20)
        ball=self.env.scene['ball'].data
        critic=torch.cat((actor,ball.root_link_pos_w-d.root_link_pos_w,ball.root_link_lin_vel_w),1)
        return TensorDict({'actor':actor,'critic':critic},batch_size=[self.num_envs])

    def reset(self):
        low,extras=super().reset()
        self.low_obs=low
        if hasattr(self,'vision'):
            self.vision.reset(slice(None))
            self.previous.zero_()
        return self.get_observations(),extras

    def step(self,actions):
        a=actions.clamp(-1,1)
        progress=self.phase()[:,None]
        reference=self.home+(self.target[None]-self.home)*progress
        target=(reference+self.residual_scale*a).clamp(self.low,self.high)
        raw_action=(target-self.home)/self.scale
        with torch.no_grad():
            self.low_obs,_,done,extras=super().step(raw_action)
        done=done.bool()
        m=measurements(self.env)
        out=self.env._collect
        # Dense distances/kinematics supervise rewards only, not RGB actor inputs.
        reach=torch.exp(-(m['distances']-.135).abs().sum(1)/.15)
        support=m['support'].float()
        center=torch.exp(-(m['com_error']/.07).square())
        both=m['contacts'].all(1).float()
        ball_moved=(m['ball_pos'][:,:2]-m['ball_pos'].new_tensor((.20,0))).norm(dim=1)
        reward=self.env.step_dt*(2.*reach+support+center+2.*both+2.*m['valid'].float()
            -.5*(m['speed']/.15).clamp(max=4)-.5*(ball_moved/.15).clamp(max=4))*(~done)
        reward-=.02*(a-self.previous).square().mean(1)+.002*a.square().mean(1)
        reward+=10.*(done&out.last['success']).float()-8.*(done&out.last['failed']).float()
        ids=done.nonzero().flatten()
        if len(ids): self.vision.reset(ids)
        self.previous=a.clone()
        self.previous[done]=0
        extras['log']={'Collection/bilateral_contact':both,'Collection/valid_control':m['valid'].float(),
            'Collection/hand_distance':m['distances'].mean(1),'Collection/com_error':m['com_error'],
            'Collection/success_event':(done&out.last['success']).float()}
        return self.get_observations(),reward,done.long(),extras
