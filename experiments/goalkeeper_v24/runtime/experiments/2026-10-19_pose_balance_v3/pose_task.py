"""Stationary-ball candidate-pose maintenance and real bilateral contact."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
import copy
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=(BUNDLE_ROOT / 'runtime')
LOW=PROJECT/'experiments/2026-10-19_ground_collect_v3'
sys.path.insert(0,str(LOW))
import torch
from tensordict import TensorDict
from mjlab.managers import EventTermCfg,TerminationTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply
from booster_mjlab.robots.booster_k1.k1_constants import HOME_KEYFRAME
from collection import CollectionEnv,make_cfg as old_cfg,measurements as legacy_measurements
from goalkeeper_sim.full_goal.outcomes import invalid_ball,out_of_guard_region
from goalkeeper_sim.full_goal.settings import FullGoalSettings
sys.path.insert(0,str(PROJECT/'experiments/2026-10-19_handoff_speed_v3'))
from handoff import preserved as previous_preserved,sha


def book(env):
    if hasattr(env,'_pose_book'): return env._pose_book
    source=json.loads((HERE/'candidates.json').read_text())
    names=source['training_candidates']
    poses=[source['poses'][name] for name in names]
    robot=env.scene['robot']
    device=env.device
    result=dict(names=names,poses=poses,
        q=torch.tensor([[p['joint_pos'][n] for n in robot.joint_names] for p in poses],device=device),
        ff=torch.tensor([[p['gravity_offset'][n] for n in robot.joint_names] for p in poses],device=device),
        x=torch.tensor([p['root_x'] for p in poses],device=device),
        z=torch.tensor([p['root_z'] for p in poses],device=device),
        pitch=torch.tensor([p['root_pitch'] for p in poses],device=device),
        knee=torch.tensor([math.radians(p['knee_degrees']) for p in poses],device=device),
        feet=torch.tensor(source['feet_reference'],device=device))
    env._pose_book=result
    return result


def reset_pose(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    b=book(env)
    if not hasattr(env,'_pose_ids'): env._pose_ids=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    pose=ids%len(b['names'])
    if getattr(env,'_forced_pose',None) is not None: pose[:]=env._forced_pose
    env._pose_ids[ids]=pose
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0],state[:,2]=b['x'][pose],b['z'][pose]+.005
    state[:,3],state[:,5]=torch.cos(b['pitch'][pose]/2),torch.sin(b['pitch'][pose]/2)
    state[:,:3]+=env.scene.env_origins[ids]
    env.scene['robot'].write_root_state_to_sim(state,env_ids=ids)
    q=b['q'][pose]+(torch.rand((len(ids),b['q'].shape[1]),device=env.device)-.5)*.008
    env.scene['robot'].write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)


def reset_ball(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0]=.20+(torch.rand(len(ids),device=env.device)-.5)*.020
    state[:,1]=(torch.rand(len(ids),device=env.device)-.5)*.016
    state[:,2],state[:,3]=.112,1.
    state[:,:3]+=env.scene.env_origins[ids]
    env.scene['ball'].write_root_state_to_sim(state,env_ids=ids)


def measurements(env):
    m=legacy_measurements(env)
    b=book(env)
    robot=env.scene['robot']
    d=robot.data
    hi,fi,masses=env._collect_geometry
    com=(d.body_com_pos_w*masses[None,:,None]).sum(1)
    local=d.body_link_pos_w.new_tensor(((-.025,.025,-.03),(.075,.025,-.03),
        (-.025,-.025,-.03),(.075,-.025,-.03)))
    # Conservative quadrilateral from the two outer sole edges. The reference
    # feet are parallel; foot tilt and drift are independently constrained.
    pos=torch.stack((m['feet'][:,0],m['feet'][:,0],m['feet'][:,1],m['feet'][:,1]),1)
    q=torch.stack((d.body_link_quat_w[:,fi[0]],d.body_link_quat_w[:,fi[0]],
        d.body_link_quat_w[:,fi[1]],d.body_link_quat_w[:,fi[1]]),1)
    corners=pos+quat_apply(q.reshape(-1,4),local[None].expand(env.num_envs,-1,-1).reshape(-1,3)).reshape(env.num_envs,4,3)
    polygon=corners[:,[2,3,1,0],:2]
    edge=torch.roll(polygon,-1,1)-polygon
    relative=com[:,None,:2]-polygon
    margin=((edge[:,:,0]*relative[:,:,1]-edge[:,:,1]*relative[:,:,0])/edge.norm(dim=-1).clamp_min(1e-6)).min(1).values
    normals=quat_apply(d.body_link_quat_w[:,fi].reshape(-1,4),
        pos.new_tensor((0.,0.,1.)).expand(env.num_envs*2,-1)).reshape(env.num_envs,2,3)
    foot_tilt=torch.acos(normals[:,:,2].clamp(-1,1)).max(1).values
    contact=env.scene['feet_ground_contact'].data
    loads=contact.force.reshape(env.num_envs,2,-1,3).norm(dim=-1).sum(2)
    robot_mass=sum(float(env.sim.mj_model.body('robot/'+n).mass[0]) for n in robot.body_names)
    load_fraction=loads/(robot_mass*9.81)
    supported=contact.found.flatten(1).all(1)&(load_fraction>.08).all(1)
    actual_feet=m['feet']-env.scene.env_origins[:,None]
    drift=(actual_feet[:,:,:2]-b['feet'][None,:,:2]).norm(dim=-1).max(1).values
    ids=env._pose_ids
    ki=[robot.joint_names.index(n) for n in ('Left_Knee_Pitch','Right_Knee_Pitch')]
    knee_error=(d.joint_pos[:,ki]-b['knee'][ids,None]).abs().max(1).values
    lean_error=(m['tilt']-b['pitch'][ids]).abs()
    illegal=env.scene['non_foot_ground_contact'].data.found.flatten(1).any(1)
    after_settle=env.episode_length_buf*env.step_dt>.4
    fallen=(m['height']<.25)|illegal|(after_settle&((margin<-.035)|(foot_tilt>.45)
        |(m['height']-b['z'][ids]).abs().gt(.10)|(lean_error>.30)))
    failed=fallen|invalid_ball(env)|out_of_guard_region(env,FullGoalSettings())
    stable=supported&(margin>.008)&(foot_tilt<.15)&(drift<.025)
    stable&=(d.root_link_lin_vel_w.norm(dim=1)<.06)&(d.root_link_ang_vel_w.norm(dim=1)<.35)
    profile=(knee_error<math.radians(8))&(lean_error<math.radians(10))
    ball=env.scene['ball'].data
    grounded=env.scene['ball_ground_contact'].data.found.flatten(1).any(1)
    relative_speed=(ball.root_link_lin_vel_w-d.body_link_lin_vel_w[:,hi].mean(1)).norm(dim=1)
    valid=(m['contacts'].all(1)&grounded&stable&profile&~failed&(m['speed']<.08)&(relative_speed<.12)
        &(m['ball_pos'][:,2]>.10)&(m['ball_pos'][:,2]<.14)
        &((ball.root_link_pos_w-m['hands'].mean(1)).norm(dim=1)<.15))
    m.update(com=com,margin=margin,foot_tilt=foot_tilt,loads=load_fraction,support=supported,
        foot_drift=drift,stable=stable,profile=profile,knee_error=knee_error,lean_error=lean_error,
        fallen=fallen,failed=failed,valid=valid,
        foot_velocity=d.body_link_lin_vel_w[:,fi,:2].mean(1))
    return m


class PoseOutcome(ManagerTermBase):
    def __init__(self,cfg,env):
        super().__init__(env)
        self.stamp=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
        self.hold=torch.zeros_like(self.stamp)
        self.peak_hold=self.hold.clone()
        self.last_peak_hold=self.hold.clone()
        self.pose_id=self.hold.clone()
        self.last_pose_id=self.hold.clone()
        self.last={}
        for name in ('success','failed','fallen','left_seen','right_seen','both_seen'):
            value=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
            setattr(self,name,value)
            self.last[name]=value.clone()
        env._pose_outcome=self

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.last_peak_hold[ids]=self.peak_hold[ids]
        self.last_pose_id[ids]=self.pose_id[ids]
        for name in self.last:
            self.last[name][ids]=getattr(self,name)[ids]
            getattr(self,name)[ids]=False
        self.hold[ids]=self.peak_hold[ids]=0
        self.stamp[ids]=-1

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        self.pose_id[fresh]=env._pose_ids[fresh]
        self.failed[fresh]|=m['failed'][fresh]
        self.fallen[fresh]|=m['fallen'][fresh]
        self.left_seen[fresh]|=m['contacts'][fresh,0]
        self.right_seen[fresh]|=m['contacts'][fresh,1]
        self.both_seen[fresh]|=m['contacts'][fresh].all(1)
        valid=m['valid']&~self.failed&(env.episode_length_buf*env.step_dt>.6)
        self.hold[fresh]=torch.where(valid[fresh],self.hold[fresh]+1,0)
        self.peak_hold[fresh]=torch.maximum(self.peak_hold[fresh],self.hold[fresh])
        self.success[fresh]|=self.hold[fresh]>=math.ceil(2./env.step_dt-1e-9)
        self.stamp[fresh]=env.common_step_counter
        return self.success&~self.failed


def physical_failure(env): return measurements(env)['failed']


def make_cfg(num_envs=16):
    cfg=old_cfg(num_envs)
    cfg.scene.entities['robot'].init_state=copy.deepcopy(HOME_KEYFRAME)
    cfg.events={'pose_reset':EventTermCfg(func=reset_pose,mode='reset'),
        'ball_reset':EventTermCfg(func=reset_ball,mode='reset')}
    cfg.episode_length_s=8.
    cfg.terminations={'pose_success':TerminationTermCfg(func=PoseOutcome),
        'physical_failure':TerminationTermCfg(func=physical_failure),
        'time_out':cfg.terminations['time_out']}
    return cfg


class PoseEnv(CollectionEnv):
    def __init__(self,env):
        super().__init__(env)
        self.filtered=torch.zeros_like(self.previous)
        self.order=[env.scene['robot'].joint_names.index(n) for n in self.action_names]
        self.amplitude=torch.tensor([.15 if 'Shoulder' in n or 'Elbow' in n else .08 for n in self.action_names],device=env.device)

    def get_observations(self):
        base=super().get_observations()
        b=book(self.env)
        m=measurements(self.env)
        d=self.env.scene['robot'].data
        ids=self.env._pose_ids
        extra=torch.cat((torch.nn.functional.one_hot(ids,len(b['names'])).float(),
            m['com'][:,:2]-m['feet'][:,:,:2].mean(1),m['margin'][:,None],m['loads'].clamp(0,2),
            d.root_link_lin_vel_w,m['foot_velocity'],m['support'].float()[:,None],
            b['knee'][ids,None],b['pitch'][ids,None],b['z'][ids,None]),1)
        actor=torch.cat((base['actor'],extra),1)
        return TensorDict({'actor':actor,'critic':torch.cat((actor,base['critic'][:,-6:]),1)},batch_size=[self.num_envs])

    def reset(self):
        result=super().reset()
        if hasattr(self,'filtered'): self.filtered.zero_()
        return result

    def step(self,actions):
        a=actions.clamp(-1,1)
        b=book(self.env)
        ids=self.env._pose_ids
        self.filtered=.8*self.filtered+.2*a
        reference=b['q'][ids][:,self.order]
        feedforward=b['ff'][ids][:,self.order]
        target=(reference+feedforward+self.amplitude*self.filtered).clamp(self.low,self.high)
        low,_,done,extras=RslRlVecEnvWrapper.step(self,(target-self.home)/self.scale)
        done=done.bool()
        m=measurements(self.env)
        out=self.env._pose_outcome
        angles=self.env.scene['robot'].data.joint_pos[:,self.term.target_ids]
        track=torch.exp(-((angles-book(self.env)['q'][self.env._pose_ids][:,self.order])/.15).square().mean(1))
        reach=torch.exp(-(m['distances']-.130).abs().sum(1)/.10)
        center=(m['margin']/.04).clamp(0,1)
        reward=self.env.step_dt*(-.5+.3*track+.3*center+.2*m['stable'].float()
            +.5*reach+m['contacts'].float().sum(1)+2*m['contacts'].all(1)+4*m['valid'].float()
            -2*(m['speed']/.15).clamp(max=3)
            -3*((m['ball_pos'][:,2]-.14)/.05).clamp(0,3))*(~done)
        reward-=.01*(a-self.previous).square().mean(1)+.001*a.square().mean(1)
        reward+=50*(done&out.last['success']).float()-25*(done&out.last['failed']).float()
        reward-=6*(done&self.env.termination_manager.get_term('time_out')&~out.last['success']).float()
        self.previous=a.clone()
        self.previous[done]=self.filtered[done]=0
        reset_ids=done.nonzero().flatten()
        if len(reset_ids): self.vision.reset(reset_ids)
        extras['log']={'Pose/support_margin':m['margin'],'Pose/stable':m['stable'].float(),
            'Pose/bilateral_contact':m['contacts'].all(1).float(),'Pose/valid_hold':m['valid'].float(),
            'Pose/success_event':(done&out.last['success']).float()}
        return self.get_observations(),reward,done.long(),extras


def preserved():
    return bundle_integrity()
