"""Bounded arm-only Cartesian/force feedback after measured stop and descent.

Uses RGB ball estimate, robot FK/Jacobian and real hand tactile forces. Writes
motor targets only. Private-model FK is not a write to live simulation state.
"""
import sys
sys.dont_write_bytecode=True
import numpy as np
import mujoco
import torch
from occlusion_task import OcclusionIncomingEnv,make_cfg,measurements
from bridge import phase
from reach_collision import arm_collision_pairs


class TouchIncomingEnv(OcclusionIncomingEnv):
    force_target=.7
    force_radius_rate=.0004
    servo_phase_min=.98
    def __init__(self,env):
        super().__init__(env)
        self.arm_offset=torch.zeros((self.num_envs,len(self.action_names)),device=self.device)
        self.hand_radius=np.full((self.num_envs,2),.133)
        self.servo_joint_names=list(env.scene['robot'].joint_names)
        self.servo_order=[self.servo_joint_names.index(n) for n in self.action_names]
        model=self.vision.model_self
        self.servo_arms=[np.array([i for i,n in enumerate(self.servo_joint_names)
            if n.startswith(side) and ('Shoulder' in n or 'Elbow' in n)]) for side in ('Left','Right')]
        self.servo_dofs=np.array([model.joint('robot/'+n).dofadr[0] for n in self.servo_joint_names])
        self.servo_hands=[model.body('robot/'+side+'_hand_end_ball').id for side in ('left','right')]
        self.servo_pairs=arm_collision_pairs(model)
        self.servo_saturated=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)

    def desired_hand_position(self,ball,delta,radius,side,index=None):
        return ball+delta/np.linalg.norm(delta)*radius

    def recover_blocked_step(self,index,joints,point,current_clearance,step):
        return step,0.

    def perception_ready(self,vision,time):
        return vision.usable&((time-vision.corrected_at)<.4)

    def control_ball_points(self,vision): return vision.point

    def update_hand_radius(self,index,forces):
        self.hand_radius[index]+=np.clip(forces-self.force_target,-1.,3.)*self.force_radius_rate
        self.hand_radius[index]=np.clip(self.hand_radius[index],.129,.137)

    def adjust_motor_targets(self,target):
        raw=self.env; d=raw.scene['robot'].data; v=self.vision
        late=(raw._bridge_stage==2)&(phase(raw)>self.servo_phase_min)
        if getattr(self,'servo_requires_braked',True):late&=raw._bridge_braked
        late&=getattr(raw,'_sync_hand_ready',torch.ones_like(late))
        self.arm_offset[~late]=0
        self.hand_radius[(~late).cpu().numpy()]=.133
        if raw.common_step_counter%5==0:
            supported=raw.scene['feet_ground_contact'].data.found.flatten(1).all(1)
            quiet=(d.root_link_lin_vel_w.norm(dim=1)<.06)&(d.root_link_ang_vel_w.norm(dim=1)<.35)
            active=late&supported&quiet&self.perception_ready(v,raw.common_step_counter*raw.step_dt)
            ids=active.nonzero().flatten().tolist()
            if ids:
                roots=(d.root_link_pos_w-raw.scene.env_origins).cpu().numpy()
                quat=d.root_link_quat_w.cpu().numpy(); joints=d.joint_pos.cpu().numpy()
                points=self.control_ball_points(v).cpu().numpy()
                tactile=np.stack([raw.scene[side+'_ball_contact'].data.force.reshape(self.num_envs,-1,3).norm(dim=-1).sum(1).cpu().numpy()
                    for side in ('left','right')],1)
                offsets=self.arm_offset.cpu().numpy()
                model,data=v.model_self,v.data_self
                for i in ids:
                    data.qpos[:]=model.key_qpos[0]
                    data.qpos[v.root:v.root+3]=roots[i]; data.qpos[v.root+3:v.root+7]=quat[i]
                    data.qpos[v.adr]=joints[i]; mujoco.mj_kinematics(model,data)
                    mujoco.mj_comPos(model,data)  # Populate cdof before mj_jacBody.
                    current_clearance=min(mujoco.mj_geomDistance(model,data,a,b,.08,None) for a,b in self.servo_pairs)
                    step=np.zeros(len(self.servo_joint_names))
                    self.update_hand_radius(i,tactile[i])
                    for side,(arm,hand) in enumerate(zip(self.servo_arms,self.servo_hands)):
                        ball=np.r_[points[i],.11]  # RGB estimate + known ground-ball height.
                        delta=data.xpos[hand]-ball
                        distance=np.linalg.norm(delta)
                        if not .10<distance<.18: continue
                        desired=self.desired_hand_position(ball,delta,self.hand_radius[i,side],side,i)
                        error=desired-data.xpos[hand]
                        error*=min(1.,.006/max(np.linalg.norm(error),1e-8))
                        jac=np.zeros((3,model.nv)); rot=np.zeros_like(jac)
                        mujoco.mj_jacBody(model,data,jac,rot,hand)
                        j=jac[:,self.servo_dofs[arm]]
                        dq=.35*j.T@np.linalg.solve(j@j.T+.0001*np.eye(3),error)
                        step[arm]=np.clip(dq,-.015,.015)
                    accepted=0.
                    for scale in (1.,.5,.25):
                        data.qpos[v.adr]=joints[i]+scale*step
                        mujoco.mj_kinematics(model,data)
                        gap=min(mujoco.mj_geomDistance(model,data,a,b,.08,None) for a,b in self.servo_pairs)
                        if gap>=min(current_clearance,.0005)-.0001 and gap>-.0025:
                            accepted=scale; break
                    if accepted==0.:
                        step,accepted=self.recover_blocked_step(i,joints[i],points[i],current_clearance,step)
                    proposed=offsets[i]+accepted*step[self.servo_order]
                    self.servo_saturated[i]=bool(np.max(np.abs(proposed))>.10)
                    self.arm_offset[i]=torch.as_tensor(np.clip(proposed,-.10,.10),device=self.device,dtype=target.dtype)
        # Offsets are applied only after the original stopping/descent gates.
        # Legs, root pose and ball state are never adjusted by this controller.
        return target+self.arm_offset
