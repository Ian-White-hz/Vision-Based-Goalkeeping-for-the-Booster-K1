"""Earlier descent plus bounded millimetre-scale saturated-hand recovery.

Only RGB ball estimates, measured robot state and real tactile forces enter
these motor corrections. Keep all actual stop/ground/hold acceptance unchanged.
"""
import numpy as np
import mujoco
import torch
from scipy.optimize import lsq_linear
from settled_capture_task import QuickCaptureEnv,make_cfg,measurements,raw_class
from bridge import phase


class CoordinatedCaptureEnv(QuickCaptureEnv):
    def __init__(self,env):
        super().__init__(env)
        self.micro_offset=torch.zeros_like(self.arm_offset)
        self.micro_attempts=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.micro_accepted=torch.zeros_like(self.micro_attempts)

    def adjust_motor_targets(self,target):
        target=super().adjust_motor_targets(target)
        raw=self.env; d=raw.scene['robot'].data; v=self.vision
        self.micro_offset[raw._bridge_stage<2]=0
        if raw.common_step_counter%5:
            return target+self.micro_offset
        late=(raw._bridge_stage==2)&(phase(raw)>.98)&raw._bridge_braked
        supported=raw.scene['feet_ground_contact'].data.found.flatten(1).all(1)
        quiet=(d.root_link_lin_vel_w.norm(dim=1)<.06)&(d.root_link_ang_vel_w.norm(dim=1)<.35)
        active=late&supported&quiet&self.servo_saturated&self.perception_ready(v,raw.common_step_counter*raw.step_dt)
        ids=active.nonzero().flatten().tolist()
        if not ids:
            return target+self.micro_offset
        forces=np.stack([raw.scene[side+'_ball_contact'].data.force.reshape(self.num_envs,-1,3).norm(dim=-1).sum(1).cpu().numpy()
            for side in ('left','right')],1)
        roots=(d.root_link_pos_w-raw.scene.env_origins).cpu().numpy()
        quat=d.root_link_quat_w.cpu().numpy(); joints=d.joint_pos.cpu().numpy()
        points=self.control_ball_points(v).cpu().numpy()
        extra=self.micro_offset.cpu().numpy()
        arms=np.concatenate(self.servo_arms)
        arm_action_ids=np.asarray([self.servo_order.index(j) for j in arms])
        model,data=v.model_self,v.data_self
        adr=np.asarray(v.adr)
        for i in ids:
            # Correct only a missing hand when the other has light real contact.
            # Never use this to chase an escaped ball or increase high pressure.
            if not ((forces[i]<.15).any() and (forces[i]>.15).any() and forces[i].max()<3.):
                continue
            data.qpos[:]=model.key_qpos[0]
            data.qpos[v.root:v.root+3]=roots[i]; data.qpos[v.root+3:v.root+7]=quat[i]
            data.qpos[adr]=joints[i]
            mujoco.mj_kinematics(model,data); mujoco.mj_comPos(model,data)
            ball=np.r_[points[i],.11]
            delta=data.xpos[self.servo_hands]-ball
            distance=np.linalg.norm(delta,axis=1)
            if not ((distance>.10)&(distance<.18)).all():
                continue
            jacobians=[]; errors=[]
            for side,hand in enumerate(self.servo_hands):
                desired=(self.desired_hand_position(ball,delta[side],.1335,side,i)
                    if forces[i,side]<.15 else data.xpos[hand].copy())
                error=desired-data.xpos[hand]
                error*=min(1.,.002/max(np.linalg.norm(error),1e-8))
                jac=np.zeros((3,model.nv)); rot=np.zeros_like(jac)
                mujoco.mj_jacBody(model,data,jac,rot,hand)
                jacobians.append(jac[:,self.servo_dofs[arms]])
                errors.append(error)
            old=extra[i,arm_action_ids]
            joint_ids=[model.joint('robot/'+self.servo_joint_names[j]).id for j in arms]
            lower=np.maximum.reduce([np.full(len(arms),-.004),-.02-old,
                model.jnt_range[joint_ids,0]+.005-joints[i,arms]])
            upper=np.minimum.reduce([np.full(len(arms),.004),.02-old,
                model.jnt_range[joint_ids,1]-.005-joints[i,arms]])
            # Interior numerical bound: a saturated coordinate may stay still.
            valid=upper-lower>1e-9
            if not valid.any(): continue
            J=np.vstack(jacobians)
            solve=lsq_linear(np.vstack((J[:,valid],.025*np.eye(int(valid.sum())))),
                np.r_[.5*np.concatenate(errors),np.zeros(int(valid.sum()))],
                bounds=(lower[valid],upper[valid]),tol=1e-8)
            step=np.zeros(len(arms)); step[valid]=solve.x
            current=min(mujoco.mj_geomDistance(model,data,a,b,.08,None) for a,b in self.servo_pairs)
            self.micro_attempts[i]+=1
            for scale in (1.,.5,.25):
                data.qpos[adr]=joints[i]; data.qpos[adr[arms]]=joints[i,arms]+scale*step
                mujoco.mj_kinematics(model,data)
                gap=min(mujoco.mj_geomDistance(model,data,a,b,.08,None) for a,b in self.servo_pairs)
                if gap>=min(current,.0005)-.0001 and gap>-.0025:
                    extra[i,arm_action_ids]=np.clip(old+scale*step,-.02,.02)
                    self.micro_accepted[i]+=1
                    break
            self.micro_offset[i]=torch.as_tensor(extra[i],device=self.device,dtype=target.dtype)
        # Additional targets are at most .02rad, with total arm offset <=.12rad.
        # The normal motor joint limits still apply in IncomingEnv.step.
        return target+self.micro_offset


env_class=CoordinatedCaptureEnv
