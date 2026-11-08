"""Slower force admittance and optional constrained shoulder/elbow recovery."""
import numpy as np
import mujoco
import torch
from scipy.optimize import least_squares
from spacing_touch_task import SpacingTouchEnv,make_cfg,measurements
from anchor_width_task import AnchorWidthRawEnv as raw_class


class SlowTouchEnv(SpacingTouchEnv):
    # The observed force limit cycle was ~2.6s, with both contacts repeatedly
    # lost after ~1.9s. Reduce the outer integral gain instead of weakening the
    # actual 2s/0.15N acceptance requirements.
    force_radius_rate=.0001


class IKTouchEnv(SpacingTouchEnv):
    def __init__(self,env):
        super().__init__(env)
        self.ik_attempts=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.ik_accepted=torch.zeros_like(self.ik_attempts)

    def recover_blocked_step(self,index,joints,point,current_clearance,step):
        self.ik_attempts[index]+=1
        model,data=self.vision.model_self,self.vision.data_self
        adr=np.asarray(self.vision.adr); arms=np.concatenate(self.servo_arms)
        data.qpos[adr]=joints; mujoco.mj_kinematics(model,data)
        # Anti-windup while the position loop is blocked. Bound penetration
        # demand to 3mm relative to the calibrated hand/ball radii.
        self.hand_radius[index]=np.maximum(self.hand_radius[index],.132)
        targets=[]
        ball=np.r_[point,.11]
        for side,hand in enumerate(self.servo_hands):
            delta=data.xpos[hand]-ball; distance=np.linalg.norm(delta)
            if not .10<distance<.18: return step,0.
            desired=self.desired_hand_position(ball,delta,self.hand_radius[index,side],side,index)
            error=desired-data.xpos[hand]
            error*=min(1.,.006/max(np.linalg.norm(error),1e-8))
            targets.append(data.xpos[hand]+error)
        targets=np.asarray(targets)
        joint_ids=[model.joint('robot/'+self.servo_joint_names[j]).id for j in arms]
        lower=np.maximum(-.015,model.jnt_range[joint_ids,0]+.005-joints[arms])
        upper=np.minimum(.015,model.jnt_range[joint_ids,1]-.005-joints[arms])
        if not (lower<upper).all(): return step,0.
        def residual(delta):
            data.qpos[adr]=joints; data.qpos[adr[arms]]=joints[arms]+delta
            mujoco.mj_kinematics(model,data)
            avoid=np.array([max(.0015-mujoco.mj_geomDistance(model,data,a,b,.08,None),0.) for a,b in self.servo_pairs])
            return np.r_[80*(data.xpos[self.servo_hands]-targets).ravel(),.05*delta,120*avoid]
        fit=least_squares(residual,np.clip(np.zeros(len(arms)),lower+1e-8,upper-1e-8),bounds=(lower,upper),
            max_nfev=50,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        recovered=np.zeros_like(joints); recovered[arms]=fit.x
        for scale in (.5,.25):
            data.qpos[adr]=joints+scale*recovered; mujoco.mj_kinematics(model,data)
            gap=min(mujoco.mj_geomDistance(model,data,a,b,.08,None) for a,b in self.servo_pairs)
            if gap>=min(current_clearance,.0005)-.0001 and gap>-.0025:
                self.ik_accepted[index]+=1
                return recovered,scale
        return step,0.


class SlowIKTouchEnv(IKTouchEnv):
    force_radius_rate=.0001
