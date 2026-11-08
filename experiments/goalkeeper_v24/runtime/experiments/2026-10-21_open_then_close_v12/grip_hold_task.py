"""A tactile-established hold retains its last RGB target, never the ball state."""
import numpy as np
import torch
from spacing_touch_task import SpacingTouchEnv,make_cfg,measurements
from damped_touch_task import IKTouchEnv
from anchor_width_task import AnchorWidthRawEnv as raw_class


class GripHold:
    def __init__(self,env):
        super().__init__(env)
        self.grip_hold=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.grip_rgb_anchor=torch.zeros((self.num_envs,2),device=self.device)
        self.grip_reference_x=torch.zeros(self.num_envs,device=self.device)
        self.grip_rgb_time=torch.zeros(self.num_envs,device=self.device)
        self.grip_normals=np.zeros((self.num_envs,2,3))
        self.grip_contact_dwell=np.zeros(self.num_envs)
        self.grip_release_dwell=np.zeros(self.num_envs)

    def adjust_motor_targets(self,target):
        reset=self.env._bridge_stage<2
        self.grip_hold[reset]=False
        ids=reset.cpu().numpy()
        self.grip_contact_dwell[ids]=self.grip_release_dwell[ids]=0.
        return super().adjust_motor_targets(target)

    def perception_ready(self,vision,time):
        return super().perception_ready(vision,time)|self.grip_hold

    def control_ball_points(self,vision):
        return torch.where(self.grip_hold[:,None],self.grip_rgb_anchor,vision.point)

    def collection_ready(self,v,feet,visible):
        ready=super().collection_ready(v,feet,visible)
        held=self.grip_hold&(self.env._bridge_stage==2)
        self.env._incoming_reach_x[held]=self.grip_reference_x[held]
        return ready

    def desired_hand_position(self,ball,delta,radius,side,index=None):
        if index is None or not self.grip_hold[index]:
            return super().desired_hand_position(ball,delta,radius,side,index)
        offset=self.grip_normals[index,side]*radius
        offset[1]=(1. if side==0 else -1.)*max(abs(offset[1]),.030)
        height2=radius**2-offset[0]**2-offset[1]**2
        if height2>.0025: offset[2]=np.sqrt(height2)
        return ball+offset

    def update_hand_radius(self,index,forces):
        held=bool(self.grip_hold[index])
        if not held:
            self.grip_contact_dwell[index]=self.grip_contact_dwell[index]+.1 if (forces>.25).all() else 0.
            if self.grip_contact_dwell[index]>=.199:
                self.grip_hold[index]=True; held=True
                self.grip_rgb_anchor[index]=self.vision.point[index]
                self.grip_reference_x[index]=self.env._incoming_reach_x[index]
                self.grip_rgb_time[index]=self.vision.corrected_at[index]
                ball=np.r_[self.vision.point[index].cpu().numpy(),.11]
                delta=self.vision.data_self.xpos[self.servo_hands]-ball
                self.grip_normals[index]=delta/np.linalg.norm(delta,axis=1,keepdims=True)
        if held:
            self.grip_release_dwell[index]=self.grip_release_dwell[index]+.1 if (forces<.05).any() else 0.
            if self.grip_release_dwell[index]>=.299:
                self.grip_hold[index]=False
                self.grip_contact_dwell[index]=0.
                self.grip_release_dwell[index]=0.
                return super().update_hand_radius(index,forces)
            # A band avoids continually opening a secure grasp to chase one
            # precise force. Use a faster release only for excess pressure.
            change=np.where(forces<.4,forces-.4,np.where(forces>1.2,forces-1.2,0.))*.0001
            change=np.where(forces>2.,np.clip(forces-2.,0.,3.)*.0004,change)
            self.hand_radius[index]=np.clip(self.hand_radius[index]+change,.129,.137)
            return
        return super().update_hand_radius(index,forces)


class GripTouchEnv(GripHold,SpacingTouchEnv): pass
class GripIKTouchEnv(GripHold,IKTouchEnv): pass
