"""Read-only reuse of the existing RGB detector and calibrated ground projection."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
import math
from pathlib import Path
import sys
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from goalkeeper_sim.full_goal.env_cfg import head_camera_quaternion

PROJECT=(BUNDLE_ROOT / 'runtime')
sys.path.insert(0,str(PROJECT/'simulator'))
from train_ball_vision import BallVision
VISION=model_path('ball_vision')


class VisionEstimate:
    def __init__(self,env):
        self.n,self.device=env.num_envs,env.device
        saved=torch.load(VISION,map_location=env.device,weights_only=True)
        self.model=BallVision().to(env.device).eval().requires_grad_(False)
        self.model.load_state_dict(saved['model_state'])
        assert (saved['width'],saved['height'])==(128,96)
        self.x=torch.linspace(-1+1/128,1-1/128,128,device=env.device)[None,None,:]
        self.y=torch.linspace(-1+1/96,1-1/96,96,device=env.device)[None,:,None]
        self.head=env.scene['robot'].find_bodies('Head_2')[0][0]
        self.positions=torch.zeros((self.n,12,2),device=env.device)
        self.times=torch.zeros((self.n,12),device=env.device)
        self.history_valid=torch.zeros((self.n,12),device=env.device,dtype=torch.bool)
        self.point=torch.zeros((self.n,2),device=env.device)
        self.velocity=torch.zeros_like(self.point)
        self.age=torch.ones(self.n,device=env.device)*10
        self.visible=torch.zeros(self.n,device=env.device,dtype=torch.bool)
        self.usable=self.visible.clone()
        self.fit_ready=self.visible.clone()
        self.mask=torch.zeros((self.n,96,128),device=env.device,dtype=torch.bool)
        self.stamp=-1

    def reset(self,ids):
        self.history_valid[ids]=False
        self.point[ids]=self.velocity[ids]=0
        self.age[ids]=10
        self.visible[ids]=self.usable[ids]=self.fit_ready[ids]=False
        self.stamp=-1

    @torch.no_grad()
    def update(self,env):
        if self.stamp==env.common_step_counter:
            return
        env.sim.sense()
        rgb=env.scene['keeper_rgb'].data.rgb
        self.mask=self.model(rgb.permute(0,3,1,2).float()/255).sigmoid()[:,0]>.5
        weights=self.mask.float()
        count=weights.sum((1,2))
        u=(weights*self.x).sum((1,2))/count.clamp_min(1)
        v=(weights*self.y).sum((1,2))/count.clamp_min(1)
        data=env.scene['robot'].data
        q=data.body_link_quat_w[:,self.head]
        camera=data.body_link_pos_w[:,self.head]+quat_apply(q,q.new_tensor((.12,0,.07)).expand(self.n,-1))
        rotation=quat_mul(q,q.new_tensor(head_camera_quaternion()).expand(self.n,-1))
        tangent=math.tan(math.radians(100)/2)
        ray=quat_apply(rotation,torch.stack((u*tangent*128/96,-v*tangent,-torch.ones_like(u)),1))
        distance=(env.scene.env_origins[:,2]+.11-camera[:,2])/ray[:,2].clamp(max=-1e-5)
        valid=(count>=4)&(ray[:,2]<-.03)&(distance>0)&(distance<6)
        point=(camera+distance[:,None]*ray)[:,:2]-env.scene.env_origins[:,:2]
        self.age+=env.step_dt
        self.age[valid]=0
        self.history_valid[self.age>.4]=False
        slot=env.common_step_counter%12
        self.positions[:,slot]=point
        self.times[:,slot]=env.common_step_counter*env.step_dt
        self.history_valid[:,slot]=valid
        w=self.history_valid.float()
        n=w.sum(1).clamp_min(1)
        mean_t=(self.times*w).sum(1)/n
        mean_p=(self.positions*w[:,:,None]).sum(1)/n[:,None]
        dt=self.times-mean_t[:,None]
        denominator=(dt.square()*w).sum(1)
        fit=((self.positions-mean_p[:,None])*dt[:,:,None]*w[:,:,None]).sum(1)/denominator.clamp_min(.001)[:,None]
        ready=(n>=5)&(denominator>.003)&valid
        self.velocity[ready]=.65*self.velocity[ready]+.35*fit[ready].clamp(-1.5,1.5)
        self.point[valid]=point[valid]
        self.point[~valid]+=self.velocity[~valid]*env.step_dt
        self.visible,self.usable,self.fit_ready=valid,self.age<=.35,ready
        self.stamp=env.common_step_counter
