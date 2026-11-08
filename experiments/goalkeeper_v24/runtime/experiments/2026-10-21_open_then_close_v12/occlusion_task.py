"""Near-ball RGB sphere fitting with a proprioceptive self-occlusion model.

The private CPU model only projects the robot's measured root/joints. Ball
geometries are hidden; live ball state is never read by the estimator/controller.
It is a kinematic rendering aid, not a state setter for the live simulation.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import copy
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import mujoco
import torch
from reach_task import ReachIncomingEnv,make_cfg,measurements
from vision import VisionEstimate
from bridge import book
from fast_sphere_fit import fit_center


class OcclusionVision(VisionEstimate):
    def __init__(self,env):
        super().__init__(env)
        self.legacy_point=self.point.clone(); self.offset=self.point.clone()
        self.corrected_at=torch.full((self.n,),-1000.,device=self.device)
        self.fit_loss=torch.ones(self.n,device=self.device)
        self.model_self=copy.copy(env.sim.mj_model)
        self.data_self=mujoco.MjData(self.model_self)
        self.root=self.model_self.joint('robot/floating_base_joint').qposadr[0]
        self.adr=[self.model_self.joint('robot/'+n).qposadr[0] for n in env.scene['robot'].joint_names]
        for g in range(self.model_self.ngeom):
            if self.model_self.body(int(self.model_self.geom_bodyid[g])).name.startswith('ball/'):
                self.model_self.geom_group[g]=5
        self.option=mujoco.MjvOption(); self.option.geomgroup[5]=0
        self.robot_geom=np.array([self.model_self.body(int(self.model_self.geom_bodyid[g])).name.startswith('robot/')
            for g in range(self.model_self.ngeom)])
        self.camera=self.model_self.camera('keeper_rgb').id
        self.renderer=mujoco.Renderer(self.model_self,height=96,width=128)
        self.renderer.enable_segmentation_rendering()

    def reset(self,ids):
        super().reset(ids)
        self.legacy_point[ids]=self.offset[ids]=0
        self.corrected_at[ids]=-1000; self.fit_loss[ids]=1

    @torch.no_grad()
    def update(self,env):
        if self.stamp==env.common_step_counter: return
        self.point.copy_(self.legacy_point)
        super().update(env)
        self.legacy_point.copy_(self.point)
        robot=env.scene['robot'].data
        feet=robot.body_link_pos_w[:,env._collect_geometry[1],:2].mean(1)-env.scene.env_origins[:,:2]
        relative=self.point-feet
        anticipatory=getattr(env,'_anticipatory_receive',False)
        near=(env._bridge_stage==2)&(env._incoming_touched|anticipatory)&self.visible&self.fit_ready
        near&=(relative[:,0]>-.10)&(relative[:,0]<( .70 if anticipatory else .40))&(relative[:,1].abs()<.12)
        near&=(self.velocity.norm(dim=1)<(.60 if anticipatory else .15))
        t=env.common_step_counter*env.step_dt
        due=near&((t-self.corrected_at)>=.099)
        ids=due.nonzero().flatten().tolist()
        if ids:
            roots=(robot.root_link_pos_w-env.scene.env_origins).cpu().numpy()
            quaternions=robot.root_link_quat_w.cpu().numpy(); joints=robot.joint_pos.cpu().numpy()
            masks=self.mask.cpu().numpy(); initial=self.legacy_point.cpu().numpy()
            for i in ids:
                if int(masks[i].sum())<12: continue
                # The hidden ball remains at an unused private-model default.
                # Only known robot configuration is copied for self projection.
                self.data_self.qpos[:]=self.model_self.key_qpos[0]
                self.data_self.qpos[self.root:self.root+3]=roots[i]
                self.data_self.qpos[self.root+3:self.root+7]=quaternions[i]
                self.data_self.qpos[self.adr]=joints[i]
                mujoco.mj_kinematics(self.model_self,self.data_self)
                mujoco.mj_camlight(self.model_self,self.data_self)
                self.renderer.update_scene(self.data_self,camera='keeper_rgb',scene_option=self.option)
                seg=self.renderer.render()
                valid=(seg[:,:,0]>=0)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                self_mask=valid&self.robot_geom[np.clip(seg[:,:,0],0,len(self.robot_geom)-1)]
                point,loss=fit_center(masks[i],self_mask,self.data_self.cam_xpos[self.camera],
                    self.data_self.cam_xmat[self.camera].reshape(3,3),initial[i])
                delta=point-initial[i]; self.fit_loss[i]=loss
                if loss<.05 and np.linalg.norm(delta)<.07:
                    self.offset[i]=.4*self.offset[i]+.6*torch.as_tensor(delta,device=self.device,dtype=self.point.dtype)
                    self.corrected_at[i]=t
        use=near&((t-self.corrected_at)<.4)
        self.point[use]+=self.offset[use]

    def close(self): self.renderer.close()


class OcclusionIncomingEnv(ReachIncomingEnv):
    def __init__(self,env):
        super().__init__(env)
        self.vision=OcclusionVision(env)

    def collection_ready(self,v,feet,visible):
        e=self.env; relative=v.point-feet[:,:2]
        near=visible&v.fit_ready&(v.velocity.norm(dim=1)<.07)
        near&=(relative[:,0]>-.06)&(relative[:,0]<.33)&(relative[:,1].abs()<.045)
        near&=e._incoming_touched&(e._bridge_stage==2)
        self.reach_quiet=torch.where(near,self.reach_quiet+e.step_dt,0.)
        foot0=float(book(e)['feet'][:,0].mean())
        target=(relative[:,0]+foot0).clamp(-.04,.28)
        track=visible&v.fit_ready&(relative[:,0]>-.08)&(relative[:,0]<.36)
        track&=(relative[:,1].abs()<.10)&(v.velocity.norm(dim=1)<.15)
        update=near|(track&e._incoming_ready&(e._bridge_stage==2))
        delta=(.15*(target-e._incoming_reach_x)).clamp(-.05*e.step_dt,.05*e.step_dt)
        e._incoming_reach_x=torch.where(update,e._incoming_reach_x+delta,e._incoming_reach_x)
        return self.reach_quiet>=.4

    def close(self):
        self.vision.close()
        super().close()
