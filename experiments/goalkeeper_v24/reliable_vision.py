"""RGB-only near-sphere validation and short trusted-track fallback."""
import numpy as np
import mujoco,torch
from bundle_support import model_path
from vision import VisionEstimate
from occlusion_task import OcclusionVision
from fast_sphere_fit import fit_center


class ReliableVision(OcclusionVision):
    def __init__(self,e):
        super().__init__(e)
        saved=torch.load(model_path('ball_vision_v17'),map_location=e.device,weights_only=True)
        self.model.load_state_dict(saved['model_state'])
        self.raw_point=self.point.clone()
        self.control_quality=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)
        self.near=self.control_quality.clone()
        self.trusted_point=self.point.clone();self.trusted_velocity=self.point.clone()
        self.trusted_at=torch.full((e.num_envs,),-1000.,device=e.device)
        self.fit_point=self.point.clone();self.attempted_at=self.trusted_at.clone()

    def reset(self,ids):
        super().reset(ids)
        if hasattr(self,'trusted_at'):
            self.trusted_at[ids]=self.attempted_at[ids]=-1000.
            self.control_quality[ids]=self.near[ids]=False
            self.trusted_point[ids]=self.trusted_velocity[ids]=0

    @torch.no_grad()
    def update(self,e):
        if self.stamp==e.common_step_counter:return
        self.point.copy_(self.legacy_point)
        VisionEstimate.update(self,e)
        self.raw_point=self.point.clone();self.legacy_point.copy_(self.point)
        raw_visible=self.visible.clone();t=e.common_step_counter*e.step_dt
        robot=e.scene['robot'].data
        feet=robot.body_link_pos_w[:,e._collect_geometry[1],:2].mean(1)-e.scene.env_origins[:,:2]
        rel=self.point-feet
        previous_recent=(t-self.trusted_at)<=.35
        predicted=self.trusted_point+self.trusted_velocity*(t-self.trusted_at).clamp(0,.35)[:,None]
        previous_rel=predicted-feet
        self.near=((rel[:,0]<.85)&(rel[:,0]>-.20)&(rel[:,1].abs()<.30)&raw_visible)
        self.near|=previous_recent&(previous_rel[:,0]<.75)&(previous_rel[:,0]>-.20)&(previous_rel[:,1].abs()<.30)
        due=self.near&raw_visible&((t-self.attempted_at)>=.099)
        ids=due.nonzero().flatten().tolist()
        if ids:
            roots=(robot.root_link_pos_w-e.scene.env_origins).cpu().numpy()
            quaternions=robot.root_link_quat_w.cpu().numpy();joints=robot.joint_pos.cpu().numpy()
            masks=self.mask.cpu().numpy();raw=self.raw_point.cpu().numpy();prior=predicted.cpu().numpy()
            recent=previous_recent.cpu().numpy()
            for i in ids:
                self.attempted_at[i]=t
                if masks[i].sum()<12:self.fit_loss[i]=1.;continue
                self.data_self.qpos[:]=self.model_self.key_qpos[0]
                self.data_self.qpos[self.root:self.root+3]=roots[i]
                self.data_self.qpos[self.root+3:self.root+7]=quaternions[i]
                self.data_self.qpos[self.adr]=joints[i]
                mujoco.mj_kinematics(self.model_self,self.data_self);mujoco.mj_camlight(self.model_self,self.data_self)
                self.renderer.update_scene(self.data_self,camera='keeper_rgb',scene_option=self.option)
                seg=self.renderer.render();valid=(seg[:,:,0]>=0)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
                robot_mask=valid&self.robot_geom[np.clip(seg[:,:,0],0,len(self.robot_geom)-1)]
                seeds=[raw[i]]
                if recent[i] and np.linalg.norm(prior[i]-raw[i])>.04:seeds.append(prior[i])
                candidates=[]
                for seed in seeds:
                    p,loss=fit_center(masks[i],robot_mask,self.data_self.cam_xpos[self.camera],self.data_self.cam_xmat[self.camera].reshape(3,3),seed)
                    if recent[i] and np.linalg.norm(p-prior[i])>.10:continue
                    candidates.append((loss,p))
                if not candidates:self.fit_loss[i]=1.;continue
                loss,p=min(candidates,key=lambda a:a[0]);self.fit_loss[i]=loss
                if loss<.035:
                    self.fit_point[i]=torch.as_tensor(p,device=e.device,dtype=self.point.dtype)
                    self.offset[i]=self.fit_point[i]-self.raw_point[i]
                    self.corrected_at[i]=t
        fit_fresh=(t-self.corrected_at>=0)&(t-self.corrected_at<=.18)&(self.fit_loss<.035)
        corrected=self.raw_point+self.offset
        fitted=self.fit_point+self.trusted_velocity*(t-self.corrected_at).clamp(0,.18)[:,None]
        point=torch.where(self.near[:,None],.75*fitted+.25*corrected,self.raw_point)
        finite=torch.isfinite(point).all(1)
        continuity=~previous_recent|((point-predicted).norm(dim=1)<.10)
        self.control_quality=raw_visible&finite&continuity&(~self.near|fit_fresh)
        self.point=torch.where(self.control_quality[:,None],point,predicted)
        # Invalid detections cannot pollute the short speed fit or long lane fit.
        slot=e.common_step_counter%12
        self.positions[:,slot]=self.point;self.history_valid[:,slot]=self.control_quality
        self.age=(t-self.trusted_at).clamp(0,10)
        self.age[self.control_quality]=0
        self.history_valid[self.age>.4]=False
        w=self.history_valid.float();n=w.sum(1).clamp_min(1)
        mt=(self.times*w).sum(1)/n;mp=(self.positions*w[:,:,None]).sum(1)/n[:,None]
        dt=self.times-mt[:,None];den=(dt.square()*w).sum(1)
        vel=((self.positions-mp[:,None])*dt[:,:,None]*w[:,:,None]).sum(1)/den.clamp_min(.001)[:,None]
        ready=(n>=5)&(den>.003)&self.control_quality
        self.velocity=self.trusted_velocity.clone()
        self.velocity[ready]=.65*self.trusted_velocity[ready]+.35*vel[ready].clamp(-1.5,1.5)
        self.trusted_point=torch.where(self.control_quality[:,None],self.point,self.trusted_point)
        self.trusted_velocity=torch.where(self.control_quality[:,None],self.velocity,self.trusted_velocity)
        self.trusted_at=torch.where(self.control_quality,torch.full_like(self.trusted_at,t),self.trusted_at)
        self.visible=self.control_quality.clone();self.usable=self.age<=.35;self.fit_ready=ready
