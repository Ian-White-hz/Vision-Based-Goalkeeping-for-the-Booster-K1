"""Ball-sized stance references, normal motor handoff and bounded PPO residuals."""
import json
import torch
from bundle_support import BUNDLE_ROOT
from priority_control import PriorityCaptureEnv
from channel_geometry import ChannelGeometry
from bridge import phase,table

def reference_width(measured,target,max_change,entry_floor,minimum,maximum):
    cap=torch.maximum(torch.full_like(measured,target),measured-max_change)
    limited=torch.minimum(measured,cap)
    return torch.maximum(torch.maximum(limited,torch.full_like(measured,target)),entry_floor).clamp(minimum,maximum)

class ChannelTable:
    def __init__(self,original,owner):
        self.__dict__.update(original.__dict__);self.owner=owner
    def at(self,f):
        e=self.env;p=f.clamp(0,1)*(self.values.shape[2]-1)
        i=p.long().clamp(max=self.values.shape[2]-2);u=(p-i)[:,None]
        width=self.owner.reference_width()
        k=torch.searchsorted(self.widths,width.contiguous()).clamp(1,len(self.widths)-1)
        v=((width-self.widths[k-1])/(self.widths[k]-self.widths[k-1]))[:,None]
        x=e._incoming_reach_x.clamp(float(self.xs[0]),float(self.xs[-1]));j=torch.searchsorted(self.xs,x.contiguous()).clamp(1,len(self.xs)-1)
        z=((x-self.xs[j-1])/(self.xs[j]-self.xs[j-1]))[:,None]
        def sample(a,b):return self.values[a,b,i]*(1-u)+self.values[a,b,i+1]*u
        low=sample(k-1,j-1)*(1-z)+sample(k-1,j)*z
        high=sample(k,j-1)*(1-z)+sample(k,j)*z
        return low*(1-v)+high*v

class ChannelCaptureEnv(PriorityCaptureEnv):
    def __init__(self,e):
        super().__init__(e)
        self.channel_config=json.loads((BUNDLE_ROOT/'channel_calibration.json').read_text())
        assert self.channel_config['ball_radius_m']==.11,'Recalibrate before changing task ball size'
        self.channel_entry_floor=torch.zeros(self.num_envs,device=self.device)
        self.channel_front=torch.zeros_like(self.channel_entry_floor);self.channel_rear=torch.zeros_like(self.channel_front)
        self.channel_gap=torch.zeros_like(self.channel_front);self.channel_foot_width=torch.zeros_like(self.channel_front)
        self.channel_measured=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.channel_stamp=-1
        self.channel_geometry=ChannelGeometry(list(e.scene['robot'].joint_names),self.channel_config['ball_radius_m'])
        e._bridge_table=ChannelTable(table(e),self);self.open_table=ChannelTable(self.open_table,self)

    def reference_width(self):
        c=self.channel_config
        return reference_width(self.env._bridge_width,c['reference_width_target_m'],.042,self.channel_entry_floor,
                               float(self.open_table.widths[0]),float(self.open_table.widths[-1]))

    def update_channel_metrics(self):
        e=self.env
        if e.common_step_counter==self.channel_stamp:return
        if e.common_step_counter%5:return
        self.channel_stamp=e.common_step_counter
        active=(e._bridge_stage>=1)&(e.episode_length_buf>0)
        self.channel_measured=active.clone()
        ids=active.nonzero().flatten().tolist()
        if not ids:return
        d=e.scene['robot'].data
        roots=(d.root_link_pos_w-e.scene.env_origins).cpu().numpy();q=d.root_link_quat_w.cpu().numpy();joints=d.joint_pos.cpu().numpy()
        for i in ids:
            self.channel_geometry.pose(roots[i],q[i],joints[i]);m=self.channel_geometry.measure()
            self.channel_front[i]=m['front_clearance_m'];self.channel_rear[i]=m['rear_route_clearance_m']
            self.channel_gap[i]=m['equivalent_gap_m'];self.channel_foot_width[i]=m['foot_width_m']

    def prepare_interception_update(self,feet,visible):
        super().prepare_interception_update(feet,visible)
        e=self.env;reset=e.episode_length_buf==0;self.channel_entry_floor[reset]=0
        self.update_channel_metrics()
        supported=e.scene['feet_ground_contact'].data.found.flatten(1).all(1)
        # Only the upright handoff may request a small opening if the entrance
        # is obstructed. Existing narrow stances with a free entrance are kept.
        early=(e._bridge_stage>=1)&(phase(e)<.05)&supported&self.channel_measured
        obstructed=early&(self.channel_front<self.channel_config['entry_clearance_min_m'])
        proposed=(self.reference_width()+.004).clamp(max=self.channel_config['maximum_reference_width_m'])
        self.channel_entry_floor=torch.where(obstructed,torch.maximum(self.channel_entry_floor,proposed),self.channel_entry_floor)

    def step(self,a):
        obs,reward,done,extras=super().step(a)
        self.update_channel_metrics();e=self.env
        active=~done.bool()&(e._bridge_stage==2)&self.channel_measured
        # Robot-geometry reward only. Actual goal penalty is an offline outcome
        # label, never a perception input or stance target.
        too_wide=((self.channel_rear-self.channel_config['rear_margin_target_m'])/.02).clamp(0,3)
        too_narrow=((self.channel_config['rear_margin_target_m']-self.channel_config['allowed_overclosure_m']-self.channel_rear)/.02).clamp(0,3)
        too_early=((.008-self.channel_front)/.02).clamp(0,3)
        reward+=e.step_dt*(.5-2.*too_wide-2.*too_narrow-2.*too_early)*active
        reward-=80.*(done.bool()&e._pose_outcome.last['goal']).float()
        extras['log']['Channel/equivalent_gap_m']=self.channel_gap
        extras['log']['Channel/reference_width_m']=self.reference_width()
        extras['log']['Channel/front_clearance_m']=self.channel_front
        self.channel_entry_floor[done.bool()]=0;self.channel_measured[done.bool()]=False
        return obs,reward,done,extras

env_class=ChannelCaptureEnv
