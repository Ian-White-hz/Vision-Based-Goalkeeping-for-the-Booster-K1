"""Proprioceptive restraint against outward foot drift during upright handoff."""
import torch
from controller import BalancedRawEnv
from spacing_touch_task import SpacingTouchEnv as env_class,make_cfg,measurements


class AnchorWidthRawEnv(BalancedRawEnv):
    def step(self,actions):
        if not hasattr(self,'_bridge_stage'): return super().step(actions)
        d=self.scene['robot'].data
        if not hasattr(self,'_width_delta'):
            self._width_delta=torch.zeros(self.num_envs,device=self.device)
            self._width_feet=self.scene['robot'].find_bodies(('left_foot_link','right_foot_link'),preserve_order=True)[0]
            names=self.action_manager.get_term('joint_pos').target_names
            self._width_hips=[names.index(side+'_Hip_Roll') for side in ('Left','Right')]
            self._width_ankles=[names.index(side+'_Ankle_Roll') for side in ('Left','Right')]
        feet=d.body_link_pos_w[:,self._width_feet]
        width=(feet[:,0,1]-feet[:,1,1]).abs()
        anchor=(self._bridge_anchor[:,0,1]-self._bridge_anchor[:,1,1]).abs()
        tilt=torch.acos((-d.projected_gravity_b[:,2]).clamp(-1,1))
        upright=((.35-tilt)/.20).clamp(0,1)
        active=self._bridge_stage==2
        desired=(3.*(width-anchor).clamp(min=0)).clamp(max=.05)*upright*active
        self._width_delta=.9*self._width_delta+.1*desired
        self._width_delta[self.episode_length_buf==0]=0
        term=self.action_manager.get_term('joint_pos')
        target=term.offset+term.scale*actions
        target=target.clone(); correction=self._width_delta
        target[:,self._width_hips[0]]-=correction
        target[:,self._width_hips[1]]+=correction
        # Local Jacobian check: hip roll gives ~.338m/rad foot translation;
        # opposing ankle roll cancels its ~.926rad/rad foot-tilt component.
        target[:,self._width_ankles[0]]+=.9256*correction
        target[:,self._width_ankles[1]]-=.9256*correction
        return super().step((target-term.offset)/term.scale)
