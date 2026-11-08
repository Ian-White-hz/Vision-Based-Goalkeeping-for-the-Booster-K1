"""Proprioceptive balance adapter; RGB-only ball targeting and real contacts."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=(BUNDLE_ROOT / 'runtime/experiments/2026-10-19_lateral_stationary_handoff_v3')
sys.path.insert(0,str(OLD))
import torch
from mjlab.envs import ManagerBasedRlEnv
from bridge import BridgeEnv,make_cfg,geometry,measurements,phase,table,sha
from bridge import preserved as legacy_preserved


def preserved():
    return bundle_integrity()


class BalancedRawEnv(ManagerBasedRlEnv):
    def step(self,actions):
        if not hasattr(self,'_bridge_stage'): return super().step(actions)
        m=geometry(self)
        relative=m['com'][:,:2]-m['feet'][:,:,:2].mean(1)
        if not hasattr(self,'_balance_previous'):
            self._balance_previous=relative.clone()
            self._balance_velocity=torch.zeros_like(relative)
            self._balance_delta=torch.zeros_like(relative)
            self._balance_p=torch.full((self.num_envs,),2.5,device=self.device)
            self._balance_d=torch.full((self.num_envs,),.5,device=self.device)
        reset=self.episode_length_buf==0
        velocity=(relative-self._balance_previous)/self.step_dt
        self._balance_velocity=.7*self._balance_velocity+.3*velocity
        self._balance_velocity[reset]=0
        self._balance_previous=relative.clone()
        error=relative-relative.new_tensor((.025,0.))
        desired=self._balance_p[:,None]*error+self._balance_d[:,None]*self._balance_velocity
        desired[:,1]*=-1
        desired=desired.clamp(desired.new_tensor((-.10,-.06)),desired.new_tensor((.10,.06)))
        active=self._bridge_stage==2
        desired=torch.where(active[:,None],desired,0.)
        self._balance_delta=.85*self._balance_delta+.15*desired
        self._balance_delta[reset]=0
        term=self.action_manager.get_term('joint_pos')
        if not hasattr(self,'_balance_ankles'):
            self._balance_ankles=([term.target_names.index(s+'_Ankle_Pitch') for s in ('Left','Right')],
                [term.target_names.index(s+'_Ankle_Roll') for s in ('Left','Right')])
        target=term.offset+term.scale*actions
        target=target.clone()
        for axis,ids in enumerate(self._balance_ankles): target[:,ids]+=self._balance_delta[:,axis,None]
        limits=self.scene['robot'].data.joint_pos_limits[:,term.target_ids]
        target=target.clamp(limits[:,:,0]+.005,limits[:,:,1]-.005)
        return super().step((target-term.offset)/term.scale)


class ZeroEntry(torch.nn.Module):
    def forward(self,obs): return obs['actor'].new_zeros((obs.batch_size[0],22))


class FreezeGait(torch.nn.Module):
    def __init__(self,model,env):
        super().__init__(); self.model=model; self.env=env
        self.last=torch.zeros((env.num_envs,22),device=env.device)

    def forward(self,obs):
        out=self.model(obs)
        frozen=self.env.unwrapped._bridge_stage==2
        self.last[~frozen]=out[~frozen]
        return torch.where(frozen[:,None],self.last,out)


class BalancedKeeperEnv(BridgeEnv):
    def __init__(self,env):
        super().__init__(env)
        self.entry_teacher=ZeroEntry()
        self.gait_teacher=FreezeGait(self.gait_teacher,self)
