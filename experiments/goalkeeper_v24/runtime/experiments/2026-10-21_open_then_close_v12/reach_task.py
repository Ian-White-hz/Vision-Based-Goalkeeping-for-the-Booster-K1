"""RGB-position-conditioned arm collection; no perfect ball state in control."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode=True
import torch
from tensordict import TensorDict
from rolling_task import IncomingEnv,make_cfg,measurements
from bridge import book
HERE=Path(__file__).resolve().parent


class ReachTable:
    def __init__(self,env,source_name="reach_paths_surface.json"):
        self.env=env
        source=json.loads((HERE/source_name).read_text())
        self.names=source['joint_names']
        self.knees=[self.names.index(n) for n in ('Left_Knee_Pitch','Right_Knee_Pitch')]
        self.widths=torch.tensor(source['widths'],device=env.device)
        self.xs=torch.tensor(source['ball_x'],device=env.device)
        self.minimum_width,self.maximum_width=source['widths'][0],source['widths'][-1]
        paths={(p['width'],p['ball_x']):p['frames'] for p in source['paths']}
        self.values=torch.tensor([[[[f['root_x'],f['root_z'],f['root_pitch'],*f['joint_pos'],*f['gravity_offset']]
            for f in paths[(w,x)]] for x in source['ball_x']] for w in source['widths']],device=env.device)
        env._incoming_reach_x=torch.full((env.num_envs,),.20,device=env.device)

    def at(self,f):
        e=self.env
        p=f.clamp(0,1)*(self.values.shape[2]-1)
        i=p.long().clamp(max=self.values.shape[2]-2); u=(p-i)[:,None]
        width=e._bridge_width.clamp(self.minimum_width,self.maximum_width)
        k=torch.searchsorted(self.widths,width.contiguous()).clamp(1,len(self.widths)-1)
        v=((width-self.widths[k-1])/(self.widths[k]-self.widths[k-1]))[:,None]
        x=e._incoming_reach_x.clamp(float(self.xs[0]),float(self.xs[-1]))
        j=torch.searchsorted(self.xs,x.contiguous()).clamp(1,len(self.xs)-1)
        z=((x-self.xs[j-1])/(self.xs[j]-self.xs[j-1]))[:,None]
        def sample(a,b): return self.values[a,b,i]*(1-u)+self.values[a,b,i+1]*u
        low=sample(k-1,j-1)*(1-z)+sample(k-1,j)*z
        high=sample(k,j-1)*(1-z)+sample(k,j)*z
        return low*(1-v)+high*v


class ReachIncomingEnv(IncomingEnv):
    def __init__(self,env):
        super().__init__(env)
        env._bridge_table=ReachTable(env)
        self.reach_quiet=torch.zeros(self.num_envs,device=self.device)
        self.track_positions=torch.zeros((self.num_envs,64,2),device=self.device)
        self.track_times=torch.zeros((self.num_envs,64),device=self.device)
        self.track_valid=torch.zeros((self.num_envs,64),device=self.device,dtype=torch.bool)
        self.lane_velocity=torch.zeros((self.num_envs,2),device=self.device)
        self.lane_ready=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.bridge_amplitude=torch.tensor([.08 if 'Shoulder' in n or 'Elbow' in n else .025
            for n in self.action_names],device=self.device)

    def get_observations(self):
        base=super().get_observations()
        if not hasattr(self,'reach_quiet'): return base
        extra=torch.stack((self.env._incoming_reach_x/.30,(self.reach_quiet/.4).clamp(max=1),
            self.lane_ready.float()),1)
        actor=torch.cat((base['actor'],extra),1)
        return TensorDict({'actor':actor,'critic':torch.cat((actor,base['critic'][:,-6:]),1)},batch_size=[self.num_envs])

    def step(self,actions):
        obs,reward,done,extras=super().step(actions)
        m=measurements(self.env)
        from bridge import phase
        active=~done.bool(); late=phase(self.env)>.70
        # Privileged distances below supervise learning only. They never choose
        # the reference, trigger descent, or enter the actor observations.
        accuracy=torch.exp(-((m['distances']-.133)/.025).square()).mean(1)
        excess=((m['forces']-12.)/12.).clamp(min=0,max=3).sum(1)
        lift=((m['ball_pos'][:,2]-.14)/.05).clamp(min=0,max=3)
        reward+=self.env.step_dt*(2.5*late*accuracy-.3*late*excess-4*lift)*active
        extras['log']['Reach/surface_accuracy']=accuracy
        return obs,reward,done,extras

    def target_point(self,v,feet,visible):
        e=self.env; t=e.common_step_counter*e.step_dt
        reset=e.episode_length_buf==0
        self.track_valid[reset]=False; self.reach_quiet[reset]=0
        e._incoming_reach_x[reset]=.20
        slot=e.common_step_counter%64
        self.track_positions[:,slot]=v.point
        self.track_times[:,slot]=t
        self.track_valid[:,slot]=v.visible&visible
        w=self.track_valid.float()
        def fit(weights):
            n=weights.sum(1).clamp_min(1)
            mean_t=(self.track_times*weights).sum(1)/n
            mean_p=(self.track_positions*weights[:,:,None]).sum(1)/n[:,None]
            dt=self.track_times-mean_t[:,None]
            denominator=(dt.square()*weights).sum(1)
            velocity=((self.track_positions-mean_p[:,None])*dt[:,:,None]*weights[:,:,None]).sum(1)/denominator.clamp_min(.001)[:,None]
            return mean_t,mean_p,velocity,denominator
        mt,mp,vel,den=fit(w)
        residual=(self.track_positions-(mp[:,None]+vel[:,None]*(self.track_times-mt[:,None])[:,:,None])).norm(dim=2)
        robust=w*(.04/residual.clamp_min(.04))
        mt,mp,vel,den=fit(robust)
        self.lane_velocity=vel.clamp(-1.5,1.5)
        self.lane_ready=(w.sum(1)>=20)&(den>.12)&visible
        current=mp+self.lane_velocity*(t-mt)[:,None]
        approaching=self.lane_ready&(self.lane_velocity[:,0]<-.04)
        tta=((current[:,0]-.20)/(-self.lane_velocity[:,0]).clamp_min(.04)).clamp(0,12)
        lane=torch.where(approaching,current[:,1]+self.lane_velocity[:,1]*tta,v.point[:,1])
        # Guard the same fore/aft plane. Arm references adapt to the stopped ball.
        x=torch.full_like(lane,float(book(e)['feet'][:,0].mean()))
        return torch.stack((x,lane.clamp(-1.07,1.07)),1)

    def collection_ready(self,v,feet,visible):
        e=self.env
        relative=v.point-feet[:,:2]
        near=visible&v.fit_ready&(v.velocity[:,:2].norm(dim=1)<.07)
        near&=(relative[:,0]>.005)&(relative[:,0]<.33)&(relative[:,1].abs()<.045)
        near&=e._incoming_touched&(e._bridge_stage==2)
        self.reach_quiet=torch.where(near,self.reach_quiet+e.step_dt,0.)
        foot0=float(book(e)['feet'][:,0].mean())
        target=(relative[:,0]+foot0).clamp(-.04,.28)
        # A shin may displace a blocked ball during descent. Continue following
        # its RGB estimate with bounded reference speed rather than freezing the
        # pre-descent point. No truth position is used here.
        track=visible&v.fit_ready&(relative[:,0]>-.08)&(relative[:,0]<.36)
        track&=(relative[:,1].abs()<.10)&(v.velocity[:,:2].norm(dim=1)<.15)
        update=near|(track&e._incoming_ready&(e._bridge_stage==2))
        delta=(.15*(target-e._incoming_reach_x)).clamp(-.05*e.step_dt,.05*e.step_dt)
        e._incoming_reach_x=torch.where(update,e._incoming_reach_x+delta,e._incoming_reach_x)
        return self.reach_quiet>=.4

    def forward_command(self,error,velocity):
        # This frozen skill was accepted for lateral walking. Fore/aft chase
        # caused a guard-region exit in the preceding trial; keep guarding x.
        return torch.zeros_like(error[:,0])

    def endpoint_weight(self):
        # The old endpoint skill was only trained near x=.20. Other reaches use
        # the new reference and a separately trained bounded residual.
        return torch.exp(-((self.env._incoming_reach_x-.20)/.035).square())
