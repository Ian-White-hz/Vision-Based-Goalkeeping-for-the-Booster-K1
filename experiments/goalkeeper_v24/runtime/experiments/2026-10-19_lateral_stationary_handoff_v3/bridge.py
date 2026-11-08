"""RGB lateral alignment and physically gated handoff to accepted entry skill."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=(BUNDLE_ROOT / 'runtime')
ENTRY=PROJECT/'experiments/2026-10-19_standing_entry_balance_v3'
GAIT=PROJECT/'experiments/2026-10-18_natural_lateral_v2'
sys.path.insert(0,str(ENTRY))
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.managers import EventTermCfg,TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from entry import EntryEnv,reset_standing,pose_cfg,pose_measurements,book,PoseOutcome,SETTLE,DURATION,HOLD,sha
from entry import preserved as entry_preserved
from train_entry import config as entry_config
from collection import CollectionEnv

ENTRY_MODEL=model_path('standing_entry')
GAIT_MODEL=model_path('lateral_gait')
HANDOFF_SETTLE=2.5


class StanceTable:
    def __init__(self,env):
        self.env=env
        source=json.loads((HERE/'stance_paths.json').read_text())
        self.names=source['paths'][0]['joint_names']
        self.knees=[self.names.index(n) for n in ('Left_Knee_Pitch','Right_Knee_Pitch')]
        self.widths=torch.tensor(source['widths'],device=env.device)
        self.minimum_width,self.maximum_width=source['widths'][0],source['widths'][-1]
        self.values=torch.tensor([[[f['root_x'],f['root_z'],f['root_pitch'],
            *f['joint_pos'],*f['gravity_offset']] for f in p['frames']] for p in source['paths']],device=env.device)

    def at(self,f):
        init_state(self.env)
        p=f.clamp(0,1)*(self.values.shape[1]-1)
        i=p.long().clamp(max=self.values.shape[1]-2); u=(p-i)[:,None]
        width=self.env._bridge_width.clamp(self.minimum_width,self.maximum_width)
        k=torch.searchsorted(self.widths,width).clamp(1,len(self.widths)-1)
        v=((width-self.widths[k-1])/(self.widths[k]-self.widths[k-1]))[:,None]
        a=self.values[k-1,i]*(1-u)+self.values[k-1,i+1]*u
        b=self.values[k,i]*(1-u)+self.values[k,i+1]*u
        return a*(1-v)+b*v


def table(env):
    if not hasattr(env,'_bridge_table'): env._bridge_table=StanceTable(env)
    return env._bridge_table


def init_state(env):
    if hasattr(env,'_bridge_stage'): return
    for name in ('clock','quiet','distance','start_y','last_error','recenter'):
        setattr(env,'_bridge_'+name,torch.zeros(env.num_envs,device=env.device))
    env._bridge_stage=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    env._bridge_cmd=torch.zeros((env.num_envs,3),device=env.device)
    env._bridge_target=torch.zeros((env.num_envs,2),device=env.device)
    env._bridge_anchor=torch.zeros((env.num_envs,2,3),device=env.device)
    env._bridge_braked=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    env._bridge_unsafe=env._bridge_braked.clone()
    env._bridge_left=env._bridge_braked.clone()
    env._bridge_width=torch.full((env.num_envs,),.1924,device=env.device)


def phase(env):
    init_state(env)
    u=((env._bridge_clock-HANDOFF_SETTLE)/DURATION).clamp(0,1)
    return u**3*(10-15*u+6*u.square())


def reset_robot(env,env_ids):
    reset_standing(env,env_ids)
    init_state(env)
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    for name in ('clock','quiet','distance','start_y','last_error','recenter','stage'):
        getattr(env,'_bridge_'+name)[ids]=0
    env._bridge_cmd[ids]=env._bridge_target[ids]=0
    env._bridge_anchor[ids]=book(env)['feet'][None]+env.scene.env_origins[ids,None]
    env._bridge_braked[ids]=env._bridge_unsafe[ids]=False
    env._bridge_width[ids]=.1924
    env._bridge_geometry_stamp=-1


def reset_ball(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    init_state(env)
    sign=torch.where(torch.rand(len(ids),device=env.device)<.5,-1.,1.)
    if getattr(env,'_balanced_eval',False): sign=torch.where(ids%2==0,-1.,1.)
    state=torch.zeros((len(ids),13),device=env.device)
    state[:,0]=.20+(torch.rand(len(ids),device=env.device)-.5)*.020
    state[:,1]=sign*(.16+.06*torch.rand(len(ids),device=env.device))
    state[:,2],state[:,3]=.112,1.
    env._bridge_left[ids]=sign>0
    state[:,:3]+=env.scene.env_origins[ids]
    env.scene['ball'].write_root_state_to_sim(state,env_ids=ids)


def geometry(env):
    # Multiple actors and termination terms inspect the same physical state.
    # Cache geometry once per physics step, and invalidate on every reset.
    if getattr(env,'_bridge_geometry_stamp',-1)!=env.common_step_counter:
        env._bridge_geometry=pose_measurements(env)
        env._bridge_geometry_stamp=env.common_step_counter
    return env._bridge_geometry


def measurements(env):
    init_state(env)
    m=dict(geometry(env))
    d=env.scene['robot'].data
    ref=table(env).at(phase(env))
    collecting=env._bridge_stage==2
    foot_drift=(m['feet'][:,:,:2]-env._bridge_anchor[:,:,:2]).norm(dim=-1).max(1).values
    stable=(m['support']&(m['margin']>.008)&(m['foot_tilt']<.15)&(foot_drift<.025)
        &(d.root_link_lin_vel_w.norm(dim=1)<.06)&(d.root_link_ang_vel_w.norm(dim=1)<.35))
    illegal=env.scene['non_foot_ground_contact'].data.found.flatten(1).any(1)
    settled=env.episode_length_buf*env.step_dt>.4
    bad_collect=(m['margin']<-.035)|(m['foot_tilt']>.45)|((m['height']-ref[:,1]-.005).abs()>.11)|((m['tilt']-ref[:,2]).abs()>.35)
    fallen=(m['height']<.25)|illegal|(settled&torch.where(collecting,bad_collect,m['tilt']>.55))
    from goalkeeper_sim.full_goal.outcomes import invalid_ball,out_of_guard_region
    from goalkeeper_sim.full_goal.settings import FullGoalSettings
    failed=fallen|invalid_ball(env)|out_of_guard_region(env,FullGoalSettings())
    ball=env.scene['ball'].data
    hi=env._collect_geometry[0]
    relative=(ball.root_link_lin_vel_w-d.body_link_lin_vel_w[:,hi].mean(1)).norm(dim=1)
    ground=env.scene['ball_ground_contact'].data.found.flatten(1).any(1)
    ready=collecting&(phase(env)>.9999)&(env._bridge_clock>HANDOFF_SETTLE+DURATION+.2)
    valid=(ready&m['contacts'].all(1)&ground&stable&m['profile']&~failed
        &(m['speed']<.08)&(relative<.12)&(m['ball_pos'][:,2]>.10)&(m['ball_pos'][:,2]<.14)
        &((ball.root_link_pos_w-m['hands'].mean(1)).norm(dim=1)<.15)
        &env._bridge_braked&~env._bridge_unsafe&(env._bridge_distance>.08))
    m.update(failed=failed,fallen=fallen,valid=valid,ready=ready,reference=ref,stable=stable,foot_drift=foot_drift)
    return m


class BridgeOutcome(PoseOutcome):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        for name in ('moved','braked','unsafe','left','descent'):
            setattr(self,name,self.success.clone()); self.last[name]=self.success.clone()

    def __call__(self,env):
        fresh=self.stamp!=env.common_step_counter
        m=measurements(env)
        self.pose_id[fresh]=env._pose_ids[fresh]
        self.failed[fresh]|=m['failed'][fresh]
        self.fallen[fresh]|=m['fallen'][fresh]
        self.left_seen[fresh]|=m['contacts'][fresh,0]
        self.right_seen[fresh]|=m['contacts'][fresh,1]
        self.both_seen[fresh]|=m['contacts'][fresh].all(1)
        self.moved[fresh]|=env._bridge_distance[fresh]>.08
        self.braked[fresh]|=env._bridge_braked[fresh]
        self.unsafe[fresh]|=env._bridge_unsafe[fresh]
        self.descent[fresh]|=phase(env)[fresh]>.9999
        self.left[fresh]=env._bridge_left[fresh]
        valid=m['valid']&~self.failed
        self.hold[fresh]=torch.where(valid[fresh],self.hold[fresh]+1,0)
        self.peak_hold[fresh]=torch.maximum(self.peak_hold[fresh],self.hold[fresh])
        self.success[fresh]|=self.hold[fresh]>=math.ceil(HOLD/env.step_dt-1e-9)
        self.stamp[fresh]=env.common_step_counter
        return self.success&~self.failed


def failure(env): return measurements(env)['failed']


def make_cfg(num_envs=12):
    cfg=pose_cfg(num_envs)
    cfg.events={'pose_reset':EventTermCfg(func=reset_robot,mode='reset'),
        'ball_reset':EventTermCfg(func=reset_ball,mode='reset')}
    cfg.episode_length_s=22.
    cfg.terminations['pose_success']=TerminationTermCfg(func=BridgeOutcome)
    cfg.terminations['physical_failure']=TerminationTermCfg(func=failure)
    return cfg


def frozen_model(obs,checkpoint,cfg,device):
    args={k:cfg[k] for k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
    model=MLPModel(obs,{'actor':['actor']},'actor',22,**args).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['actor_state_dict'])
    return model.eval().requires_grad_(False)


class BridgeEnv(EntryEnv):
    def __init__(self,env):
        self.bridge_ready=False
        super().__init__(env)
        obs=self._base_observations()
        self.entry_teacher=frozen_model(obs,ENTRY_MODEL,asdict(entry_config())['actor'],env.device)
        gait_cfg=json.loads((GAIT_MODEL.parent/'setup.json').read_text())['algorithm']['actor']
        self.gait_teacher=frozen_model(TensorDict({'actor':obs['actor'][:,:75]},batch_size=[self.num_envs]),GAIT_MODEL,gait_cfg,env.device)
        self.previous_bridge=torch.zeros((self.num_envs,24),device=env.device)
        self.bridge_filter=torch.zeros((self.num_envs,22),device=env.device)
        self.teacher_entry_filter=torch.zeros_like(self.bridge_filter)
        self.bridge_amplitude=torch.tensor([.06 if 'Shoulder' in n or 'Elbow' in n else .04 for n in self.action_names],device=env.device)
        self.num_actions=24
        self.bridge_ready=True

    def phase(self): return phase(self.env)

    def _base_observations(self):
        base=CollectionEnv.get_observations(self)
        e=self.env; b=book(e); m=geometry(e); d=e.scene['robot'].data
        ids=e._pose_ids; ref=table(e).at(phase(e))
        extra=torch.cat((torch.nn.functional.one_hot(ids,len(b['names'])).float(),
            m['com'][:,:2]-m['feet'][:,:,:2].mean(1),m['margin'][:,None],m['loads'].clamp(0,2),
            d.root_link_lin_vel_w,m['foot_velocity'],m['support'].float()[:,None],
            ref[:,3:25][:,table(e).knees].mean(1)[:,None],ref[:,2,None],ref[:,1,None]),1)
        actor=torch.cat((base['actor'],extra),1)
        return TensorDict({'actor':actor,'critic':torch.cat((actor,base['critic'][:,-6:]),1)},batch_size=[self.num_envs])

    def get_observations(self):
        base=self._base_observations()
        if not self.bridge_ready: return base
        e=self.env; d=e.scene['robot'].data
        feet=geometry(e)['feet'].mean(1)-e.scene.env_origins
        # Targets are updated only from the RGB estimator, never ball truth.
        error=e._bridge_target-feet[:,:2]
        extra=torch.cat((e._bridge_cmd,torch.nn.functional.one_hot(e._bridge_stage,3).float(),
            error,e._bridge_quiet[:,None]/.6,d.root_link_lin_vel_w[:,:2],e._bridge_braked.float()[:,None],
            e._bridge_width[:,None]/.32),1)
        actor=torch.cat((base['actor'],extra),1)
        return TensorDict({'actor':actor,'critic':torch.cat((actor,base['critic'][:,-6:]),1)},batch_size=[self.num_envs])

    def reset(self):
        result=super().reset()
        if hasattr(self,'previous_bridge'):
            self.previous_bridge.zero_(); self.bridge_filter.zero_(); self.teacher_entry_filter.zero_()
        return result

    def step(self,actions):
        a=actions.clamp(-1,1); e=self.env
        base=self._base_observations()
        d=e.scene['robot'].data; m=measurements(e)
        feet=m['feet'].mean(1)-e.scene.env_origins
        own=d.root_link_pos_w-e.scene.env_origins
        e._bridge_distance=torch.maximum(e._bridge_distance,(own[:,1]-e._bridge_start_y).abs())
        v=self.vision
        visible=v.usable&(v.age<1.)
        desired_point=v.point.clone()
        desired_point[:,0]+=-.20+float(book(e)['feet'][:,0].mean())
        tracking=(e._bridge_stage<2)&visible
        e._bridge_target=torch.where(tracking[:,None],.75*e._bridge_target+.25*desired_point,e._bridge_target)
        error=e._bridge_target-feet[:,:2]
        elapsed=e.episode_length_buf*e.step_dt
        moving=e._bridge_stage==0
        arrived=moving&(elapsed>.8)&visible&(error[:,1].abs()<.025)&(error[:,0].abs()<.035)&(e._bridge_distance>.08)
        e._bridge_stage[arrived]=1
        reacquire=(e._bridge_stage==1)&(e._bridge_quiet>.15)&visible&(error[:,1].abs()>.055)
        e._bridge_stage[reacquire]=0
        e._bridge_quiet[reacquire]=0
        moving=e._bridge_stage==0
        speed=.24+.04*a[:,22]
        lateral=(2.6*(error[:,1]+.008*a[:,23])-.45*d.root_link_lin_vel_w[:,1]).clamp(-speed,speed)
        # Keep active commands inside the accepted gait's useful range; brake
        # through the command slew limiter when the support center is aligned.
        lateral=torch.sign(lateral)*torch.maximum(lateral.abs(),torch.full_like(lateral,.18))
        lateral=torch.where(moving&visible&(elapsed>.8),lateral,0.)
        forward=(1.2*error[:,0]-.3*d.root_link_lin_vel_w[:,0]).clamp(-.08,.08)
        forward=torch.where((e._bridge_stage<2)&visible,forward,0.)
        q=d.root_link_quat_w
        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        cmd=torch.stack((torch.cos(yaw)*forward+torch.sin(yaw)*lateral,
            -torch.sin(yaw)*forward+torch.cos(yaw)*lateral,(-1.5*yaw).clamp(-.4,.4)),1)
        cmd=torch.where((e._bridge_stage<2)[:,None],cmd,0.)
        e._bridge_cmd+=(cmd-e._bridge_cmd).clamp(-.8*e.step_dt,.8*e.step_dt)
        foot_sep=(m['feet'][:,0,1]-m['feet'][:,1,1]).abs()
        fore_aft=(m['feet'][:,0,0]-m['feet'][:,1,0]).abs()
        foot_ids=e._collect_geometry[1]
        foot_speed=d.body_link_lin_vel_w[:,foot_ids,:2].norm(dim=-1).max(1).values
        quiet=(m['support']&(m['margin']>.012)&(m['foot_tilt']<.15)&(m['tilt']<.15)
            &(d.root_link_lin_vel_w[:,:2].norm(dim=1)<.045)&(d.root_link_ang_vel_w.norm(dim=1)<.30)
            &(e._bridge_cmd[:,:2].norm(dim=1)<.025)&(yaw.abs()<.08)
            &(foot_sep>.17)&(foot_sep<.30)&(fore_aft<.03)&(foot_speed<.04))
        braking=e._bridge_stage==1
        e._bridge_quiet=torch.where(braking&quiet,(e._bridge_quiet+e.step_dt).clamp(max=.6),
            torch.where(e._bridge_stage==2,e._bridge_quiet,0.))
        trigger=braking&(e._bridge_quiet>=.599)&visible&(error[:,1].abs()<.04)&(error[:,0].abs()<.035)
        # Match the reference to the measured landing stance rather than
        # forcing loaded feet back to a narrower stance. Only motor commands
        # are blended; no foot/root state is written at the handoff.
        e._bridge_width[trigger]=foot_sep[trigger]
        e._bridge_anchor[trigger]=m['feet'][trigger]
        e._bridge_braked|=trigger
        e._bridge_stage[trigger]=2
        e._bridge_clock=torch.where(e._bridge_stage==2,e._bridge_clock+e.step_dt,0.)
        e._bridge_unsafe|=(phase(e)>0)&~e._bridge_braked
        s=self.phase(); ref=table(e).at(s)
        reference=ref[:,3:25][:,self.path_order]; ff=ref[:,25:47][:,self.path_order]
        entry_obs=self._base_observations()['actor'].clone()
        gait_obs=base['actor'][:,:75].clone(); gait_obs[:,72:75]=e._bridge_cmd
        with torch.no_grad():
            gaita=self.gait_teacher(TensorDict({'actor':gait_obs},batch_size=[self.num_envs]))
            entrya=self.entry_teacher(TensorDict({'actor':entry_obs},batch_size=[self.num_envs])).clamp(-1,1)
            endpoint_obs=entry_obs.clone(); b=book(e); ids=e._pose_ids
            endpoint_obs[:,83]=1.
            endpoint_obs[:,-3:]=torch.stack((b['knee'][ids],b['pitch'][ids],b['z'][ids]),1)
            enda=self.endpoint(TensorDict({'actor':endpoint_obs},batch_size=[self.num_envs])).clamp(-1,1)
        self.teacher_filtered=.8*self.teacher_filtered+.2*enda
        self.teacher_entry_filter=.8*self.teacher_entry_filter+.2*entrya
        self.bridge_filter=.8*self.bridge_filter+.2*a[:,:22]
        collection=reference+ff+s[:,None]**10*self.amplitude*self.teacher_filtered+self.amplitude*self.teacher_entry_filter
        gait=self.home+self.scale*gaita
        blend=(e._bridge_clock/HANDOFF_SETTLE).clamp(0,1)
        blend=blend*blend*(3-2*blend)
        target=(1-blend[:,None])*gait+blend[:,None]*collection+self.bridge_amplitude*self.bridge_filter
        _,_,done,extras=RslRlVecEnvWrapper.step(self,(target.clamp(self.low,self.high)-self.home)/self.scale)
        done=done.bool(); m=measurements(e); out=e._pose_outcome
        feet2=m['feet'].mean(1)-e.scene.env_origins
        # Perfect ball state supervises only reward and evaluation.
        desired=m['ball_pos'][:,:2].clone(); desired[:,0]+=-.20+float(book(e)['feet'][:,0].mean())
        alignment=torch.exp(-((desired-feet2[:,:2])/.05).square().sum(1))
        angles=e.scene['robot'].data.joint_pos[:,self.term.target_ids]
        track=torch.exp(-((angles-m['reference'][:,3:25][:,self.path_order])/.20).square().mean(1))
        collecting=e._bridge_stage==2; late=phase(e)>.9
        reach=torch.exp(-(m['distances']-.13).abs().sum(1)/.1)*late
        hand=m['contacts'].all(1)
        stopped_data=e.scene['robot'].data
        stance=(m['feet'][:,0,:2]-m['feet'][:,1,:2])
        brake_quality=torch.exp(-(stopped_data.root_link_lin_vel_w[:,:2].norm(dim=1)/.06).square()
            -(stopped_data.root_link_ang_vel_w.norm(dim=1)/.4).square()
            -(((stance[:,1].abs()-.235).abs()-.065).clamp(min=0)/.04).square()
            -(stance[:,0]/.04).square())*m['support'].float()
        reward=e.step_dt*(-.4+1.5*alignment+.3*m['support'].float()+.4*collecting*track
            +.8*(e._bridge_stage==1)*brake_quality+.5*collecting.float()
            +.4*collecting*(m['margin']/.04).clamp(0,1)+.5*reach+late*m['contacts'].float().sum(1)
            +2*late*hand+4*m['valid'].float()-2*(m['speed']/.15).clamp(max=3)
            -3*((m['ball_pos'][:,2]-.14)/.05).clamp(0,3))*(~done)
        reward-=.008*(a-self.previous_bridge).square().mean(1)+.001*a.square().mean(1)
        reward+=4*trigger.float()*(~done)
        reward+=50*(done&out.last['success']).float()-25*(done&out.last['failed']).float()
        reward-=8*(done&e.termination_manager.get_term('time_out')&~out.last['success']).float()
        self.previous_bridge=a.clone()
        self.previous_bridge[done]=0
        self.bridge_filter[done]=self.teacher_entry_filter[done]=self.teacher_filtered[done]=0
        ids=done.nonzero().flatten()
        if len(ids): self.vision.reset(ids)
        extras['log']={'Bridge/alignment':alignment,'Bridge/braked':e._bridge_braked.float(),
            'Bridge/phase':phase(e),'Bridge/valid_hold':m['valid'].float(),
            'Bridge/success_event':(done&out.last['success']).float()}
        return self.get_observations(),reward,done.long(),extras


def preserved():
    return bundle_integrity()
