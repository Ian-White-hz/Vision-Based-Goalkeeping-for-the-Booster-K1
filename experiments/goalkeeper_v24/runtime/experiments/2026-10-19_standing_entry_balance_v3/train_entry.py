"""Finite full-standing-entry PPO and held-out full-motion evaluation."""
import argparse
import copy
from dataclasses import asdict
import fcntl
import gc
import json
import os
from pathlib import Path
import time
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from entry import HERE,ENDPOINT,EntryEnv,make_cfg,measurements,phase,preserved,pose_config


def status(**updates):
    path=HERE/'status.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    state.update(updates,updated_at=time.time())
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2))
    temp.replace(path)


def config():
    c=pose_config()
    c.actor.distribution_cfg.update(init_std=.10,std_range=(.035,.18))
    c.experiment_name='k1_v3_standing_entry_balance'
    c.run_name='standing_to_shallow_knee_two_hand_ground_control'
    return c


@torch.inference_mode()
def evaluate(checkpoint,episodes,path,seed):
    count=min(4,episodes)
    assert episodes%count==0
    raw=ManagerBasedRlEnv(cfg=make_cfg(count),device='cuda:0')
    env=EntryEnv(raw)
    try:
        if checkpoint:
            runner=MjlabOnPolicyRunner(env,asdict(config()),None,'cuda:0')
            runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
            actor=runner.alg.actor.eval()
        else: actor=lambda obs:torch.zeros((count,22),device=env.device)
        raw.seed(seed)
        obs,_=env.reset()
        m=measurements(raw)
        assert (phase(raw)==0).all() and (m['tilt']<.02).all() and (m['height']>.48).all()
        completed=torch.zeros(count,device=env.device,dtype=torch.long)
        quota=episodes//count
        reached=torch.zeros(count,device=env.device,dtype=torch.bool)
        totals=dict(episodes=0,successes=0,failures=0,falls=0,timeouts=0,bilateral_contacts=0,
            reached_endpoint=0,full_standing_starts=episodes,hold_seconds=[],elapsed_seconds=[],
            stable_samples=0,endpoint_samples=0,min_support_margin_m=1.,max_foot_drift_m=0.)
        trace=[]
        steps=0
        while int(completed.sum())<episodes:
            active=completed<quota
            reached|=phase(raw)>.9999
            obs,reward,done,_=env.step(actor(obs))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            done=done.bool(); m=measurements(raw)
            sample=active&~done&(raw.episode_length_buf*raw.step_dt>.6)
            late=sample&(phase(raw)>.9999)
            totals['endpoint_samples']+=int(late.sum())
            totals['stable_samples']+=int((late&m['stable']).sum())
            if sample.any():
                totals['min_support_margin_m']=min(totals['min_support_margin_m'],float(m['margin'][sample].min()))
                totals['max_foot_drift_m']=max(totals['max_foot_drift_m'],float(m['foot_drift'][sample].max()))
            out=raw._pose_outcome
            for j in (done&active).nonzero().flatten().tolist():
                completed[j]+=1
                totals['episodes']+=1
                for key,flag in (('successes','success'),('failures','failed'),('falls','fallen'),('bilateral_contacts','both_seen')):
                    totals[key]+=int(out.last[flag][j])
                totals['timeouts']+=int(raw.termination_manager.get_term('time_out')[j])
                totals['reached_endpoint']+=int(reached[j])
                totals['hold_seconds'].append(float(out.last_peak_hold[j])*raw.step_dt)
            reached[done]=False
            if steps%40==0:
                trace.append(dict(step=steps,phase=phase(raw).tolist(),height=m['height'].tolist(),
                    tilt_degrees=(m['tilt']*180/3.14159265).tolist(),margin=m['margin'].tolist(),
                    feet_supported=m['support'].tolist(),hand_forces=m['forces'].tolist(),
                    ball=m['ball_pos'].tolist(),ball_speed=m['speed'].tolist(),
                    hold_seconds=(out.hold*raw.step_dt).tolist()))
            steps+=1
            if steps%100==0: status(evaluation_completed=totals['episodes'],evaluation_target=episodes,evaluation_steps=steps)
        totals.update(success_rate=totals['successes']/episodes,fall_rate=totals['falls']/episodes,
            endpoint_stable_fraction=totals['stable_samples']/max(1,totals['endpoint_samples']),
            checkpoint=str(checkpoint) if checkpoint else 'reference path + frozen endpoint teacher; zero new residual',
            seed=seed,scope='all episodes start upright; no partial pose starts; stationary ball; real bilateral grounded stable hold 2s',
            steps=steps)
        path.write_text(json.dumps(totals,indent=2))
        path.with_suffix('.trace.json').write_text(json.dumps(trace,indent=2))
        print('ENTRY_EVALUATION',json.dumps(totals),flush=True)
        return totals
    finally: env.close()


def check():
    before=preserved()
    raw=ManagerBasedRlEnv(cfg=make_cfg(4),device='cuda:0')
    env=EntryEnv(raw)
    try:
        obs,_=env.reset()
        assert obs['actor'].shape==(4,109) and obs['critic'].shape==(4,115)
        assert (phase(raw)==0).all() and (measurements(raw)['tilt']<.02).all()
        folder=HERE/'checks'/time.strftime('%Y-%m-%d_%H-%M-%S')
        folder.mkdir(parents=True,exist_ok=False)
        runner=MjlabOnPolicyRunner(env,asdict(config()),str(folder),'cuda:0')
        original=copy.deepcopy(runner.alg.actor.state_dict())
        teacher=copy.deepcopy(env.endpoint.state_dict())
        runner.learn(2,init_at_random_ep_len=False)
        assert any(not torch.equal(v,original[k]) for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.'))
        assert all(torch.equal(v,teacher[k]) for k,v in env.endpoint.state_dict().items())
        assert before==preserved()
        result=dict(actor_features=109,critic_features=115,updates=2,all_standing_starts=True,
            actor_weights_changed=True,frozen_teacher_unchanged=True,prior_versions_unchanged=True)
        (folder/'check.json').write_text(json.dumps(result,indent=2))
        print('ENTRY_CHECK_PASS',json.dumps(result),flush=True)
    finally: env.close()


def train(args):
    output=HERE/'runs'/time.strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True,exist_ok=False)
    status(state='initializing',phase='验证旧版本并检查完整站立起步基线',pid=os.getpid(),iteration=0,
        target=args.iterations,last_saved_iteration=0,checkpoint=None,output=str(output),
        baseline=None,evaluation=None,error=None,evaluation_completed=0,evaluation_target=32)
    before=preserved()
    (output/'preserved_before.json').write_text(json.dumps(before,indent=2))
    # The full-motion probe was just completed with these sources and the same
    # frozen teacher. Keep that evidence instead of repeating identical trials.
    baseline=json.loads((HERE/'probe.json').read_text())
    assert baseline['episodes']==4 and baseline['full_standing_starts']==4
    assert baseline['checkpoint']=='reference path + frozen endpoint teacher; zero new residual'
    (output/'baseline.json').write_text(json.dumps(baseline,indent=2))
    status(baseline=baseline)
    gc.collect(); torch.cuda.empty_cache()
    raw=ManagerBasedRlEnv(cfg=make_cfg(args.num_envs),device='cuda:0')
    env=EntryEnv(raw)
    try:
        c=asdict(config())
        (output/'setup.json').write_text(json.dumps(dict(algorithm=c,iterations=args.iterations,num_envs=args.num_envs,
            all_starts='normal standing with small joint errors; no partial initialization',
            reference='balance-aware IK path; 0.8s standing settling,6s smooth descent; frozen accepted endpoint teacher',
            actor='109 RGB/proprioception/contact/phase/balance inputs; no ball truth',critic='actor+6 ball truth',
            success='full standing entry,physical bilateral hand contact,grounded slow ball,stable support and profile,2 continuous seconds',
            endpoint_model=str(ENDPOINT),zero_actuator_delay_inherited=True,
            moving_ball_completed=False,lateral_integration_completed=False),indent=2,default=str))
        runner=MjlabOnPolicyRunner(env,c,str(output),'cuda:0')
        last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
        original=copy.deepcopy(runner.alg.actor.state_dict())
        raw.seed(2026101923)
        env.reset()
        update_original=runner.alg.update; save_original=runner.save
        count=0; start=time.time()
        def save(path,infos=None):
            save_original(path,infos)
            status(checkpoint=str(Path(path).resolve()),last_saved_iteration=count)
        runner.save=save
        def update():
            nonlocal count
            result=update_original(); count+=1
            assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
            status(state='training',phase='正常站立→缓慢前俯→双手地面控球2秒',iteration=count,
                iteration_seconds=(time.time()-start)/count)
            return result
        runner.alg.update=update
        runner.learn(args.iterations,init_at_random_ep_len=False)
        final=output/'model_final.pt'
        runner.save(str(final))
        change=sum(float((v-original[k]).abs().sum()) for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.'))
        assert change>0 and count==args.iterations
        (output/'training_check.json').write_text(json.dumps(dict(updates=count,actor_weight_change=change),indent=2))
    finally: env.close()
    del raw,env,runner
    gc.collect(); torch.cuda.empty_cache()
    status(state='evaluating',phase='独立验收：32回合完整站立俯身控球',evaluation_completed=0,evaluation_target=32)
    report=evaluate(final,32,output/'evaluation.json',2026101929)
    after=preserved(); assert before==after
    (output/'preserved_after.json').write_text(json.dumps(after,indent=2))
    passed=report['success_rate']>=.8 and report['falls']<=1 and report['endpoint_stable_fraction']>=.9
    result=dict(standing_entry_passed=passed,baseline=baseline,candidate=report,
        prior_versions_unchanged=True,needs_visual_review=True,lateral_integration_completed=False,incoming_ball_completed=False)
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    status(state='completed' if passed else 'needs_review',phase='本轮结束，等待完整动作检查',pid=None,evaluation=result)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('check','probe','train'))
    p.add_argument('--iterations',type=int,default=1000); p.add_argument('--num-envs',type=int,default=16)
    args=p.parse_args(); torch.set_num_threads(4)
    if args.mode=='check': check()
    elif args.mode=='probe': evaluate(None,4,HERE/'probe.json',2026101921)
    else:
        with (HERE/'training.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                torch.manual_seed(2026101923); train(args)
            except BaseException as error:
                status(state='failed',phase='本轮停止，请检查错误',pid=None,error=f'{type(error).__name__}: {error}')
                raise
