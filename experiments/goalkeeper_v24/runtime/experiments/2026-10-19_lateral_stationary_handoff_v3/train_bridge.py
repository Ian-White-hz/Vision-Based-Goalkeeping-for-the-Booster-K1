"""Finite full-flow PPO; balanced held-out left/right physical validation."""
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
from bridge import HERE,ENTRY_MODEL,GAIT_MODEL,BridgeEnv,make_cfg,measurements,phase,preserved,entry_config,EntryEnv


def status(**updates):
    path=HERE/'status.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    state.update(updates,updated_at=time.time())
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)); temp.replace(path)


def config():
    c=entry_config()
    c.actor.distribution_cfg.update(init_std=.10,std_range=(.035,.18))
    c.experiment_name='k1_v3_lateral_stationary_handoff'
    c.run_name='rgb_lateral_measured_stop_stationary_two_hand_control'
    return c


@torch.inference_mode()
def evaluate(checkpoint,episodes,path,seed):
    count=min(4,episodes); assert episodes%count==0
    raw=ManagerBasedRlEnv(cfg=make_cfg(count),device='cuda:0')
    raw._balanced_eval=True
    env=BridgeEnv(raw)
    try:
        if checkpoint:
            runner=MjlabOnPolicyRunner(env,asdict(config()),None,'cuda:0')
            runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
            actor=runner.alg.actor.eval()
        else: actor=lambda obs:torch.zeros((count,24),device=env.device)
        raw.seed(seed); obs,_=env.reset()
        m=measurements(raw)
        assert (phase(raw)==0).all() and (raw._bridge_stage==0).all() and (m['tilt']<.02).all() and (m['height']>.48).all()
        finished=torch.zeros(count,device=env.device,dtype=torch.long); quota=episodes//count
        totals=dict(episodes=0,successes=0,failures=0,falls=0,timeouts=0,bilateral_contacts=0,
            moved=0,braked=0,unsafe_handoff=0,completed_descent=0,full_standing_starts=episodes,
            hold_seconds=[],endpoint_samples=0,stable_samples=0,min_collect_support_margin_m=1.,max_collect_foot_drift_m=0.)
        sides={side:dict(episodes=0,successes=0,falls=0,moved=0,braked=0,completed_descent=0) for side in ('left','right')}
        trace=[]; steps=0
        while int(finished.sum())<episodes:
            active=finished<quota
            obs,reward,done,_=env.step(actor(obs))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            done=done.bool(); m=measurements(raw); out=raw._pose_outcome
            sample=active&~done&(raw._bridge_stage==2)
            late=sample&(phase(raw)>.9999)
            totals['endpoint_samples']+=int(late.sum()); totals['stable_samples']+=int((late&m['stable']).sum())
            if sample.any():
                totals['min_collect_support_margin_m']=min(totals['min_collect_support_margin_m'],float(m['margin'][sample].min()))
                totals['max_collect_foot_drift_m']=max(totals['max_collect_foot_drift_m'],float(m['foot_drift'][sample].max()))
            for j in (done&active).nonzero().flatten().tolist():
                finished[j]+=1; totals['episodes']+=1
                side=sides['left' if out.last['left'][j] else 'right']; side['episodes']+=1
                for key,flag in (('successes','success'),('failures','failed'),('falls','fallen'),
                    ('bilateral_contacts','both_seen'),('moved','moved'),('braked','braked'),
                    ('unsafe_handoff','unsafe'),('completed_descent','descent')):
                    value=int(out.last[flag][j]); totals[key]+=value
                    if key in side: side[key]+=value
                totals['timeouts']+=int(raw.termination_manager.get_term('time_out')[j])
                totals['hold_seconds'].append(float(out.last_peak_hold[j])*raw.step_dt)
            if steps%40==0:
                trace.append(dict(step=steps,stage=raw._bridge_stage.tolist(),phase=phase(raw).tolist(),
                    command=raw._bridge_cmd.tolist(),rgb_target=raw._bridge_target.tolist(),
                    rgb_usable=env.vision.usable.tolist(),quiet=raw._bridge_quiet.tolist(),
                    recenter=raw._bridge_recenter.tolist(),
                    root=(raw.scene['robot'].data.root_link_pos_w-raw.scene.env_origins).tolist(),
                    foot_positions=(m['feet']-raw.scene.env_origins[:,None]).tolist(),
                    margin=m['margin'].tolist(),height=m['height'].tolist(),
                    tilt_degrees=(m['tilt']*180/3.14159265).tolist(),support=m['support'].tolist(),
                    hand_forces=m['forces'].tolist(),ball=m['ball_pos'].tolist(),ball_speed=m['speed'].tolist(),
                    hold_seconds=(out.hold*raw.step_dt).tolist()))
            steps+=1
            if steps%100==0: status(evaluation_completed=totals['episodes'],evaluation_target=episodes,evaluation_steps=steps)
        for r in sides.values(): r['success_rate']=r['successes']/max(r['episodes'],1)
        totals.update(success_rate=totals['successes']/episodes,fall_rate=totals['falls']/episodes,
            endpoint_stable_fraction=totals['stable_samples']/max(1,totals['endpoint_samples']),per_side=sides,
            checkpoint=str(checkpoint) if checkpoint else 'frozen gait+standing-entry+endpoint skills; zero new bridge residual',
            seed=seed,steps=steps,scope='normal standing; RGB approach to static ball left/right; measured braking; real bilateral grounded hold 2s; no rolling shots')
        path.write_text(json.dumps(totals,indent=2)); path.with_suffix('.trace.json').write_text(json.dumps(trace,indent=2))
        status(evaluation_completed=totals['episodes'],evaluation_target=episodes,evaluation_steps=steps)
        print('BRIDGE_EVALUATION',json.dumps(totals),flush=True)
        return totals
    finally: env.close()


def check():
    before=preserved()
    raw=ManagerBasedRlEnv(cfg=make_cfg(4),device='cuda:0'); env=BridgeEnv(raw)
    try:
        obs,_=env.reset()
        assert obs['actor'].shape==(4,122) and obs['critic'].shape==(4,128) and env.num_actions==24
        assert (phase(raw)==0).all() and (measurements(raw)['tilt']<.02).all()
        legacy=EntryEnv.get_observations(env)
        current=env._base_observations()
        assert torch.allclose(current['actor'][:,:106],legacy['actor'][:,:106],atol=2e-5), 'Frozen entry input convention changed'
        folder=HERE/'checks'/time.strftime('%Y-%m-%d_%H-%M-%S'); folder.mkdir(parents=True,exist_ok=False)
        runner=MjlabOnPolicyRunner(env,asdict(config()),str(folder),'cuda:0')
        original=copy.deepcopy(runner.alg.actor.state_dict())
        teachers=[(model,copy.deepcopy(model.state_dict())) for model in (env.endpoint,env.entry_teacher,env.gait_teacher)]
        runner.learn(2,init_at_random_ep_len=False)
        assert any(not torch.equal(v,original[k]) for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.'))
        assert all(torch.equal(v,state[k]) for model,state in teachers for k,v in model.state_dict().items())
        assert before==preserved()
        result=dict(actor_features=122,critic_features=128,actions=24,updates=2,
            all_standing_starts=True,frozen_entry_inputs_compatible=True,actor_weights_changed=True,
            all_frozen_teachers_unchanged=True,prior_versions_unchanged=True)
        (folder/'check.json').write_text(json.dumps(result,indent=2)); print('BRIDGE_CHECK_PASS',json.dumps(result),flush=True)
    finally: env.close()


def train(args):
    output=HERE/'runs'/time.strftime('%Y-%m-%d_%H-%M-%S'); output.mkdir(parents=True,exist_ok=False)
    status(state='initializing',phase='验证旧版本与完整左右衔接基线',pid=os.getpid(),iteration=0,target=args.iterations,
        last_saved_iteration=0,checkpoint=None,output=str(output),baseline=None,evaluation=None,error=None,
        evaluation_completed=0,evaluation_target=32,evaluation_steps=0)
    before=preserved(); (output/'preserved_before.json').write_text(json.dumps(before,indent=2))
    status(state='evaluating',phase='启动基线：左右各2回合，检查更慢的实测停稳过渡',evaluation_target=4)
    baseline=evaluate(None,4,output/'baseline.json',2026101937)
    assert baseline['episodes']==4 and baseline['full_standing_starts']==4
    assert all(r['episodes']==2 for r in baseline['per_side'].values())
    status(baseline=baseline)
    gc.collect(); torch.cuda.empty_cache()
    raw=ManagerBasedRlEnv(cfg=make_cfg(args.num_envs),device='cuda:0'); env=BridgeEnv(raw)
    try:
        c=asdict(config())
        setup=dict(algorithm=c,iterations=args.iterations,num_envs=args.num_envs,
            starts='normal standing; stationary ground ball x=.20+/-.01m,y=left/right .16-.22m; no partial starts',
            actor='122 RGB/proprioception/contact/phase/stop-state inputs; no ball truth',critic='actor+6 ball truth',
            actions='24 bounded outputs:22 smooth motor residuals, lateral speed adjustment,small RGB target correction',
            frozen_models=[str(ENTRY_MODEL),str(GAIT_MODEL)],
            handoff='continuous0.6s actual low body/foot speeds,double support,COM margin,parallel stance and RGB alignment; reference width matched to measured stance',
            handoff_duration_seconds=2.5,descent_duration_seconds=6.,
            success='actual lateral move>8cm,measured braking,complete descent,real bilateral grounded stable control2s',
            inherited_zero_actuator_delay=True,incoming_ball_completed=False)
        (output/'setup.json').write_text(json.dumps(setup,indent=2,default=str))
        runner=MjlabOnPolicyRunner(env,c,str(output),'cuda:0')
        last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
        original=copy.deepcopy(runner.alg.actor.state_dict())
        raw.seed(2026101939); env.reset()
        update_original=runner.alg.update; save_original=runner.save; count=0; start=time.time()
        def save(path,infos=None):
            save_original(path,infos); status(checkpoint=str(Path(path).resolve()),last_saved_iteration=count)
        runner.save=save
        def update():
            nonlocal count
            result=update_original(); count+=1
            assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
            status(state='training',phase='视觉侧移对位→收步停稳→俯身稳住固定球2秒',iteration=count,
                iteration_seconds=(time.time()-start)/count)
            return result
        runner.alg.update=update; runner.learn(args.iterations,init_at_random_ep_len=False)
        final=output/'model_final.pt'; runner.save(str(final))
        change=sum(float((v-original[k]).abs().sum()) for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.'))
        assert change>0 and count==args.iterations
        (output/'training_check.json').write_text(json.dumps(dict(updates=count,actor_weight_change=change),indent=2))
    finally: env.close()
    del raw,env,runner; gc.collect(); torch.cuda.empty_cache()
    status(state='evaluating',phase='独立验收：左右各16回合完整侧移停稳收球',evaluation_completed=0,evaluation_target=32)
    report=evaluate(final,32,output/'evaluation.json',2026101941)
    after=preserved(); assert before==after; (output/'preserved_after.json').write_text(json.dumps(after,indent=2))
    passed=(report['success_rate']>=.8 and report['falls']<=1 and report['unsafe_handoff']==0
        and report['endpoint_stable_fraction']>=.9 and all(r['success_rate']>=.75 for r in report['per_side'].values()))
    result=dict(lateral_stationary_handoff_passed=passed,baseline=baseline,candidate=report,
        prior_versions_unchanged=True,needs_visual_review=True,incoming_ball_completed=False)
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    status(state='completed' if passed else 'needs_review',phase='本轮结束，等待完整衔接动作检查',pid=None,evaluation=result)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('check','probe','train'))
    p.add_argument('--iterations',type=int,default=1000); p.add_argument('--num-envs',type=int,default=12)
    args=p.parse_args(); torch.set_num_threads(4)
    if args.mode=='check': check()
    elif args.mode=='probe': evaluate(None,4,HERE/'probe.json',2026101937)
    else:
        with (HERE/'training.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                torch.manual_seed(2026101939); train(args)
            except BaseException as error:
                status(state='failed',phase='本轮停止，请检查错误',pid=None,error=f'{type(error).__name__}: {error}')
                raise
