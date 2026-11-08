"""Finite shared-policy training with independent per-pose stationary-ball tests."""
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
from pose_task import HERE,PoseEnv,make_cfg,measurements,book,preserved
from learn import config as old_config


def status(**updates):
    path=HERE/'status.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    state.update(updates,updated_at=time.time())
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2))
    temp.replace(path)


def config():
    c=old_config()
    c.actor.distribution_cfg.update(init_std=.12,std_range=(.04,.22))
    c.experiment_name='k1_v3_stationary_pose_balance'
    c.run_name='knee_hip_pose_comparison_2s_hold'
    return c


@torch.inference_mode()
def evaluate(checkpoint,episodes_per_pose,path,seed):
    source=json.loads((HERE/'candidates.json').read_text())
    candidates=source['training_candidates']
    raw=ManagerBasedRlEnv(cfg=make_cfg(2*len(candidates)),device='cuda:0')
    env=PoseEnv(raw)
    try:
        if checkpoint:
            runner=MjlabOnPolicyRunner(env,asdict(config()),None,'cuda:0')
            runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
            actor=runner.alg.actor.eval()
        else: actor=lambda obs:torch.zeros((env.num_envs,22),device=env.device)
        raw.seed(seed)
        obs,_=env.reset()
        assert episodes_per_pose%2==0
        quota=episodes_per_pose//2
        finished=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        counts={name:dict(episodes=0,successes=0,failures=0,falls=0,bilateral_contacts=0,timeouts=0,
            peak_hold_seconds=[],stable_samples=0,samples=0,min_support_margin_m=1.,max_foot_drift_m=0.) for name in candidates}
        trace=[]
        step=0
        total=episodes_per_pose*len(candidates)
        completed=0
        while completed<total:
            active=finished<quota
            obs,reward,done,_=env.step(actor(obs))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            done=done.bool()
            m=measurements(raw)
            sample=active&~done&(raw.episode_length_buf*raw.step_dt>.6)
            for i,name in enumerate(candidates):
                mask=sample&(raw._pose_ids==i)
                if mask.any():
                    r=counts[name]
                    r['samples']+=int(mask.sum())
                    r['stable_samples']+=int(m['stable'][mask].sum())
                    r['min_support_margin_m']=min(r['min_support_margin_m'],float(m['margin'][mask].min()))
                    r['max_foot_drift_m']=max(r['max_foot_drift_m'],float(m['foot_drift'][mask].max()))
            if step%20==0:
                trace.append(dict(step=step,pose=raw._pose_ids.tolist(),margin=m['margin'].tolist(),
                    height=m['height'].tolist(),lean_degrees=(m['tilt']*180/3.14159265).tolist(),
                    knee_error_degrees=(m['knee_error']*180/3.14159265).tolist(),support=m['support'].tolist(),
                    foot_loads=m['loads'].tolist(),foot_drift=m['foot_drift'].tolist(),
                    hand_forces=m['forces'].tolist(),ball=m['ball_pos'].tolist(),
                    visual_visible=env.vision.visible.tolist(),hold=(raw._pose_outcome.hold*raw.step_dt).tolist()))
            end=(done&active).nonzero().flatten()
            finished[end]+=1
            out=raw._pose_outcome
            for j in end.tolist():
                name=candidates[int(out.last_pose_id[j])]
                r=counts[name]
                r['episodes']+=1
                completed+=1
                for key,flag in (('successes','success'),('failures','failed'),('falls','fallen'),('bilateral_contacts','both_seen')):
                    r[key]+=int(out.last[flag][j])
                r['timeouts']+=int(raw.termination_manager.get_term('time_out')[j])
                r['peak_hold_seconds'].append(float(out.last_peak_hold[j])*raw.step_dt)
            step+=1
            if step%100==0: status(evaluation_completed=completed,evaluation_target=total,evaluation_steps=step)
        for name,r in counts.items():
            r.update(success_rate=r['successes']/r['episodes'],failure_rate=r['failures']/r['episodes'],
                stable_fraction=r['stable_samples']/max(r['samples'],1),
                nominal_knee_degrees=source['poses'][name]['knee_degrees'],
                nominal_lean_degrees=source['poses'][name]['lean_degrees'])
        report=dict(per_pose=counts,episodes=completed,seed=seed,
            checkpoint=str(checkpoint) if checkpoint else 'IK+gravity-feedforward reference;zero learned correction',
            scope='reset in candidate pose;stationary grounded ball;2s real bilateral hold;no standing-entry/lateral/rolling claim')
        path.write_text(json.dumps(report,indent=2))
        path.with_suffix('.trace.json').write_text(json.dumps(trace,indent=2))
        status(evaluation_completed=completed,evaluation_target=total,evaluation_steps=step)
        print('POSE_EVALUATION',json.dumps(report),flush=True)
        return report
    finally: env.close()


def check():
    frozen=preserved()
    raw=ManagerBasedRlEnv(cfg=make_cfg(3),device='cuda:0')
    env=PoseEnv(raw)
    try:
        obs,_=env.reset()
        expected=106+len(book(raw)['names'])
        assert obs['actor'].shape==(3,expected) and obs['critic'].shape==(3,expected+6)
        assert env.num_actions==22
        for _ in range(40):
            obs,reward,_,_=env.step(torch.zeros((3,22),device=env.device))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        folder=HERE/'checks'/time.strftime('%Y-%m-%d_%H-%M-%S')
        folder.mkdir(parents=True,exist_ok=False)
        runner=MjlabOnPolicyRunner(env,asdict(config()),str(folder),'cuda:0')
        original=copy.deepcopy(runner.alg.actor.state_dict())
        runner.learn(2,init_at_random_ep_len=False)
        assert any(not torch.equal(v,original[k]) for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.'))
        assert frozen==preserved()
        m=measurements(raw)
        report=dict(actor_features=expected,critic_features=expected+6,ppo_updates=2,
            old_versions_unchanged=True,support_force_shape=list(raw.scene['feet_ground_contact'].data.force.shape),
            root_tilt_degrees=(m['tilt']*180/3.14159265).tolist(),support_margin=m['margin'].tolist())
        (folder/'check.json').write_text(json.dumps(report,indent=2))
        print('POSE_CHECK_PASS',json.dumps(report),flush=True)
    finally: env.close()


def train(args):
    output=HERE/'runs'/time.strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True,exist_ok=False)
    source=json.loads((HERE/'candidates.json').read_text())
    status(state='initializing',phase='校验旧版本与候选姿态',pid=os.getpid(),iteration=0,target=args.iterations,
        last_saved_iteration=0,checkpoint=None,output=str(output),error=None,baseline=None,evaluation=None,
        candidate_names=source['training_candidates'],evaluation_completed=0,evaluation_target=4*len(source['training_candidates']))
    before=preserved()
    (output/'preserved_before.json').write_text(json.dumps(before,indent=2))
    status(state='evaluating',phase='训练前：逐姿态固定球基线')
    baseline=evaluate(None,4,output/'baseline.json',seed=2026101909)
    status(baseline=baseline)
    gc.collect(); torch.cuda.empty_cache()
    raw=ManagerBasedRlEnv(cfg=make_cfg(args.num_envs),device='cuda:0')
    env=PoseEnv(raw)
    try:
        c=asdict(config())
        (output/'setup.json').write_text(json.dumps(dict(algorithm=c,iterations=args.iterations,num_envs=args.num_envs,
            candidates=source,actor='RGB estimates + robot kinematics,COM margin,foot loads and pose descriptor;no ball truth',
            critic='actor+6 ball truth values',actions='22 bounded smooth joint corrections to IK+static-load targets',
            success='real bilateral forces,ball stays on ground,low speed,feet supported,COM in sole support,pose profile preserved,2s continuous',
            starts='reset directly into each candidate reference with small joint errors;not a standing transition',
            zero_actuator_delay_inherited=True,standing_entry_completed=False,moving_ball_completed=False),indent=2,default=str))
        runner=MjlabOnPolicyRunner(env,c,str(output),'cuda:0')
        last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
        original=copy.deepcopy(runner.alg.actor.state_dict())
        raw.seed(2026101913)
        env.reset()
        update_original=runner.alg.update
        save_original=runner.save
        count=0; start=time.time()
        def save(path,infos=None):
            save_original(path,infos)
            status(checkpoint=str(Path(path).resolve()),last_saved_iteration=count)
        runner.save=save
        def update():
            nonlocal count
            result=update_original(); count+=1
            assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
            status(state='training',phase='比较三档屈膝/髋部前俯：站稳并双手控静止球2秒',iteration=count,
                iteration_seconds=(time.time()-start)/count)
            return result
        runner.alg.update=update
        runner.learn(args.iterations,init_at_random_ep_len=False)
        final=output/'model_final.pt'
        runner.save(str(final))
        change=sum(float((v-original[k]).abs().sum()) for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.'))
        assert change>0 and count==args.iterations
        (output/'training_check.json').write_text(json.dumps(dict(actor_weight_change=change,updates=count),indent=2))
    finally: env.close()
    del raw,env,runner
    gc.collect(); torch.cuda.empty_cache()
    status(state='evaluating',phase='独立验收：每种姿态16回合',evaluation_completed=0,evaluation_target=16*len(source['training_candidates']))
    report=evaluate(final,16,output/'evaluation.json',seed=2026101917)
    after=preserved()
    assert before==after
    (output/'preserved_after.json').write_text(json.dumps(after,indent=2))
    passed=[name for name,r in report['per_pose'].items() if r['success_rate']>=.75 and r['failures']<=1 and r['falls']<=1 and r['stable_fraction']>=.9]
    selected=max(passed,key=lambda n:(report['per_pose'][n]['success_rate'],report['per_pose'][n]['min_support_margin_m'],
        -report['per_pose'][n]['max_foot_drift_m']),default=None)
    comparison=dict(candidate_pose_stage_passed=bool(selected),selected_pose=selected,passed_poses=passed,
        old_versions_unchanged=True,needs_visual_review=True,standing_entry_completed=False,
        lateral_integration_completed=False,incoming_ball_completed=False,baseline=baseline,candidate=report)
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    status(state='completed' if selected else 'needs_review',phase='本轮候选姿态训练结束，等待动作检查',evaluation=comparison,pid=None)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('check','probe','train'))
    p.add_argument('--iterations',type=int,default=600)
    p.add_argument('--num-envs',type=int,default=15)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.mode=='check': check()
    elif args.mode=='probe': evaluate(None,4,HERE/'probe.json',seed=2026101909)
    else:
        with (HERE/'training.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                torch.manual_seed(2026101913); train(args)
            except BaseException as error:
                status(state='failed',phase='本轮停止，请检查错误',error=f'{type(error).__name__}: {error}',pid=None)
                raise
