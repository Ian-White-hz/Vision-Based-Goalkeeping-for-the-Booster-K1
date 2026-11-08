"""Bounded stationary-ball ground-collection training, separate from preserved v2."""
from bundle_support import BUNDLE_ROOT, model_path, bundle_integrity
import argparse
import copy
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import booster_k1_ppo_runner_cfg
from collection import HERE,PROJECT,V2,V3,CollectionEnv,make_cfg,measurements


def sha(p):
    with p.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def preserved():
    return bundle_integrity()


def status(**updates):
    p=HERE/'status.json'
    s=json.loads(p.read_text()) if p.exists() else {}
    s.update(updates,updated_at=time.time())
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2))
    tmp.replace(p)


def config():
    c=booster_k1_ppo_runner_cfg()
    c.actor.hidden_dims=c.critic.hidden_dims=(128,128)
    c.actor.distribution_cfg.update(init_std=.2,std_range=(.05,.3))
    c.algorithm.symmetry_cfg=None
    c.algorithm.learning_rate=3e-4
    c.algorithm.schedule='fixed'
    c.algorithm.entropy_coef=.001
    c.algorithm.num_learning_epochs=3
    c.algorithm.clip_param=.1
    c.logger='tensorboard'
    c.upload_model=False
    c.save_interval=50
    c.experiment_name='k1_ground_collection_v3'
    c.run_name='stationary_upper_side_contacts'
    return c


@torch.inference_mode()
def evaluate(checkpoint,episodes,path,seed=2026101931):
    raw=ManagerBasedRlEnv(cfg=make_cfg(min(4,episodes)),device='cuda:0')
    env=CollectionEnv(raw)
    try:
        if checkpoint:
            runner=MjlabOnPolicyRunner(env,asdict(config()),None,'cuda:0')
            runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
            actor=runner.alg.actor.eval()
        else:
            actor=lambda obs:torch.zeros((env.num_envs,22),device=env.device)
        raw.seed(seed)
        obs,_=env.reset()
        quotas=torch.full((env.num_envs,),episodes//env.num_envs,device=env.device,dtype=torch.long)
        quotas[:episodes%env.num_envs]+=1
        finished=torch.zeros_like(quotas)
        totals=dict(episodes=0,successes=0,failures=0,left_contacts=0,right_contacts=0,bilateral_contacts=0,timeouts=0)
        min_dist=10.
        max_force=0.
        trace=[]
        steps=0
        while totals['episodes']<episodes:
            active=finished<quotas
            obs,rew,done,_=env.step(actor(obs))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(rew).all()
            done=done.bool()
            m=measurements(raw)
            if steps%10==0:
                trace.append(dict(step=steps,root_height=float(m['height'][0]),tilt_deg=float(m['tilt'][0])*180/3.14159265,
                    com_error=float(m['com_error'][0]),hand_distances=m['distances'][0].tolist(),
                    contact_forces=m['forces'][0].tolist(),ball=m['ball_pos'][0].tolist(),
                    bilateral=bool(m['contacts'][0].all()),valid=bool(m['valid'][0])))
            valid=active&~done
            if valid.any():
                min_dist=min(min_dist,float(m['distances'][valid].max(1).values.min()))
                max_force=max(max_force,float(m['forces'][valid].max()))
            ids=(done&active).nonzero().flatten()
            finished[ids]+=1
            o=raw._collect
            totals['episodes']+=len(ids)
            for key,name in (('successes','success'),('failures','failed'),('left_contacts','left_seen'),
                ('right_contacts','right_seen'),('bilateral_contacts','both_seen')):
                totals[key]+=int(o.last[name][ids].sum())
            totals['timeouts']+=int(raw.termination_manager.get_term('time_out')[ids].sum())
            steps+=1
            if steps%100==0:
                status(evaluation_steps=steps,evaluation_completed=totals['episodes'],evaluation_target=episodes)
        report=dict(**totals,success_rate=totals['successes']/episodes,failure_rate=totals['failures']/episodes,
            min_max_hand_distance_m=min_dist,max_hand_force_n=max_force,seed=seed,
            checkpoint=str(checkpoint) if checkpoint else 'IK reference,zero learned residual',
            scope='curriculum1: stationary ball,reset in low posture,free physical residual control; no standing transition or incoming-ball claim')
        path.write_text(json.dumps(report,indent=2))
        path.with_suffix('.trace.json').write_text(json.dumps(trace,indent=2))
        print('COLLECTION_EVALUATION',json.dumps(report),flush=True)
        return report
    finally:
        env.close()


def check():
    raw=ManagerBasedRlEnv(cfg=make_cfg(2),device='cuda:0')
    env=CollectionEnv(raw)
    try:
        obs,_=env.reset()
        assert obs['actor'].shape==(2,92) and obs['critic'].shape==(2,98)
        assert env.action_names==raw.action_manager.get_term('joint_pos').target_names
        for _ in range(30):
            obs,rew,_,_=env.step(torch.zeros((2,22),device=env.device))
            assert torch.isfinite(rew).all() and torch.isfinite(obs['actor']).all()
        log=HERE/'checks'/time.strftime('%Y-%m-%d_%H-%M-%S')
        log.mkdir(parents=True,exist_ok=False)
        runner=MjlabOnPolicyRunner(env,asdict(config()),str(log),'cuda:0')
        runner.learn(2,init_at_random_ep_len=False)
        print('CHECK PASS:22 residual actions,92 RGB/proprio/contact actor inputs,hand sensors,PPO updates',flush=True)
    finally:
        env.close()


def train(args):
    output=HERE/'runs'/time.strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True,exist_ok=False)
    status(state='initializing',phase='校验保留的v1/v2及旧来球模型',pid=os.getpid(),iteration=0,
        target=args.iterations,checkpoint=None,error=None,output=str(output),evaluation=None,baseline=None)
    before=preserved()
    (output/'preserved_before.json').write_text(json.dumps(before,indent=2))
    status(state='evaluating',phase='静止球：参考姿态物理基线',evaluation_completed=0,evaluation_target=args.episodes)
    baseline=evaluate(None,args.episodes,output/'baseline.json')
    status(baseline=baseline)
    torch.cuda.empty_cache()
    raw=ManagerBasedRlEnv(cfg=make_cfg(args.num_envs),device='cuda:0')
    env=CollectionEnv(raw)
    try:
        c=asdict(config())
        (output/'setup.json').write_text(json.dumps(dict(algorithm=c,iterations=args.iterations,num_envs=args.num_envs,
            episodes=args.episodes,reference_pose=json.loads((HERE/'poses.json').read_text()),
            action_reference='reset in low IK posture;0.15rad bounded learned residual;standing transition is a later curriculum',
            actor='92 proprioception,RGB estimate,memory,hand contacts,phase; no ball truth',
            critic='actor plus true relative ball position/velocity',
            success='both hands with real force,grounded slow ball,supported posture,0.6s continuous after0.8s',
            stage='stationary ball,not full goalkeeping integration'),indent=2,default=str))
        runner=MjlabOnPolicyRunner(env,c,str(output),'cuda:0')
        last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        initial=copy.deepcopy(runner.alg.actor.state_dict())
        raw.seed(2026101932)
        env.reset()
        original=runner.alg.update
        original_save=runner.save
        count=0
        start=time.time()
        def save(path,infos=None):
            original_save(path,infos)
            status(checkpoint=str(Path(path).resolve()),last_saved_iteration=count)
        runner.save=save
        def update():
            nonlocal count
            r=original()
            count+=1
            assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
            status(state='training',phase='第一阶段：从低位姿态开始，练站稳与双手控静止球',iteration=count,
                iteration_seconds=(time.time()-start)/count)
            return r
        runner.alg.update=update
        runner.learn(args.iterations,init_at_random_ep_len=False)
        final=output/'model_final.pt'
        runner.save(str(final))
        delta=sum(float((v-initial[k]).abs().sum()) for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.'))
        assert delta>0
        (output/'training_check.json').write_text(json.dumps(dict(actor_weight_change=delta,iterations=count),indent=2))
    finally:
        env.close()
    del runner,env,raw
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    status(state='evaluating',phase='静止球：保留种子独立验收',evaluation_completed=0,evaluation_steps=0)
    report=evaluate(final,args.episodes,output/'evaluation.json')
    after=preserved()
    assert before==after,'Preserved inputs changed'
    (output/'preserved_after.json').write_text(json.dumps(after,indent=2))
    gate=report['success_rate']>=.5 and report['failure_rate']<=.1
    comparison=dict(stationary_stage_passed=gate,needs_visual_review=True,moving_ball_collection_completed=False,
        approved_for_replacement=False,preserved_inputs_unchanged=True,baseline=baseline,candidate=report)
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    status(state='completed' if gate else 'needs_review',phase='首轮静止球收球训练结束，等待检查',
        evaluation=comparison,pid=None)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('check','probe','train'))
    p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--episodes',type=int,default=16)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.mode=='check': check()
    elif args.mode=='probe': evaluate(None,4,HERE/'physical_probe.json')
    else:
        with (HERE/'training.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                torch.manual_seed(2026101932)
                train(args)
            except BaseException as e:
                status(state='failed',phase='本轮已停止，请检查错误',error=f'{type(e).__name__}: {e}',pid=None)
                raise
