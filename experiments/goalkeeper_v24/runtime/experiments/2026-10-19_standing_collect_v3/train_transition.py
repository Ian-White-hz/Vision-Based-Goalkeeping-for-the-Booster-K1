"""Bounded PPO transition training. Existing gait and low collection stay frozen."""
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
from transition import HERE,TransitionEnv,make_cfg,measurements,phase,LOW_MODEL,GAIT_MODEL,preserved,low_config


def status(**updates):
    p=HERE/'status.json'
    s=json.loads(p.read_text()) if p.exists() else {}
    s.update(updates,updated_at=time.time())
    temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(s,ensure_ascii=False,indent=2))
    temporary.replace(p)


def config():
    c=low_config()
    c.actor.distribution_cfg.update(init_std=.15,std_range=(.04,.25))
    c.experiment_name='k1_v3_standing_to_collection'
    c.run_name='standing_transition_residual'
    return c


@torch.inference_mode()
def evaluate(checkpoint,episodes,path):
    raw=ManagerBasedRlEnv(cfg=make_cfg(min(4,episodes)),device='cuda:0')
    env=TransitionEnv(raw)
    try:
        if checkpoint:
            runner=MjlabOnPolicyRunner(env,asdict(config()),None,'cuda:0')
            runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
            actor=runner.alg.actor.eval()
        else:
            actor=lambda obs:torch.zeros((env.num_envs,22),device=env.device)
        raw._allow_partial_starts=False
        raw.seed(2026101947)
        obs,_=env.reset()
        m=measurements(raw)
        assert (m['height']>.48).all() and (m['tilt']<.1).all()
        assert (phase(raw)==0).all()
        quotas=torch.full((env.num_envs,),episodes//env.num_envs,device=env.device,dtype=torch.long)
        quotas[:episodes%env.num_envs]+=1
        completed=torch.zeros_like(quotas)
        counts=dict(episodes=0,successes=0,failures=0,full_standing_starts=0,bilateral_contacts=0,timeouts=0)
        trace=[]
        max_foot_drift=0.
        max_joint_speed=0.
        step=0
        while counts['episodes']<episodes:
            active=completed<quotas
            obs,reward,done,_=env.step(actor(obs))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            done=done.bool()
            m=measurements(raw)
            valid=active&~done
            if valid.any():
                local=m['feet']-raw.scene.env_origins[:,None]
                drift=(local[:,:,:2]-env.path.feet[None,:,:2]).norm(dim=2)
                max_foot_drift=max(max_foot_drift,float(drift[valid].max()))
                max_joint_speed=max(max_joint_speed,float(raw.scene['robot'].data.joint_vel[valid].abs().max()))
            if step%10==0:
                trace.append(dict(step=step,phase=float(phase(raw)[0]),root_height=float(m['height'][0]),
                    tilt_deg=float(m['tilt'][0])*180/3.14159265,com_error=float(m['com_error'][0]),
                    hands=m['distances'][0].tolist(),forces=m['forces'][0].tolist(),
                    ball=m['ball_pos'][0].tolist(),valid=bool(m['valid'][0])))
            ids=(done&active).nonzero().flatten()
            completed[ids]+=1
            counts['episodes']+=len(ids)
            last=raw._collect.last
            for key,name in (('successes','success'),('failures','failed'),('full_standing_starts','full_start'),('bilateral_contacts','both_seen')):
                counts[key]+=int(last[name][ids].sum())
            counts['timeouts']+=int(raw.termination_manager.get_term('time_out')[ids].sum())
            step+=1
            if step%100==0:
                status(evaluation_completed=counts['episodes'],evaluation_target=episodes,evaluation_steps=step)
        assert counts['full_standing_starts']==episodes
        report=dict(**counts,success_rate=counts['successes']/episodes,failure_rate=counts['failures']/episodes,
            max_foot_drift_m=max_foot_drift,max_joint_speed_rad_s=max_joint_speed,seed=2026101947,
            checkpoint=str(checkpoint) if checkpoint else 'frozen endpoints + balance-aware path; zero new residual',
            scope='all episodes start standing; stationary ball only; no moving shots or lateral integration')
        path.write_text(json.dumps(report,indent=2))
        path.with_suffix('.trace.json').write_text(json.dumps(trace,indent=2))
        status(evaluation_completed=episodes,evaluation_target=episodes,evaluation_steps=step)
        print('STANDING_TRANSITION_EVALUATION',json.dumps(report),flush=True)
        return report
    finally: env.close()


def check():
    raw=ManagerBasedRlEnv(cfg=make_cfg(2),device='cuda:0')
    env=TransitionEnv(raw)
    try:
        obs,_=env.reset()
        assert obs['actor'].shape==(2,92) and obs['critic'].shape==(2,98)
        assert (measurements(raw)['height']>.48).all() and (phase(raw)==0).all()
        assert all(not p.requires_grad for teacher in (env.gait_teacher,env.low_teacher) for p in teacher.parameters())
        teacher_before=[p.detach().clone() for t in (env.gait_teacher,env.low_teacher) for p in t.parameters()]
        for _ in range(30):
            obs,reward,done,_=env.step(torch.zeros((2,22),device=env.device))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        folder=HERE/'checks'/time.strftime('%Y-%m-%d_%H-%M-%S')
        folder.mkdir(parents=True,exist_ok=False)
        runner=MjlabOnPolicyRunner(env,asdict(config()),str(folder),'cuda:0')
        runner.learn(2,init_at_random_ep_len=False)
        teacher_after=[p for t in (env.gait_teacher,env.low_teacher) for p in t.parameters()]
        assert all(torch.equal(a,b) for a,b in zip(teacher_before,teacher_after))
        print('CHECK_PASS:full standing reset,92/98 observations,2 PPO updates,frozen teachers unchanged',flush=True)
    finally: env.close()


def train(args):
    output=HERE/'runs'/time.strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True,exist_ok=False)
    status(state='initializing',phase='校验保留的v2和第一阶段模型',pid=os.getpid(),iteration=0,
        target=args.iterations,checkpoint=None,last_saved_iteration=0,error=None,output=str(output),
        baseline=None,evaluation=None,evaluation_completed=0,evaluation_target=args.episodes)
    before=preserved()
    (output/'preserved_before.json').write_text(json.dumps(before,indent=2))
    status(state='evaluating',phase='从站立开始的训练前基线')
    baseline=evaluate(None,args.episodes,output/'baseline.json')
    status(baseline=baseline)
    gc.collect()
    torch.cuda.empty_cache()
    raw=ManagerBasedRlEnv(cfg=make_cfg(args.num_envs),device='cuda:0')
    env=TransitionEnv(raw)
    try:
        c=asdict(config())
        (output/'setup.json').write_text(json.dumps(dict(algorithm=c,iterations=args.iterations,num_envs=args.num_envs,
            held_out_episodes=args.episodes,gait_teacher=str(GAIT_MODEL),low_teacher=str(LOW_MODEL),
            teachers_frozen=True,actor_features='92 RGB/proprioception/hand contacts/transition phase; no ball truth',
            critic_features='actor plus relative ball truth and velocity',
            reference='41-frame balance-aware IK,static PD feedforward,frozen standing/collection endpoint controls',
            policy='new 22-joint bounded 0.15rad residual,trained with PPO',
            partial_starts='25% partial crouch starts during first 2/3 updates;remaining starts standing;last1/3 all standing',
            evaluation='all reset standing;full4s descent before0.6s supported bilateral grounded-ball hold',
            scope='stationary ball; no incoming-ball/lateral integration; no root/ball writes except episode reset',
            latency='zero command delay inherited from first low-collection curriculum'),indent=2,default=str))
        runner=MjlabOnPolicyRunner(env,c,str(output),'cuda:0')
        last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        initial=copy.deepcopy(runner.alg.actor.state_dict())
        raw._allow_partial_starts=True
        raw.seed(2026101942)
        env.reset()
        update_original=runner.alg.update
        save_original=runner.save
        count=0
        start=time.time()
        def save(path,infos=None):
            save_original(path,infos)
            status(checkpoint=str(Path(path).resolve()),last_saved_iteration=count)
        runner.save=save
        def update():
            nonlocal count
            result=update_original()
            count+=1
            assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
            raw._allow_partial_starts=count<int(args.iterations*2/3)
            status(state='training',phase='站立→平稳屈膝俯身→双手控静止球',iteration=count,
                iteration_seconds=(time.time()-start)/count,
                partial_start_curriculum=raw._allow_partial_starts)
            return result
        runner.alg.update=update
        runner.learn(args.iterations,init_at_random_ep_len=False)
        final=output/'model_final.pt'
        runner.save(str(final))
        delta=sum(float((v-initial[k]).abs().sum()) for k,v in runner.alg.actor.state_dict().items() if k.startswith('mlp.'))
        assert delta>0
        (output/'training_check.json').write_text(json.dumps(dict(actor_weight_change=delta,updates=count),indent=2))
    finally: env.close()
    del runner,env,raw
    gc.collect()
    torch.cuda.empty_cache()
    status(state='evaluating',phase='独立验收：全部从站立开始',evaluation_completed=0,evaluation_steps=0)
    candidate=evaluate(final,args.episodes,output/'evaluation.json')
    after=preserved()
    assert before==after,'Preserved skill inputs changed'
    (output/'preserved_after.json').write_text(json.dumps(after,indent=2))
    passed=candidate['success_rate']>=.75 and candidate['failure_rate']<=.1
    comparison=dict(standing_transition_passed=passed,needs_visual_review=True,
        preserved_inputs_unchanged=True,moving_ball_collection_completed=False,
        approved_for_replacement=False,baseline=baseline,candidate=candidate)
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    status(state='completed' if passed else 'needs_review',phase='本轮站立到收球训练结束，等待检查',
        evaluation=comparison,pid=None)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('check','train'))
    p.add_argument('--iterations',type=int,default=800)
    p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--episodes',type=int,default=16)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.mode=='check': check()
    else:
        with (HERE/'training.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                torch.manual_seed(2026101942)
                train(args)
            except BaseException as e:
                status(state='failed',phase='本轮已停止，请检查错误',error=f'{type(e).__name__}: {e}',pid=None)
                raise
