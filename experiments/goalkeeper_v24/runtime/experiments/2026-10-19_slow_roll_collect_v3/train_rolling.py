"""Bounded PPO for vision-timed slow frontal rolling-ball capture and 2s hold."""
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
from rolling import HERE,RollingEnv,make_cfg,measurements,preserved,STAND_MODEL,HOLD_SECONDS
from rolling import standing_config


def status(**updates):
    p=HERE/'status.json'
    s=json.loads(p.read_text()) if p.exists() else {}
    s.update(updates,updated_at=time.time())
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(s,ensure_ascii=False,indent=2))
    temp.replace(p)


def config():
    c=standing_config()
    c.actor.distribution_cfg.update(init_std=.12,std_range=(.04,.22))
    c.experiment_name='k1_v3_slow_rolling_collection'
    c.run_name='rgb_timed_frontal_roll_2s_hold'
    return c


@torch.inference_mode()
def evaluate(checkpoint,episodes,path):
    raw=ManagerBasedRlEnv(cfg=make_cfg(min(4,episodes)),device='cuda:0')
    env=RollingEnv(raw)
    try:
        if checkpoint:
            runner=MjlabOnPolicyRunner(env,asdict(config()),None,'cuda:0')
            runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
            actor=runner.alg.actor.eval()
        else: actor=lambda obs:torch.zeros((env.num_envs,24),device=env.device)
        raw._allow_partial_starts=False
        raw.seed(2026101961)
        obs,_=env.reset()
        ball=raw.scene['ball'].data
        assert (measurements(raw)['height']>.48).all() and (raw._roll_phase==0).all()
        assert (ball.root_link_lin_vel_w[:,0]<-.17).all()
        assert torch.allclose(ball.root_link_ang_vel_w[:,1],ball.root_link_lin_vel_w[:,0]/.11)
        quota=torch.full((env.num_envs,),episodes//env.num_envs,device=env.device,dtype=torch.long)
        quota[:episodes%env.num_envs]+=1
        finished=torch.zeros_like(quota)
        totals=dict(episodes=0,successes=0,failures=0,full_standing_starts=0,motion_seen=0,travelled=0,
            touched=0,bilateral_contacts=0,escaped=0,timeouts=0)
        peaks=[]
        trace=[]
        step=0
        while totals['episodes']<episodes:
            active=finished<quota
            obs,reward,done,_=env.step(actor(obs))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            done=done.bool()
            m=measurements(raw)
            if step%20==0:
                tta,_=env.estimated_tta()
                trace.append(dict(step=step,phase=float(raw._roll_phase[0]),tta=float(tta[0]),
                    triggered=bool(raw._roll_triggered[0]),root_height=float(m['height'][0]),
                    tilt_deg=float(m['tilt'][0])*180/3.14159265,
                    ball=m['ball_pos'][0].tolist(),speed=float(m['speed'][0]),forces=m['forces'][0].tolist(),
                    valid=bool(m['valid'][0]),hold_s=float(raw._rolling.hold[0])*raw.step_dt))
            ids=(done&active).nonzero().flatten()
            finished[ids]+=1
            totals['episodes']+=len(ids)
            o=raw._rolling
            for key,name in (('successes','success'),('failures','failed'),('full_standing_starts','full_start'),
                ('motion_seen','motion_seen'),('travelled','travelled'),('touched','touched'),
                ('bilateral_contacts','both_seen'),('escaped','escaped')):
                totals[key]+=int(o.last[name][ids].sum())
            peaks.extend((o.last_peak_hold[ids]*raw.step_dt).tolist())
            totals['timeouts']+=int(raw.termination_manager.get_term('time_out')[ids].sum())
            step+=1
            if step%100==0:
                status(evaluation_completed=totals['episodes'],evaluation_target=episodes,evaluation_steps=step)
        assert totals['full_standing_starts']==episodes
        report=dict(**totals,success_rate=totals['successes']/episodes,failure_rate=totals['failures']/episodes,
            hold_requirement_s=HOLD_SECONDS,peak_hold_seconds=peaks,mean_peak_hold_s=sum(peaks)/len(peaks),
            seed=2026101961,checkpoint=str(checkpoint) if checkpoint else 'frozen skills + RGB timing;zero new residual',
            scope='full standing start;frontal rolling ball from1.4-1.7m at0.18-0.24m/s;2s bilateral grounded hold;no lateral integration')
        path.write_text(json.dumps(report,indent=2))
        path.with_suffix('.trace.json').write_text(json.dumps(trace,indent=2))
        status(evaluation_completed=episodes,evaluation_target=episodes,evaluation_steps=step)
        print('SLOW_ROLL_EVALUATION',json.dumps(report),flush=True)
        return report
    finally: env.close()


def check():
    raw=ManagerBasedRlEnv(cfg=make_cfg(2),device='cuda:0')
    env=RollingEnv(raw)
    try:
        obs,_=env.reset()
        assert obs['actor'].shape==(2,94) and obs['critic'].shape==(2,100)
        assert env.num_actions==24 and (raw._roll_phase==0).all()
        teachers=(env.gait_teacher,env.low_teacher,env.standing_teacher)
        assert all(not p.requires_grad for t in teachers for p in t.parameters())
        before=[p.detach().clone() for t in teachers for p in t.parameters()]
        for _ in range(30):
            obs,reward,_,_=env.step(torch.zeros((2,24),device=env.device))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        folder=HERE/'checks'/time.strftime('%Y-%m-%d_%H-%M-%S')
        folder.mkdir(parents=True,exist_ok=False)
        runner=MjlabOnPolicyRunner(env,asdict(config()),str(folder),'cuda:0')
        runner.learn(2,init_at_random_ep_len=False)
        after=[p for t in teachers for p in t.parameters()]
        assert all(torch.equal(a,b) for a,b in zip(before,after))
        print('CHECK_PASS:94/100 observations,24 outputs,moving ball,2 PPO updates,all3 old skills frozen',flush=True)
    finally: env.close()


def train(args):
    output=HERE/'runs'/time.strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True,exist_ok=False)
    status(state='initializing',phase='校验保留的旧模型与回放代码',pid=os.getpid(),iteration=0,target=args.iterations,
        last_saved_iteration=0,checkpoint=None,output=str(output),error=None,baseline=None,evaluation=None,
        evaluation_completed=0,evaluation_target=args.baseline_episodes)
    before=preserved()
    (output/'preserved_before.json').write_text(json.dumps(before,indent=2))
    status(state='evaluating',phase='训练前基线：真正滚来的慢球与连续2秒控球')
    baseline=evaluate(None,args.baseline_episodes,output/'baseline.json')
    status(baseline=baseline)
    gc.collect()
    torch.cuda.empty_cache()
    raw=ManagerBasedRlEnv(cfg=make_cfg(args.num_envs),device='cuda:0')
    env=RollingEnv(raw)
    try:
        c=asdict(config())
        (output/'setup.json').write_text(json.dumps(dict(algorithm=c,iterations=args.iterations,num_envs=args.num_envs,
            baseline_episodes=args.baseline_episodes,held_out_episodes=args.episodes,
            shot=dict(start_x=(1.4,1.7),start_y=(-.006,.006),speed=(.18,.24),direction='straight negativeX',
                radius=.11,spin='wy=vx/r,no initial slip',ball_and_ground_parameters='unchanged'),
            motor_primitive=str(STAND_MODEL),all_teachers_frozen=True,
            motor_primitive_context='fixed collection target (.20,0),not a moving-ball measurement',
            actor='94 features:actual RGB estimates,proprioception,contacts,phase,TTA,trigger;no ball truth',
            critic='actor plus relative ball truth/velocity',outputs='22 bounded motor corrections + trigger threshold + descent rate',
            trigger='RGB only;estimated TTA<4.1+0.4a22 or visible ball x<.65;settle0.6s',
            descent='4s reference/(1+0.12a23),smooth monotone clock',
            success='ball was moving and travelled>0.6m,bilateral real forces,grounded low speed,supported,2s continuous',
            all_training_and_eval_start_standing=True,partial_starts=False,
            scope='frontal slow roll;not lateral or full-mouth goalkeeper;zero actuator delay inherited'),indent=2,default=str))
        runner=MjlabOnPolicyRunner(env,c,str(output),'cuda:0')
        last=[m for m in runner.alg.actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        initial=copy.deepcopy(runner.alg.actor.state_dict())
        raw._allow_partial_starts=False
        raw.seed(2026101952)
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
            status(state='training',phase='视觉择机俯身→接住正面慢球→双手控稳2秒',iteration=count,
                iteration_seconds=(time.time()-start)/count)
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
    status(state='evaluating',phase='独立验收：正面慢球与2秒持续控球',evaluation_completed=0,evaluation_steps=0)
    candidate=evaluate(final,args.episodes,output/'evaluation.json')
    after=preserved()
    assert before==after,'Preserved skills changed'
    (output/'preserved_after.json').write_text(json.dumps(after,indent=2))
    passed=candidate['success_rate']>=.75 and candidate['failure_rate']<=.1
    comparison=dict(frontal_slow_roll_passed=passed,needs_visual_review=True,preserved_inputs_unchanged=True,
        lateral_integration_completed=False,full_goalkeeper_completed=False,approved_for_replacement=False,
        baseline=baseline,candidate=candidate)
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    status(state='completed' if passed else 'needs_review',phase='本轮慢速来球训练结束，等待检查',evaluation=comparison,pid=None)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('check','train'))
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--episodes',type=int,default=16)
    p.add_argument('--baseline-episodes',type=int,default=8)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.mode=='check': check()
    else:
        with (HERE/'training.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                torch.manual_seed(2026101952)
                train(args)
            except BaseException as e:
                status(state='failed',phase='本轮已停止，请检查错误',error=f'{type(e).__name__}: {e}',pid=None)
                raise
