"""Independent balanced full standing-start acceptance; real contacts only."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch
from mjlab.rl import MjlabOnPolicyRunner
from controller import BalancedRawEnv,BalancedKeeperEnv,make_cfg,measurements,phase,preserved
from train_bridge import config
from state import status
HERE=Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate(checkpoint=None,episodes=32,path=None,seed=2026101959,kp=2.5,kd=.5,task='static',audit=False,shot_stage='angled_near',comparison=False,preparation_contact_fix=True):
    count=min(32,episodes)
    assert episodes%count==0 and count%2==0
    path=Path(path or HERE/'evaluation.json')
    assert not path.exists(),f'Preserve existing report: {path}'
    records_path=path.with_suffix('.episodes.jsonl')
    assert not records_path.exists(),f'Preserve existing episode stream: {records_path}'
    before=preserved()
    cfg_func=make_cfg; env_class=BalancedKeeperEnv; measure=measurements; raw_class=BalancedRawEnv
    if task=='incoming':
        from rolling_task import make_cfg as cfg_func,IncomingEnv as env_class,measurements as measure
    elif task=='reach':
        from reach_task import make_cfg as cfg_func,ReachIncomingEnv as env_class,measurements as measure
    elif task=='occlusion':
        from occlusion_task import make_cfg as cfg_func,OcclusionIncomingEnv as env_class,measurements as measure
    elif task=='touch':
        from touch_task import make_cfg as cfg_func,TouchIncomingEnv as env_class,measurements as measure
    elif task=='spacing':
        from spacing_touch_task import make_cfg as cfg_func,SpacingTouchEnv as env_class,measurements as measure
    elif task=='guard_offset':
        from guard_offset_task import make_cfg as cfg_func,GuardOffsetEnv as env_class,measurements as measure
    elif task=='smooth_guard':
        from smooth_approach_task import make_cfg as cfg_func,SmoothGuardEnv as env_class,measurements as measure
    elif task=='smooth_spacing':
        from smooth_approach_task import make_cfg as cfg_func,SmoothSpacingEnv as env_class,measurements as measure
    elif task=='anchor_width':
        from anchor_width_task import make_cfg as cfg_func,env_class,measurements as measure,AnchorWidthRawEnv as raw_class
    elif task in ('slow_touch','ik_touch','slow_ik_touch'):
        from damped_touch_task import make_cfg as cfg_func,measurements as measure,raw_class,SlowTouchEnv,IKTouchEnv,SlowIKTouchEnv
        env_class={'slow_touch':SlowTouchEnv,'ik_touch':IKTouchEnv,'slow_ik_touch':SlowIKTouchEnv}[task]
    elif task in ('grip_touch','grip_ik_touch'):
        from grip_hold_task import make_cfg as cfg_func,measurements as measure,raw_class,GripTouchEnv,GripIKTouchEnv
        env_class=GripTouchEnv if task=='grip_touch' else GripIKTouchEnv
    elif task=='random_goal':
        from random_goal_task import make_cfg as random_cfg,env_class,measurements as measure,raw_class,describe,specification
        cfg_func=lambda count:random_cfg(count,shot_stage)
    elif task=='fast_capture':
        from fast_capture_task import make_cfg as cfg_func,env_class,measurements as measure,raw_class
    elif task in ('settled_capture','quick_capture'):
        from settled_capture_task import make_cfg as cfg_func,measurements as measure,raw_class,SettledCaptureEnv,QuickCaptureEnv
        env_class=SettledCaptureEnv if task=='settled_capture' else QuickCaptureEnv
    elif task=='coordinated_capture':
        from coordinated_capture_task import make_cfg as cfg_func,measurements as measure,raw_class,env_class
    elif task=='anticipatory':
        from anticipatory_task import make_cfg as cfg_func,measurements as measure,raw_class,env_class
    elif task=='random_anticipatory':
        from random_capture_task import make_cfg as cfg_func,measurements as measure,raw_class,env_class
        from random_capture_task import STAGE
        shot_stage=STAGE
    cfg=cfg_func(count)
    if audit:
        from numerical_audit import audited_outcome
        cfg.terminations['pose_success'].func=audited_outcome(cfg.terminations['pose_success'].func,measure)
    raw=raw_class(cfg=cfg,device='cuda:0')
    raw._balanced_eval=True
    env=env_class(raw)
    if hasattr(env,'contact_reference_fix'):env.contact_reference_fix=preparation_contact_fix
    record_stream=None
    try:
        record_stream=records_path.open('x',encoding='utf-8',buffering=1)
        record_stream.write(json.dumps(dict(type='setup',task=task,seed=seed,episodes=episodes,
            shot_stage=shot_stage if task in ('random_goal','random_anticipatory') else None,audit=audit))+'\n')
        if checkpoint:
            runner=MjlabOnPolicyRunner(env,asdict(config()),None,'cuda:0')
            runner.load(str(checkpoint),load_cfg={'actor':True},map_location='cuda:0')
            actor=runner.alg.actor.eval()
        else: actor=lambda obs:obs['actor'].new_zeros((count,24))
        raw.seed(seed); obs,_=env.reset()
        assert (phase(raw)==0).all() and (raw._bridge_stage==0).all()
        finished=torch.zeros(count,device=env.device,dtype=torch.long)
        quota=episodes//count
        totals=dict(episodes=0,successes=0,failures=0,falls=0,timeouts=0,bilateral_contacts=0,
            moved=0,braked=0,unsafe_handoff=0,completed_descent=0,full_standing_starts=episodes,
            hold_seconds=[],endpoint_samples=0,stable_samples=0,min_collect_support_margin_m=1.,max_collect_foot_drift_m=0.)
        if task!='static': totals.update(body_blocked=0,goals=0,motion_seen=0,travelled=0)
        sides={side:dict(episodes=0,successes=0,falls=0,moved=0,braked=0,completed_descent=0) for side in ('left','right')}
        trace=[]; records=[]; steps=0
        label='固定球' if task=='static' else '0.30～0.40米/秒窄范围滚动来球'
        if task=='random_goal': label=describe(shot_stage)
        if task=='random_anticipatory':
            from random_capture_task import STAGE
            from shot_curriculum import describe,specification
            shot_stage=STAGE; label=describe(STAGE)
        if task=='anticipatory': label='正面3.0～3.4米、左右±.02米、.30～.40米/秒，停稳后预测到达时间并提前俯身'
        role='诊断复测' if comparison else '独立验收'
        variant={'settled_capture':'4秒俯身＋稳定触觉反馈','quick_capture':'3秒俯身＋稳定触觉反馈',
            'coordinated_capture':'3秒俯身＋稳定触觉反馈＋手部微调',
            'random_goal':'随机起点和门内目标＋已保存的3秒收球控制器'}.get(task,'完整衔接候选')
        status(state='evaluating',phase=f'{role}：{variant}，左右{label}',completed=0,target=episodes,iteration=None,error=None,pid=os.getpid(),
            training_pid=None,checkpoint=None,candidate_checkpoint=None,candidate_evaluation=None,
            finding='当前候选正在完整站立起步的左右分层测试中；训练轮数与离线参考不能代替本轮结果。',
            metrics_text=f'本轮已验收0/{episodes}。',evaluation=None)
        while int(finished.sum())<episodes:
            if hasattr(raw,'_balance_p'):
                raw._balance_p.fill_(kp); raw._balance_d.fill_(kd)
            active=finished<quota
            if task in ('random_goal','random_anticipatory'):
                # Snapshot reset metadata before step may auto-reset. Evaluation
                # only: these sampled targets never enter the RGB actor.
                shot_start=raw._incoming_launch_start.clone()
                shot_target=raw._incoming_goal_target.clone()
                shot_guard_y=raw._incoming_guard_y.clone()
            obs,reward,done,_=env.step(actor(obs))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            done=done.bool(); m=measure(raw); out=raw._pose_outcome
            sample=active&~done&(raw._bridge_stage==2)
            late=sample&(phase(raw)>.9999)
            totals['endpoint_samples']+=int(late.sum())
            totals['stable_samples']+=int((late&m['stable']).sum())
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
                records.append(dict(env=j,side='left' if out.last['left'][j] else 'right',
                    end_step=steps,flags={k:bool(v[j]) for k,v in out.last.items()},
                    peak_hold_seconds=float(out.last_peak_hold[j])*raw.step_dt))
                if hasattr(env,'goal_clear_miss_ever'):
                    records[-1]['goal_gate_clear_miss_seen']=bool(env.goal_clear_miss_ever[j])
                    records[-1]['goal_gate_predicted_y_m']=float(env.goal_predicted_y[j])
                    records[-1]['goal_gate_predicted_time_s']=float(env.goal_predicted_time[j])
                if audit:
                    records[-1]['trajectory_audit']={k:float(v[j]) if torch.isfinite(v[j]) else None
                        for k,v in out.audit_last.items()}
                if task in ('anticipatory','random_anticipatory'):
                    records[-1]['preparation_timing']={k:float(v[j]) if torch.isfinite(v[j]) else None
                        for k,v in out.timing_last.items()}
                if task in ('random_goal','random_anticipatory'):
                    records[-1].update(shot_start=shot_start[j].tolist(),goal_target=shot_target[j].tolist(),
                        nominal_guard_crossing_y=float(shot_guard_y[j]))
                record_stream.write(json.dumps(dict(type='episode',**records[-1]))+'\n')
                if task!='static':
                    for key,flag in (('body_blocked','body_blocked'),('goals','goal'),('motion_seen','motion_seen'),('travelled','travelled')):
                        totals[key]+=int(out.last[flag][j])
            if steps%5==0:
                trace.append(dict(step=steps,stage=raw._bridge_stage.tolist(),phase=phase(raw).tolist(),
                    root=(raw.scene['robot'].data.root_link_pos_w-raw.scene.env_origins).tolist(),
                    root_quaternion=raw.scene['robot'].data.root_link_quat_w.tolist(),
                    joint_pos=raw.scene['robot'].data.joint_pos.tolist(),
                    relative_com=(m['com'][:,:2]-m['feet'][:,:,:2].mean(1)).tolist(),margin=m['margin'].tolist(),
                    support=m['support'].tolist(),foot_drift=m['foot_drift'].tolist(),
                    correction=raw._balance_delta.tolist(),hand_forces=m['forces'].tolist(),
                    ball=m['ball_pos'].tolist(),ball_speed=m['speed'].tolist(),hold=(out.hold*raw.step_dt).tolist()))
                if getattr(env,'record_camera',False) and steps%10==0:
                    import base64,io
                    from PIL import Image
                    pixels=raw.scene['keeper_rgb'].data.rgb.cpu().numpy()
                    predicted=env.vision.mask.cpu().numpy()
                    frames=[]
                    for k in range(count):
                        if not active[k] or done[k]:frames.append(None);continue
                        a=io.BytesIO();b=io.BytesIO()
                        Image.fromarray(pixels[k]).save(a,format='JPEG',quality=88)
                        Image.fromarray(predicted[k].astype('uint8')*255).save(b,format='PNG')
                        frames.append(dict(time=(steps+1)*raw.step_dt,rgb=base64.b64encode(a.getvalue()).decode(),mask=base64.b64encode(b.getvalue()).decode()))
                    trace[-1]['camera_frames']=frames
                if hasattr(env,'handoff_latched'):
                    trace[-1].update(foot_tilt=m['foot_tilt'].tolist(),tilt=m['tilt'].tolist(),root_speed=raw.scene['robot'].data.root_link_lin_vel_w[:,:2].norm(dim=1).tolist(),angular_speed=raw.scene['robot'].data.root_link_ang_vel_w.norm(dim=1).tolist(),feet_speed=raw.scene['robot'].data.body_link_lin_vel_w[:,raw._collect_geometry[1],:2].norm(dim=-1).max(1).values.tolist(),soft_support=env.soft_support_time.tolist(),handoff_latched=env.handoff_latched.tolist(),gait_residual_max=env.gait_residual_max.tolist(),receive_width_target=env.reference_width().tolist())
                if hasattr(env,'channel_measured'):
                    trace[-1].update(channel_measured=env.channel_measured.tolist(),channel_gap=env.channel_gap.tolist(),channel_front=env.channel_front.tolist(),channel_rear=env.channel_rear.tolist(),channel_foot_width=env.channel_foot_width.tolist(),channel_reference_width=env.reference_width().tolist())
                if hasattr(env.vision,'control_quality'):
                    trace[-1].update(vision_control_quality=env.vision.control_quality.tolist(),vision_near=env.vision.near.tolist(),vision_raw_point=env.vision.raw_point.tolist(),receive_priority=env.receive_priority.tolist())
                if hasattr(env,'lane_ready'):
                    trace[-1].update(lane_ready=env.lane_ready.tolist(),lane_velocity=env.lane_velocity.tolist(),rgb_target=raw._bridge_target.tolist(),command=raw._bridge_cmd.tolist())
                if hasattr(env,'prediction_stable'):
                    trace[-1].update(prediction_stable=env.prediction_stable.tolist(),prediction_rms=env.prediction_rms.tolist(),intercept_target=env.intercept_target.tolist(),realigning=env.realigning.tolist(),realign_count=env.realign_count.tolist(),closure=env.closure.tolist())
                if hasattr(env,'goal_clear_miss'):
                    trace[-1].update(goal_gate_clear_miss=env.goal_clear_miss.tolist(),goal_gate_predicted_y=env.goal_predicted_y.tolist(),goal_gate_predicted_time=env.goal_predicted_time.tolist())
                if task!='static':
                    trace[-1].update(rgb_point=env.vision.point.tolist(),rgb_velocity=env.vision.velocity.tolist(),
                        rgb_usable=env.vision.usable.tolist(),rgb_fit_ready=env.vision.fit_ready.tolist(),
                        incoming_ready=raw._incoming_ready.tolist(),tactile_block=raw._incoming_touched.tolist())
                    trace[-1]['feet']=m['feet'].tolist()
                    if hasattr(raw,'_incoming_reach_x'):
                        trace[-1].update(reach_x=raw._incoming_reach_x.tolist(),reach_quiet=env.reach_quiet.tolist())
                    if hasattr(env.vision,'legacy_point'):
                        trace[-1].update(rgb_legacy_point=env.vision.legacy_point.tolist(),rgb_fit_loss=env.vision.fit_loss.tolist(),
                            rgb_correction=env.vision.offset.tolist())
                    if hasattr(env,'arm_offset'):
                        trace[-1].update(arm_offset=env.arm_offset.tolist(),servo_saturated=env.servo_saturated.tolist(),
                            tactile_target_radius=env.hand_radius.tolist())
                    if hasattr(raw,'_width_delta'): trace[-1]['width_correction']=raw._width_delta.tolist()
                    if hasattr(env,'ik_attempts'):
                        trace[-1].update(ik_attempts=env.ik_attempts.tolist(),ik_accepted=env.ik_accepted.tolist())
                    if hasattr(env,'grip_hold'):
                        trace[-1].update(grip_hold=env.grip_hold.tolist(),grip_rgb_anchor=env.grip_rgb_anchor.tolist(),
                            grip_rgb_time=env.grip_rgb_time.tolist())
                        trace[-1]['grip_reference_x']=env.grip_reference_x.tolist()
                    if hasattr(env,'preparation_contact'):
                        trace[-1]['actual_preparation_contact']=env.preparation_contact.tolist()
                    if hasattr(env,'micro_offset'):
                        trace[-1].update(micro_offset=env.micro_offset.tolist(),micro_attempts=env.micro_attempts.tolist(),
                            micro_accepted=env.micro_accepted.tolist())
                    if hasattr(env,'arrival_time'):
                        trace[-1].update(predicted_arrival_s=env.arrival_time.tolist(),preparation_action=env.preparation_action.tolist())
            steps+=1
            if steps%100==0:
                detail=''
                if task!='static':
                    detail=f"本批仍在测试的回合中，身体已接触挡球{int((out.body_blocked&active).sum())}，已开始俯身{int(((phase(raw)>0)&active).sum())}。"
                goals=f"，进球{totals['goals']}，超时{totals['timeouts']}" if task!='static' else ''
                status(completed=totals['episodes'],target=episodes,evaluation_steps=steps,
                    metrics_text=f"本轮已验收{totals['episodes']}/{episodes}，成功{totals['successes']}，摔倒{totals['falls']}{goals}。{detail}")
                partial=path.with_suffix('.partial.json'); tmp=partial.with_suffix('.tmp')
                tmp.write_text(json.dumps(dict(**totals,seed=seed,task=task,target=episodes,
                    evaluation_steps=steps,per_side=sides,episode_results=records),indent=2))
                tmp.replace(partial)
        for r in sides.values(): r['success_rate']=r['successes']/max(r['episodes'],1)
        totals.update(success_rate=totals['successes']/episodes,fall_rate=totals['falls']/episodes,
            endpoint_stable_fraction=totals['stable_samples']/max(1,totals['endpoint_samples']),per_side=sides,
            checkpoint=str(checkpoint) if checkpoint else 'zero learned residual, frozen gait handoff, COM feedback',
            kp=kp,kd=kd,seed=seed,steps=steps,episode_results=records,controller_variant=task,
            sampling_role='diagnostic/comparison sample, not a new independent holdout' if comparison else 'held-out evaluation',
            scope=('full standing start, RGB lateral alignment and measured stop, static ball, genuine bilateral grounded hold2s; rolling shots not tested' if task=='static' else
                'full standing start, narrow parallel rolling shots x1.8-2.6m,y+/-(.18-.22)m,speed.30-.40m/s; RGB lane prediction, upright body block then real bilateral grounded hold2s; full width/speed not tested'))
        if task in ('random_goal','random_anticipatory'):
            totals['scope']=describe(shot_stage)
            totals['shot_specification']=specification(shot_stage)
            totals['scope_note']='Goal geometry unchanged. Only this sampling stage has been tested; later stages not implied.'
        if task in ('anticipatory','random_anticipatory'):
            if task=='random_anticipatory':
                from random_capture_task import STAGE
            else:
                from anticipatory_task import STAGE
            from shot_curriculum import specification
            totals['scope']=label+'；真实手或身体拦截，真实双手地面留球2秒；仅限本次报告范围，不代表更快来球达标'
            totals['shot_specification']=specification(STAGE)
            totals['descent_before_contact']=sum(r['flags']['descent_before_contact'] for r in records)
            totals['ready_before_contact']=sum(r['flags']['ready_before_contact'] for r in records)
            totals['descent_before_interception']=sum(r['flags']['descent_before_interception'] for r in records)
            totals['ready_before_interception']=sum(r['flags']['ready_before_interception'] for r in records)
            totals['interception_policy']='Real hand or body contact; preparation must begin before first real interception. Body contact separately reported, never fabricated.'
        if audit:
            qualified=lambda r:(r['flags']['success'] and r['trajectory_audit']['hand_contact_lift_frames']==0
                and r['trajectory_audit']['minimum_entire_ball_goal_clearance_during_hold_m'] is not None
                and r['trajectory_audit']['minimum_entire_ball_goal_clearance_during_hold_m']>0)
            totals['trajectory_audit_successes']=sum(qualified(r) for r in records)
            totals['trajectory_audit_successful_lifts']=sum(r['flags']['success'] and
                r['trajectory_audit']['hand_contact_lift_frames']>0 for r in records)
            totals['numerical_overflow_episodes']=sum(r['trajectory_audit']['numerical_overflow_mask']!=0 for r in records)
            totals['numerically_valid_successes']=sum(qualified(r) and r['trajectory_audit']['numerical_overflow_mask']==0 for r in records)
            totals['trajectory_audit_note']='Audit-only: max ball height while touching either hand; >.14m is a lift flag; entire ball must remain outside goal plane during valid hold. Controller unchanged.'
        assert before==preserved()
        totals['prior_versions_unchanged']=True
        path.write_text(json.dumps(totals,indent=2))
        path.with_suffix('.trace.json').write_text(json.dumps(trace,indent=2))
        status(state='reviewing',phase='分析左右完整衔接验收',completed=episodes,pid=None,
            evaluation=str(path),metrics_text=f"{label}完整流程：成功{totals['successes']}/{episodes}；摔倒{totals['falls']}/{episodes}。全门宽0.3～1.0米/秒尚未验收。")
        print('BALANCED_EVALUATION',json.dumps(totals),flush=True)
        return totals
    finally:
        if record_stream is not None: record_stream.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=32)
    p.add_argument('--checkpoint'); p.add_argument('--seed',type=int,default=2026101959)
    p.add_argument('--kp',type=float,default=2.5); p.add_argument('--kd',type=float,default=.5)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--comparison',action='store_true')
    p.add_argument('--shot-stage',default='angled_near')
    p.add_argument('--task',choices=('static','incoming','reach','occlusion','touch','spacing','guard_offset','smooth_guard','smooth_spacing','anchor_width','slow_touch','ik_touch','slow_ik_touch','grip_touch','grip_ik_touch','random_goal','fast_capture','settled_capture','quick_capture','coordinated_capture','anticipatory','random_anticipatory'),default='static')
    p.add_argument('--output',default=str(HERE/'evaluation.json')); args=p.parse_args()
    torch.set_num_threads(4)
    evaluate(args.checkpoint,args.episodes,Path(args.output),args.seed,args.kp,args.kd,args.task,args.audit,args.shot_stage,args.comparison)
