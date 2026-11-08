"""Versioned physical and numerical review; old stop is a shadow diagnostic."""
import argparse,hashlib,json
from pathlib import Path

def review(report,path):
    assert report['prior_versions_unchanged']
    rows=[]
    for e in report['episode_results']:
        f=e['flags'];a=e.get('trajectory_audit',{});t=e['preparation_timing']
        peak=a.get('peak_ball_height_while_hand_contact_m');gap=a.get('minimum_entire_ball_goal_clearance_during_hold_m')
        shared=dict(full_descent=bool(f['descent']),real_interception=bool(f['intercepted'] and f['descent_before_interception']),
            no_fall_goal_or_failure=not(f['fallen'] or f['goal'] or f['failed']),no_unsafe_handoff=not f['unsafe'],
            grounded_contact_height=peak is not None and peak<=.115,no_large_lift=a.get('hand_contact_lift_frames')==0,
            entire_ball_outside_goal=gap is not None and gap>0,
            numerical_audit_present='numerical_overflow_mask' in a,no_numerical_overflow=a.get('numerical_overflow_mask')==0,
            no_steps_after_overflow=a.get('control_steps_after_numerical_overflow')==0)
        current=dict(**shared,controller_success=bool(f['success']),current_pose_stable=bool(f['pose_stable_now']),
            current_hold_2s=e['peak_hold_seconds']>=2.-1e-6)
        legacy=dict(**shared,controller_success=bool(f['legacy_success']),measured_stop=bool(f['braked']),
            legacy_hold_2s=t['legacy_peak_hold_s']>=2.-1e-6)
        rows.append(dict(env=e['env'],side=e['side'],end_step=e['end_step'],passed=all(current.values()),
            legacy_shadow_passed=all(legacy.values()),checks=current,legacy_shadow_checks=legacy))
    result=dict(rule_id='posture_grounded_hold_v2',source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        episodes=len(rows),qualified_successes=sum(r['passed'] for r in rows),
        legacy_shadow_successes=sum(r['legacy_shadow_passed'] for r in rows),records=rows,
        limitations='Legacy shadow is scored only until the new-rule termination; archived legacy full episodes remain separate. Same grounded-height and numerical review as v1.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    result=review(json.loads(a.report.read_text()),a.report)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))
