"""Private FK and virtual-sphere clearance; no live ball state is accessed."""
from pathlib import Path
import hashlib,json,math,sys
import mujoco,numpy as np
from bundle_support import BUNDLE_ROOT, booster_asset

class ChannelGeometry:
    def __init__(self,joint_names=None,radius=.11):
        self.radius=radius
        xml=booster_asset('robots/booster_k1/xmls/k1.xml')
        spec=mujoco.MjSpec.from_file(str(xml))
        spec.worldbody.add_geom(name='private_virtual_ball',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=(radius,0,0),pos=(10,0,0),contype=0,conaffinity=0,group=5)
        self.model=spec.compile();self.data=mujoco.MjData(self.model)
        self.probe=self.model.geom('private_virtual_ball').id
        self.names=joint_names or [self.model.joint(i).name for i in range(1,self.model.njnt)]
        self.adr=[int(self.model.joint(n).qposadr[0]) for n in self.names]
        self.feet=[self.model.body(n).id for n in ('left_foot_link','right_foot_link')]
        self.legs=[[g for g in range(self.model.ngeom) if (self.model.geom(g).name or '').startswith(side)
                    and any(k in (self.model.geom(g).name or '') for k in ('shank','foot_collision'))
                    and int(self.model.geom_contype[g])!=0] for side in ('left','right')]
        assert all(self.legs)

    def pose(self,root,quat,joints):
        self.data.qpos[:3]=root;self.data.qpos[3:7]=quat;self.data.qpos[self.adr]=joints
        mujoco.mj_kinematics(self.model,self.data)

    def measure(self):
        feet=self.data.xpos[self.feet];centre=feet.mean(0);q=self.data.qpos[3:7]
        yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        c,s=math.cos(yaw),math.sin(yaw);rotation=np.array([[c,-s],[s,c]])
        def clearance(x,y):
            # Only the PRIVATE display/model-data probe is moved. Its position
            # is a hypothetical route through the measured robot stance.
            self.data.geom_xpos[self.probe]=np.r_[centre[:2]+rotation@np.array([x,y]),self.radius]
            return min(mujoco.mj_geomDistance(self.model,self.data,g,self.probe,1.,None) for ids in self.legs for g in ids)
        offsets=(-.015,0.,.015)
        entry=min(clearance(x,y) for x in (.20,.24) for y in offsets)
        routes=[min(clearance(x,y) for x in (-.06,-.02,.02,.06,.10)) for y in offsets]
        rear=max(routes)  # most permissive of the three central straight routes
        return dict(front_clearance_m=float(entry),rear_route_clearance_m=float(rear),
                    equivalent_gap_m=float(2*self.radius+2*rear),foot_width_m=float(np.linalg.norm(feet[0,:2]-feet[1,:2])))

def calibrate():
    k=BUNDLE_ROOT/'runtime/experiments/2026-10-21_open_then_close_v12'
    source=json.loads((k/'reach_paths_surface.json').read_text(encoding='utf-8'))
    geom=ChannelGeometry(source['joint_names']);target=.305;rows=[]
    low=max(w for w in source['widths'] if w<=target);high=min(w for w in source['widths'] if w>=target)
    blend=(target-low)/(high-low)
    for x in (-.04,.12,.28):
        a=next(p for p in source['paths'] if p['width']==low and p['ball_x']==x)['frames']
        b=next(p for p in source['paths'] if p['width']==high and p['ball_x']==x)['frames']
        for frac in (0.,.25,.5,.75,1.):
            i=round(frac*(len(a)-1));f={n:(1-blend)*np.asarray(a[i][n])+blend*np.asarray(b[i][n]) for n in ('root_x','root_z','root_pitch','joint_pos')}
            p=float(f['root_pitch']);geom.pose([float(f['root_x']),0,float(f['root_z'])+.005],[math.cos(p/2),0,math.sin(p/2),0],f['joint_pos'])
            rows.append(dict(reference_x=x,phase=frac,**geom.measure()))
    assert min(r['front_clearance_m'] for r in rows)>.008,'Would block before hand-reachable x=.20'
    rear=max(r['rear_route_clearance_m'] for r in rows)
    assert -.012<rear<-.004,'The ball-size gap is either too narrow or still passable'
    cfg=dict(ball_radius_m=.11,reference_width_target_m=target,minimum_reference_width_m=target,
             maximum_reference_width_m=.32,entry_clearance_min_m=.008,entry_plane_m=.20,
             rear_margin_target_m=-.004,allowed_overclosure_m=.008,
             note='22cm fixed task ball; choose the widest existing stance reference whose worst central route still blocks the ball while keeping >8mm front entry. Pixel diameter is not used as a metric ruler.')
    (BUNDLE_ROOT/'channel_calibration.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    goal=json.loads((BUNDLE_ROOT/'inputs/calibration_goal.json').read_text(encoding='utf-8'))
    g=ChannelGeometry();g.pose(goal['root'],goal['quaternion'],goal['joints']);goal_measured=g.measure()
    audit=dict(reference_validation=rows,parent_failed_pose=goal_measured,parent_provenance=goal['provenance'],
        min_reference_entry_clearance_m=min(r['front_clearance_m'] for r in rows),max_reference_rear_clearance_m=max(r['rear_route_clearance_m'] for r in rows),
        private_kinematics_only=True,live_state_writes=False)
    (BUNDLE_ROOT/'outputs/channel_calibration_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('CHANNEL_CALIBRATION',json.dumps(dict(parameters=cfg,failed_pose=goal_measured,min_entry=audit['min_reference_entry_clearance_m'],max_rear=audit['max_reference_rear_clearance_m'])),flush=True)
if __name__=='__main__':calibrate()
