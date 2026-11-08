"""Full-goal-width random rolling shots with RGB lateral/predictive receive.

The user explicitly requested expanding now, before the narrow capture gate
passed. This is an exploratory training domain, never an accepted full-width
skill. Reset metadata is display/evaluation only; it never chooses actions.
"""
import torch
from anticipatory_task import AnticipatoryEnv,PreparationOutcome,raw_class,measurements
from random_goal_task import make_cfg as range_cfg
from mjlab.managers import TerminationTermCfg
from shot_curriculum import RANGES,ShotRange,specification
from goalkeeper_sim.full_goal.settings import FullGoalSettings

STAGE='goal_width_fast_edge_mix_v2'
SETTINGS=FullGoalSettings()
# Train on faster, wider shots, including clear misses outside the goal mouth.
RANGES['goal_width_fast_edge_v1']=ShotRange(1.40,SETTINGS.target_half_width,.40,.60,3.0,4.0)
RANGES['goal_width_fast_edge_scoring_v2']=ShotRange(1.40,SETTINGS.target_half_width,.50,.75,3.0,4.0)
RANGES[STAGE]=ShotRange(1.40,1.65,.50,.75,3.0,4.0)

def make_cfg(num_envs=16):
    cfg=range_cfg(num_envs,STAGE)
    cfg.terminations['pose_success']=TerminationTermCfg(func=PreparationOutcome)
    return cfg

class RandomCaptureEnv(AnticipatoryEnv):
    def approach_lateral_command(self,error,velocity,actions):
        # Restore the previously used RGB lane/velocity lateral controller and
        # frozen natural gait. Preparation actions 22/23 keep their v6 meanings.
        command=(2.6*error[:,1]-.45*velocity[:,1]).clamp(-.24,.24)
        return torch.sign(command)*torch.maximum(command.abs(),torch.full_like(command,.18))

    def step(self,actions):
        obs,reward,done,extras=super().step(actions)
        e=self.env; d=e.scene['robot'].data
        feet=d.body_link_pos_w[:,e._collect_geometry[1],:2].mean(1)-e.scene.env_origins[:,:2]
        error=e._bridge_target[:,1]-feet[:,1]
        aligning=(e._bridge_stage<2)&~done.bool()
        # Favor the visually predicted interception lane rather than pursuing
        # the present lateral ball position in an oblique shot. Reward only;
        # actor/control inputs remain RGB and measured robot/tactile state.
        accuracy=torch.exp(-(error/.18).square())
        velocity_match=torch.exp(-((d.root_link_lin_vel_w[:,1]-e._bridge_cmd[:,1])/.15).square())
        reward+=e.step_dt*(2.*accuracy+velocity_match)*aligning
        extras['log']['Random/lane_error_m']=error.abs()
        return obs,reward,done,extras

env_class=RandomCaptureEnv
