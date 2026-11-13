"""MuJoCo state adapter for the pinned source reward/observation contracts.

Upstream formulas and ordering are intentional, including hand targets for low
regions and repeated landing rewards. See vendor/NOTICE for attribution.
"""

from types import SimpleNamespace
import random
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from .vendor.rewards import SourceRewards
from .jump_state import JumpState
from .settings import INTERCEPT_X
from .settings import REGIONS, BOUNDS, EVAL_REGIONS, REWARD_SCALES, SourceSettings


class SourceState(SourceRewards):
    def __init__(self, env, settings):
        self.env, self.settings = env, settings
        self.device, self.num_envs, self.dt = env.device, env.num_envs, env.step_dt
        self.robot = env.scene["robot"]
        self.jids = self.robot.find_joints(K1_JOINT_ORDER, preserve_order=True)[0]
        self.hand_indices = self.robot.find_bodies(
            ("left_hand_end_ball", "right_hand_end_ball"), preserve_order=True
        )[0]
        self.knee_indices = self.robot.find_bodies(
            ("Left_Shank", "Right_Shank"), preserve_order=True
        )[0]
        self.contact_feet_indices = self.robot.find_bodies(
            ("left_foot_link", "right_foot_link"), preserve_order=True
        )[0]
        self.upper_body_index = self.robot.find_bodies("Trunk")[0][0]
        self.torso_index = self.upper_body_index
        self.upper_body_joint_indices = [
            i for i, n in enumerate(K1_JOINT_ORDER) if "Elbow" in n
        ]
        self.env_origins = env.scene.env_origins
        self.num_one_step_obs = 75  # ball3 + omega3 + gravity3 + 3*22
        self.actor_obs_length = 750
        self.end_regions = (
            torch.arange(self.num_envs, device=self.device) * 6 // self.num_envs
        ).clamp_max(5)
        self.command_ranges = torch.tensor(
            EVAL_REGIONS if settings.play else REGIONS, device=self.device
        )[self.end_regions].clone()
        self.command_bound = torch.tensor(BOUNDS, device=self.device)[self.end_regions]
        self.curriculumupdate, self.curriculumsigma = 0.0, 5.0
        self.last_step_counter, self.updated_step, self.obs_step = 0, -1, -1
        self.startstep = 50 - random.randint(3, 10)
        self.catchstep = torch.full(
            (env.num_envs,), 50, device=self.device, dtype=torch.long
        )
        self.vanish_step = torch.randint(0, 30, (env.num_envs,), device=self.device)

        def z(*shape):
            return torch.zeros((env.num_envs, *shape), device=self.device)

        self.end_target, self.ball_last = z(3), z(3)
        self.ball_vel, self.dist, self.stop_flag, self.success_flag = z(), z(), z(), z()
        self.has_in_air = z().bool()
        self.jump_state = JumpState(self.num_envs, self.device)
        self.actions, self.last_actions, self.last_last_actions = z(22), z(22), z(22)
        self.last_dof_vel = z(22)
        self.history = z(750)
        self.default_dof_pos = self.robot.data.default_joint_pos[:, self.jids].clone()
        self.init_dof_pos = self.default_dof_pos.clone()
        self.p_gains, self.d_gains, self.torque_limits, self.dof_vel_limits = [
            torch.zeros(22, device=self.device) for _ in range(4)
        ]
        for a in self.robot.actuators:
            for jid in a.target_ids.tolist():
                i = self.jids.index(jid)
                self.p_gains[i] = a.cfg.stiffness
                self.d_gains[i] = a.cfg.damping
                self.torque_limits[i] = a.cfg.effort_limit
                self.dof_vel_limits[i] = a.cfg.motor.velocity_limit
        self.dof_pos_limits = self.robot.data.soft_joint_pos_limits[
            0, self.jids
        ].clone()
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(
            env.num_envs, -1
        )
        self.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                reach_th=0.2,
                strict_th=0.15,
                max_contact_force=1000.0,
                soft_dof_vel_limit=0.9,
                soft_torque_limit=0.95,
            )
        )
        self.kp_factor, self.kd_factor = (
            torch.ones_like(self.actions),
            torch.ones_like(self.actions),
        )
        self.offset = z(22)
        self.rewards = {k: z() for k in REWARD_SCALES}
        self.first_visible = torch.full_like(self.catchstep, -1)
        self.guide_steps = torch.zeros_like(self.catchstep)
        self.first_near = torch.full_like(self.catchstep, -1)
        self.diagnostics = {}
        self.terminal_critic = z(92)

    def _reward_successland(self):
        force_z = self.contact_forces[:, self.contact_feet_indices, 2]
        self.has_in_air = self.jump_state.update(self.root_states[:, 2], force_z)
        both = (force_z > 1).all(-1)
        one = (force_z > 1).sum(-1) == 1
        jumping_region = (self.end_regions == 2) | (self.end_regions == 3)
        return (
            self.has_in_air.float()
            * (1 + 5 * both.float() - one.float())
            * jumping_region
        )

    def refresh(self):
        r, b = self.robot.data, self.env.scene["ball"].data
        self.dof_pos, self.dof_vel = (
            r.joint_pos[:, self.jids],
            r.joint_vel[:, self.jids],
        )
        self.root_states = torch.cat((r.root_link_pose_w, r.root_link_vel_w), -1)
        self.rigid_body_states = torch.cat(
            (
                r.body_link_pos_w,
                r.body_link_quat_w,
                r.body_link_lin_vel_w,
                r.body_link_ang_vel_w,
            ),
            -1,
        )
        self.ball_states = torch.cat((b.root_link_pose_w, b.root_link_vel_w), -1)
        self.torso_pos = r.root_link_pos_w
        self.base_quat = r.root_link_quat_w
        self.base_ang_vel, self.base_lin_vel = (
            r.root_link_ang_vel_b,
            r.root_link_lin_vel_b,
        )
        self.projected_gravity = r.projected_gravity_b
        self.torques = r.qfrc_actuator[:, self.jids]
        self.contact_forces = torch.zeros_like(r.body_link_pos_w)
        force = self.env.scene["source_feet_contact"].data.force
        if force is not None:
            self.contact_forces[:, self.contact_feet_indices] = (
                force.sum(dim=2) if force.ndim == 4 else force
            )

    def update(self):
        if self.updated_step == self.env.common_step_counter:
            return
        self.updated_step = self.env.common_step_counter
        self.refresh()
        self.catchstep -= 1
        self.actions = self.env.action_manager.action.clamp(-100, 100).clone()
        x = self.ball_states[:, 0] - self.env_origins[:, 0]
        changed = self.ball_states[:, 7] - self.ball_vel
        approach = (x < 0.5) & (x > 0.1) & (changed < 2.0)
        self.end_target[approach] = self.ball_states[approach, :3]
        self.end_target[:, 0] = torch.maximum(
            torch.minimum(self.end_target[:, 0], self.env_origins[:, 0] + 1.0),
            self.env_origins[:, 0] + INTERCEPT_X,
        )
        hands = self.rigid_body_states[:, self.hand_indices, :3]
        chosen = hands[
            torch.arange(self.num_envs, device=self.device), self.end_regions % 2
        ]
        self.dist = (self.end_target - chosen).norm(dim=-1)
        # Source reward order: success sees the previous flag; stopball updates it.
        for name in REWARD_SCALES:
            value = getattr(self, "_reward_" + name)()
            multiplier = 1.0
            if name in ("eereach", "success", "stopball"):
                multiplier = 1 + 0.5 * self.curriculumupdate
            if name in ("dof_pos_limits", "torque_limits"):
                multiplier = (
                    3.0
                    if self.curriculumupdate > 2
                    else 2.0
                    if self.curriculumupdate > 1
                    else 1.0
                )
            self.rewards[name] = value * multiplier
        near = (x <= 1.5) & (x >= 0) & (changed < 2.0)
        fresh_near = near & (self.first_near < 0)
        self.first_near[fresh_near] = 50 - self.catchstep[fresh_near]
        self.guide_steps += near.long()
        self.diagnostics = {
            "guide_seconds": self.guide_steps.float() * self.dt,
            "action_release_seconds": torch.full_like(
                x, (50 - self.startstep) * self.dt
            ),
            "near_phase": near.float(),
            "first_near_seconds": self.first_near.float() * self.dt,
        }
        self.terminal_critic = self.critic().clone()
        self.last_last_actions.copy_(self.last_actions)
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

    def critic(self):
        self.refresh()

        def local(x):
            return quat_apply_inverse(self.base_quat, x)

        hand = (
            self.rigid_body_states[:, self.hand_indices, :3] - self.torso_pos[:, None]
        )
        visible = (self.catchstep < self.startstep)[:, None]
        self.ball_local = local(self.ball_states[:, :3] - self.torso_pos) * visible
        return torch.cat(
            (
                self.ball_local,
                self.base_ang_vel * 0.25,
                self.projected_gravity,
                self.dof_pos - self.default_dof_pos,
                self.dof_vel * 0.05,
                self.actions,
                self.base_lin_vel * 2,
                self.end_regions[:, None] / 3.0,
                local(self.end_target - self.torso_pos),
                local(self.ball_states[:, 7:10]) * 0.2,
                local(hand[:, 1]),
                local(hand[:, 0]),
                self.dist[:, None],
            ),
            -1,
        )

    def actor(self):
        if self.obs_step == self.env.common_step_counter:
            return self.history
        self.obs_step = self.env.common_step_counter
        current = self.critic()[:, :75].clone()
        p = self.ball_local
        flying = (
            (p[:, 0] > 0.05)
            & (p[:, 0] < 3.4)
            & (p[:, 1] > -2)
            & (p[:, 1] < 2)
            & (p[:, 2] < 1.8)
            & (self.catchstep > 0)
            & ((p[:, 0] < self.ball_last[:, 0]) | (self.ball_last[:, 0] == 0))
        )
        self.ball_last = p.clone()
        if self.settings.randomize and not self.settings.play:
            scale = current.new_tensor(
                [0.08] * 3
                + [0.05] * 3
                + [0.05] * 3
                + [0.01] * 22
                + [0.075] * 22
                + [0.0] * 22
            )
            current += (2 * torch.rand_like(current) - 1) * scale
            flying &= self.catchstep > self.vanish_step
        visible = flying & (self.catchstep < self.startstep)
        first = visible & (self.first_visible < 0)
        self.first_visible[first] = 50 - self.catchstep[first]
        self.diagnostics["first_visible_seconds"] = self.first_visible.float() * self.dt
        current[:, :3] *= flying[:, None]
        self.history = torch.cat((self.history[:, 75:], current), -1).clamp(-100, 100)
        return self.history


def state(env, settings=None):
    if not hasattr(env, "_source_goalkeeper"):
        env._source_goalkeeper = SourceState(
            env, settings or getattr(env.cfg, "source_settings", SourceSettings())
        )
    return env._source_goalkeeper


def reward(env, name):
    s = state(env)
    s.update()
    return s.rewards[name]


def observation(env, kind):
    s = state(env)
    if kind == "actor":
        return s.actor()
    if kind == "critic":
        return s.critic().clamp(-100, 100)
    if kind == "amp":
        return s.robot.data.joint_pos[:, s.jids]
    raise ValueError(kind)


def termination(env, kind):
    s = state(env)
    s.update()
    if kind == "timeout":
        return env.episode_length_buf > env.max_episode_length
    if kind == "knees":
        return s.rigid_body_states[:, s.knee_indices, 2].amin(-1) < 0.1
    if kind == "gravity":
        return s.projected_gravity[:, :2].norm(dim=-1) > 0.8
    if kind == "force":
        return (
            s.contact_forces[:, s.contact_feet_indices].norm(dim=-1).mean(-1) > 1500.0
        )
    raise ValueError(kind)
