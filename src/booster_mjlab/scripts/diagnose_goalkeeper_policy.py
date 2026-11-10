"""Evaluate a frozen goalkeeper policy on strictly mirrored shot pairs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import mjlab  # noqa: F401 -- initialize task entry points
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from booster_mjlab.amp.modules.categorical_collector import CategoricalAmpCollector
from booster_mjlab.amp.runners.categorical_on_policy_runner import (
    CategoricalAmpOnPolicyRunner,
)
from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards
from booster_mjlab.tasks.goalkeeper.amp_interfaces import (
    MOTION_NAMES,
    get_amp_motion_ids,
)
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_amp_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.config.k1.rl_cfg import (
    booster_k1_goalkeeper_amp_runner_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.events import sample_six_region_shots
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings
from booster_mjlab.tasks.goalkeeper.mdp.rewards import intercept_movement_reward
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import get_blocker_target_kinematics
from booster_mjlab.scripts.goalkeeper_hand_timing import (
    add_hand_contact_sensors,
    hand_timing_sample,
    save_hand_timing_plots,
    summarize_hand_timing,
)


def make_mirrored_shots(settings, *, seed=42, left_region=0):
    """Create a pair in environment coordinates; mirror polar vector y only."""
    if left_region not in (0, 2, 4):
        raise ValueError("left_region must be 0, 2, or 4")
    left = sample_six_region_shots(
        1,
        settings,
        device="cpu",
        generator=torch.Generator().manual_seed(seed),
        regions=torch.tensor([left_region], dtype=torch.int64),
    )
    pair = {name: torch.cat((v, v.clone()), dim=0) for name, v in left.items()}
    for name in (
        "start_position_e",
        "target_position_e",
        "goal_target_position_e",
        "velocity",
    ):
        pair[name][1, 1] *= -1
    pair["region"][1] = left_region + 1
    return pair


def launch_fixed_shot(env, env_ids, ball_radius):
    """Replace only the reset launch event, retaining the full shot contract."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    sample = env._diagnostic_shot
    if len(ids) != 1 or env.num_envs != 1:
        raise ValueError("Sequential paired evaluation requires one environment")
    if not hasattr(env, "_full_goal_shots"):
        env._full_goal_shots = {
            name: torch.zeros_like(value, device=env.device)
            for name, value in sample.items()
        }
    for name, value in sample.items():
        env._full_goal_shots[name][ids] = value.to(env.device)
    state = torch.zeros((1, 13), device=env.device)
    state[:, :3] = (
        sample["start_position_e"].to(env.device) + env.scene.env_origins[ids]
    )
    state[:, 3] = 1
    velocity = sample["velocity"].to(env.device)
    state[:, 7:10] = velocity
    if sample["is_rolling"].item():
        # Angular velocity is an axial vector: derive it from the mirrored v.
        state[:, 10] = -velocity[:, 1] / ball_radius
        state[:, 11] = velocity[:, 0] / ball_radius
    env.scene["ball"].write_root_state_to_sim(state, env_ids=ids)


def yaw(quaternion):
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def install_reach_probe(env):
    """Capture shaping inputs when the real reward runs, without a second call."""
    cfg = env.reward_manager.get_term_cfg("eereach")
    original = cfg.func

    def probed(current_env, **params):
        reward = original(current_env, **params)
        asset = params["blocker_cfg"]
        robot = current_env.scene[asset.name]
        origin = current_env.scene.env_origins
        points = torch.cat(
            (
                robot.data.site_pos_w[:, asset.site_ids],
                robot.data.body_link_pos_w[:, asset.body_ids],
            ),
            dim=1,
        )
        target = current_env._full_goal_shots["target_position_e"]
        distances = (points - origin[:, None] - target[:, None]).norm(dim=-1)
        torso = robot.data.body_link_pos_w[:, asset.body_ids[0]] - origin
        velocity = robot.data.body_link_lin_vel_w[:, asset.body_ids[0]]
        ball = current_env.scene["ball"]
        ball_x = ball.data.root_link_pos_w[:, 0] - origin[:, 0]
        shot = current_env._goalkeeper_shot_state
        outcome = current_env._goalkeeper_outcome_state
        active = (
            (ball.data.root_link_lin_vel_w[:, 0] < -1e-6)
            & (ball_x > target[:, 0])
            & ~shot.stop_flag
            & ~shot.success_flag
            & ~outcome.shot_resolved
        )
        distance, closing_speed = get_blocker_target_kinematics(current_env, asset)
        movement = intercept_movement_reward(
            distance, closing_speed,
            **{name: params[name] for name in (
                "reach_threshold", "movement_weight", "movement_fade_distance",
                "movement_speed_limit",
            )},
        )
        current_env._reach_probe = {
            "reach_raw": float(reward.item()),
            "reach_active": bool(active.item()),
            "reach_phase": "attained" if distance.item() <= params["reach_threshold"] else "approach",
            "reach_distance": float(distances.min().item()),
            "nearest_blocker": int(distances.argmin().item()),
            "reward_torso_y": float(torso[0, 1]),
            "reward_torso_vy": float(velocity[0, 1]),
            "reward_lateral_error": float(target[0, 1] - torso[0, 1]),
            "reach_closing_speed": float(closing_speed.item()),
            "reach_movement_raw": float(movement.item()),
        }
        return reward

    cfg.func = probed


@torch.no_grad()
def run_trial(env, wrapped, runner, policy, shot, *, seed, max_seconds, hand_timing=False):
    env._diagnostic_shot = shot
    obs_dict, _ = env.reset(seed=seed)
    obs = wrapped.get_observations()
    for name, expected in shot.items():
        torch.testing.assert_close(env._full_goal_shots[name].cpu(), expected.cpu())
    robot, ball = env.scene["robot"], env.scene["ball"]
    torch.testing.assert_close(
        ball.data.root_link_pos_w - env.scene.env_origins,
        shot["start_position_e"].to(env.device),
    )
    torch.testing.assert_close(
        ball.data.root_link_lin_vel_w, shot["velocity"].to(env.device)
    )
    initial_state = (
        torch.cat(
            (
                robot.data.root_link_pose_w,
                robot.data.root_link_vel_w,
                robot.data.joint_pos,
                robot.data.joint_vel,
            ),
            dim=-1,
        )
        .cpu()
        .clone()
    )
    initial_actor = obs_dict["actor"].cpu().clone()
    collector = CategoricalAmpCollector(runner.amp_model, obs["amp"])
    feet = env.scene["feet_ground_contact"]
    foot_bodies = [
        robot.body_names.index(n) for n in ("left_foot_link", "right_foot_link")
    ]
    feet0 = robot.data.body_link_pos_w[:, foot_bodies].clone()
    root0 = robot.data.root_link_pos_w.clone()
    yaw0 = yaw(robot.data.root_link_quat_w).item()
    region = int(shot["region"].item())
    if hand_timing:
        env._hand_initial_sample = {"time": 0.0, "ball_contact": False, **hand_timing_sample(env)}
    rows = []
    for step in range(
        min(env.max_episode_length, math.ceil(max_seconds / env.step_dt))
    ):
        collector.begin_step(get_amp_motion_ids(env))
        action = policy(obs, stochastic_output=False)
        obs, task, dones, _ = wrapped.step(action)
        amp = collector.finish_step(obs["amp"], dones.bool())
        total, task_component, style = mix_amp_rewards(
            task,
            amp,
            style_weight=runner.amp_cfg["style_weight"],
            step_dt=env.step_dt,
            scale_rewards_by_dt=env.cfg.scale_rewards_by_dt,
        )
        origin = env.scene.env_origins
        root = robot.data.root_link_pos_w - origin
        target = env._full_goal_shots["target_position_e"]
        qyaw = yaw(robot.data.root_link_quat_w).item()
        state = env._goalkeeper_shot_state
        outcome = env._goalkeeper_outcome_state
        contact = env.scene["robot_ball_contact"].data
        has_ball_contact = bool((contact.found > 0).any()) or bool(
            (contact.force_history.square().sum(-1) > 1e-12).any()
        )
        row = {
            "step": step + 1,
            "time": (step + 1) * env.step_dt,
            "region": region,
            "target_y": float(target[0, 1]),
            "root_x": float(root[0, 0]),
            "root_y": float(root[0, 1]),
            "root_dy": float(robot.data.root_link_pos_w[0, 1] - root0[0, 1]),
            "root_yaw": qyaw,
            "root_yaw_delta": math.atan2(math.sin(qyaw - yaw0), math.cos(qyaw - yaw0)),
            "world_yaw_velocity": float(robot.data.root_link_ang_vel_w[0, 2]),
            "tilt_deg": math.degrees(
                math.acos(float((-robot.data.projected_gravity_b[0, 2]).clamp(-1, 1)))
            ),
            "ball_x": float(ball.data.root_link_pos_w[0, 0] - origin[0, 0]),
            "ball_y": float(ball.data.root_link_pos_w[0, 1] - origin[0, 1]),
            "task_reward": float(task.item()),
            "amp_reward": float(amp.item()),
            "task_component": float(task_component.item()),
            "style_component": float(style.item()),
            "total_reward": float(total.item()),
            "ball_contact": has_ball_contact,
            "save_proxy": bool(state.success_flag.item()),
            "save": bool(outcome.saved.item()),
            "touch_seen": bool(outcome.robot_contact_seen.item()),
            "shot_resolved": bool(outcome.shot_resolved.item()),
            "unresolved": not bool((outcome.shot_resolved | outcome.saved).item()),
            "goal": bool(outcome.goal_conceded.item()),
            "done": bool(dones.item()),
            "amp_valid": bool(collector.last_valid_mask.item()),
            **env._reach_probe,
        }
        if hand_timing:
            row.update(hand_timing_sample(env))
        scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
        terms = env.reward_manager.get_active_iterable_terms(0)
        for name, values in terms:
            row[f"task_{name}"] = (
                float(values[0]) * scale * (1 - runner.amp_cfg["style_weight"])
            )
        assert math.isclose(
            sum(row[f"task_{n}"] for n, _ in terms), row["task_component"], abs_tol=2e-5
        )
        for i, side in enumerate(("left", "right")):
            foot = robot.data.body_link_pos_w[0, foot_bodies[i]] - origin[0]
            sensor_i = list(feet.primary_names).index(f"{side}_foot_link")
            row[f"{side}_foot_x"] = float(foot[0])
            row[f"{side}_foot_y"] = float(foot[1])
            row[f"{side}_foot_dx"] = float(
                robot.data.body_link_pos_w[0, foot_bodies[i], 0] - feet0[0, i, 0]
            )
            row[f"{side}_foot_dy"] = float(
                robot.data.body_link_pos_w[0, foot_bodies[i], 1] - feet0[0, i, 1]
            )
            row[f"{side}_foot_dz"] = float(
                robot.data.body_link_pos_w[0, foot_bodies[i], 2] - feet0[0, i, 2]
            )
            row[f"{side}_foot_contact"] = bool((feet.data.found[0, sensor_i] > 0).any())
            row[f"{side}_foot_force"] = float(feet.data.force[0, sensor_i].norm())
            row[f"{side}_foot_yaw"] = float(
                yaw(robot.data.body_link_quat_w[0, foot_bodies[i]])
            )
        for name in env.termination_manager.active_terms:
            row[f"termination_{name}"] = bool(
                env.termination_manager.get_term(name).item()
            )
        if not all(
            math.isfinite(v) for v in row.values() if isinstance(v, (int, float))
        ):
            raise FloatingPointError("Non-finite diagnostic row")
        rows.append(row)
        if dones.item():
            break
    return rows, initial_state, initial_actor


def summarize(rows, shot):
    active = [r for r in rows if r["reach_active"]]
    early = [
        r for r in rows if r["time"] <= min(0.5, float(shot["intercept_time"].item()))
    ]
    moved = [r for r in active if abs(r["root_dy"]) >= 0.03]
    direction = (
        "none" if not moved else ("left" if moved[0]["root_dy"] > 0 else "right")
    )
    target_sign = 1 if shot["target_position_e"][0, 1] > 0 else -1
    result = {
        "region": int(shot["region"].item()),
        "frames": len(rows),
        "duration_s": rows[-1]["time"],
        "target_y_m": float(shot["target_position_e"][0, 1]),
        "nominal_intercept_s": float(shot["intercept_time"].item()),
        "first_movement": direction,
        "first_movement_correct": direction != "none"
        and (direction == "left") == (target_sign > 0),
        "root_dy_early_m": early[-1]["root_dy"] if early else 0.0,
        "root_dy_end_m": rows[-1]["root_dy"],
        "active_max_abs_yaw_deg": max(
            (abs(math.degrees(r["root_yaw_delta"])) for r in active), default=0.0
        ),
        "max_tilt_deg": max(r["tilt_deg"] for r in rows),
        "save": rows[-1]["save"],
        "save_proxy": any(r["save_proxy"] for r in rows),
        "unresolved": rows[-1]["unresolved"],
        "goal": any(r["goal"] for r in rows),
        "ball_contact": any(r["ball_contact"] for r in rows),
        "terminated_by": [
            n.removeprefix("termination_")
            for n, v in rows[-1].items()
            if n.startswith("termination_") and v
        ],
    }
    for side in ("left", "right"):
        result[f"{side}_air_frames_active"] = sum(
            not r[f"{side}_foot_contact"] for r in active
        )
        result[f"{side}_max_dz_active_m"] = max(
            (r[f"{side}_foot_dz"] for r in active), default=0.0
        )
        result[f"{side}_max_xy_active_m"] = max(
            (math.hypot(r[f"{side}_foot_dx"], r[f"{side}_foot_dy"]) for r in active),
            default=0.0,
        )
    for label, subset in (("all", rows), ("active", active)):
        result[label] = {"frames": len(subset)}
        if subset:
            for key in (
                "amp_reward",
                "task_component",
                "style_component",
                "total_reward",
                "reach_raw",
                "reach_closing_speed",
                "reach_movement_raw",
            ):
                result[label][f"mean_{key}"] = float(np.mean([r[key] for r in subset]))
            for key in (
                "task_eereach",
                "task_smoothness",
                "task_feet_slippage",
                "task_feetorientaion",
                "task_success",
                "task_stopball",
                "task_goal_conceded",
                "task_heading_deviation",
                "task_yaw_rate",
            ):
                result[label][f"sum_{key}"] = sum(r.get(key, 0.0) for r in subset)
    rotating = [r for r in active if abs(math.degrees(r["root_yaw_delta"])) >= 20]
    result["rotating_active"] = {"frames": len(rotating)}
    if rotating:
        result["rotating_active"].update(
            mean_amp_reward=float(np.mean([r["amp_reward"] for r in rotating])),
            mean_task_eereach=float(np.mean([r["task_eereach"] for r in rotating])),
            nearest_blocker_counts={
                str(i): sum(r["nearest_blocker"] == i for r in rotating)
                for i in range(5)
            },
        )
    return result


def save_plots(trials, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for trial in trials:
        if trial["seed"] != trials[0]["seed"]:
            continue
        fig, axes = plt.subplots(
            4, 1, figsize=(10, 11), sharex=True, layout="constrained"
        )
        for side, style in (("left", "-"), ("right", "--")):
            rows = trial[side]["rows"]
            t = [r["time"] for r in rows]
            axes[0].plot(
                t, [r["root_y"] for r in rows], style, label=f"{side} shot: root y"
            )
            axes[0].axhline(rows[0]["target_y"], linestyle=style, alpha=0.35)
            axes[1].plot(
                t, [math.degrees(r["root_yaw_delta"]) for r in rows], style, label=side
            )
            axes[2].plot(
                t, [r["amp_reward"] for r in rows], style, label=f"{side}: AMP"
            )
            axes[3].plot(
                t,
                [r["task_eereach"] for r in rows],
                style,
                label=f"{side}: reach contribution",
            )
        for ax, label in zip(
            axes, ("y (m)", "yaw change (deg)", "AMP reward", "mixed reach / step")
        ):
            ax.set_ylabel(label)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(
            f"{MOTION_NAMES[trial['left_region']]} / mirror, seed {trial['seed']}"
        )
        fig.savefig(
            output / f"pair_{trial['left_region']}_seed_{trial['seed']}.png", dpi=150
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--left-regions", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--heading-penalty", type=float, default=None)
    parser.add_argument("--yaw-rate-penalty", type=float, default=None)
    parser.add_argument("--hand-timing", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.max_seconds <= 0:
        parser.error("Require a checkpoint and positive max-seconds")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    overrides = {
        name: value
        for name, value in (
            ("heading_penalty", args.heading_penalty),
            ("yaw_rate_penalty", args.yaw_rate_penalty),
        )
        if value is not None
    }
    settings = FullGoalSettings(**overrides)
    pairs = [
        (seed, region, make_mirrored_shots(settings, seed=seed, left_region=region))
        for seed in args.seeds
        for region in args.left_regions
    ]
    cfg = booster_k1_goalkeeper_amp_env_cfg(num_envs=1, settings=settings)
    cfg.auto_reset = False
    if args.hand_timing:
        add_hand_contact_sensors(cfg)
    cfg.events["full_goal_launch"] = EventTermCfg(
        func=launch_fixed_shot,
        mode="reset",
        params={"ball_radius": settings.ball_radius},
    )
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env._diagnostic_shot = {k: v[:1].clone() for k, v in pairs[0][2].items()}
        wrapped = RslRlVecEnvWrapper(env)
        rl = booster_k1_goalkeeper_amp_runner_cfg()
        runner = CategoricalAmpOnPolicyRunner(
            wrapped, asdict(rl), log_dir=None, device=args.device
        )
        runner.load(str(args.checkpoint), load_cfg={"actor": True, "amp": True})
        policy = runner.get_inference_policy(args.device)
        runner.amp_model.eval()
        frozen = [
            (module, {k: v.clone() for k, v in module.state_dict().items()})
            for module in (policy, runner.amp_model)
        ]
        install_reach_probe(env)
        trials = []
        for seed, region, pair in pairs:
            trial = {"seed": seed, "left_region": region}
            initial_states = []
            initial_observations = []
            for side_i, side in enumerate(("left", "right")):
                shot = {k: v[side_i : side_i + 1].clone() for k, v in pair.items()}
                rows, initial_state, initial_actor = run_trial(
                    env,
                    wrapped,
                    runner,
                    policy,
                    shot,
                    seed=seed,
                    max_seconds=args.max_seconds,
                    hand_timing=args.hand_timing,
                )
                initial_states.append(initial_state)
                initial_observations.append(initial_actor)
                result = summarize(rows, shot)
                trial[side] = {
                    "summary": result,
                    "rows": rows,
                    "shot": {k: v.tolist() for k, v in shot.items()},
                    "initial_actor_observation": initial_actor.tolist(),
                }
                if args.hand_timing:
                    initial_sample = env._hand_initial_sample
                    trial[side]["hand_initial_sample"] = initial_sample
                    trial[side]["hand_timing"] = summarize_hand_timing(
                        [initial_sample, *rows], shot["target_position_e"][0].cpu().numpy(),
                        float(shot["intercept_time"].item()),
                    )
                path = args.output_dir / f"region_{region + side_i}_seed_{seed}.csv"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps({"seed": seed, "side": side, **result}), flush=True)
            torch.testing.assert_close(
                initial_states[0], initial_states[1], atol=1e-7, rtol=0
            )
            # Current actor packs five chronological [x,y,z,valid] ball frames
            # first, followed by identical proprioception and action history.
            expected_actor = initial_observations[0].clone()
            expected_actor[:, 1:20:4] *= -1
            torch.testing.assert_close(
                expected_actor, initial_observations[1], atol=1e-6, rtol=0
            )
            trials.append(trial)
        for module, state in frozen:
            assert all(
                torch.equal(v, state[k]) for k, v in module.state_dict().items()
            ), "Frozen model changed"
        summary = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "device": args.device,
            "step_dt": env.step_dt,
            "max_seconds": args.max_seconds,
            "settings": asdict(settings),
            "actor_amp_state_unchanged": True,
            "initial_robot_states_equal_within_pair": True,
            "initial_actor_observations_mirror_only_ball_y": True,
            "nearest_blocker_order": [
                "left_hand",
                "right_hand",
                "Trunk",
                "left_foot_link",
                "right_foot_link",
            ],
            "notes": [
                "Deterministic actor, frozen checkpoint normalizers; no optimizer or expert sampler updates.",
                "Task contribution includes weight, dt and mixing; AMP uses training collector done mask.",
                "Reward probe captures inputs at reward computation; post-step poses are refreshed one physics substep later.",
                "save_proxy is the legacy velocity event; save is the final observed clearance/held-stop outcome.",
            ],
            "trials": trials,
        }
        if args.hand_timing:
            summary["hand_timing_protocol"] = {
                "position_velocity_frame": "fixed field frame; root-relative velocities also recorded",
                "hand_contact_scope": "hand-link subtree including palm and end sphere, excluding forearm",
                "contact_time_precision_s": env.step_dt,
                "contact_history_physics_steps": cfg.decimation,
                "reference": "earliest actual robot/hand contact or measured inbound plane crossing; nominal time only as fallback",
                "target_radius_m": 0.15,
                "notes": "Target-point radius is a shaping proxy, not a collision test. A pre-plane contact can change the ball trajectory. Do not extrapolate events past trial termination.",
            }
        (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2))
        save_plots(trials, args.output_dir)
        if args.hand_timing:
            save_hand_timing_plots(trials, args.output_dir)
        print(f"Saved paired-policy evaluation to {args.output_dir}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
