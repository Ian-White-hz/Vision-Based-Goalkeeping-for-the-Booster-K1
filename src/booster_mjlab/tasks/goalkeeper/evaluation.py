"""Fixed six-region validation/holdout shots, using physical outcome adjudication."""

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import math

import torch
from tensordict import TensorDict

from booster_mjlab.amp.algorithms.categorical_ppo import MOTION_NAMES
from .mdp.outcomes import update_goalkeeper_outcome_state
from .mdp.terminations import excessive_body_tilt
from .scene.events import sample_six_region_shots


MISSING_ERROR_M = 2.0


def stable_config(value):
    """JSON representation without process-dependent function/object addresses."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: stable_config(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): stable_config(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [stable_config(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if callable(value):
        return (f"{getattr(value, '__module__', type(value).__module__)}:"
                f"{getattr(value, '__qualname__', type(value).__qualname__)}")
    return f"{type(value).__module__}:{type(value).__qualname__}"


def fixed_suite(settings, seeds, *, gravity):
    """Each seed yields three sampled left shots and their exact right mirrors."""
    samples = []
    symmetric = math.isclose(settings.start_y_range[0], -settings.start_y_range[1])
    for left_region in (0, 2, 4):
        left_bounds, right_bounds = settings.intercept_region_bounds[left_region:left_region + 2]
        symmetric &= all(math.isclose(a, b) for a, b in zip(
            (-left_bounds[1], -left_bounds[0], left_bounds[2], left_bounds[3]),
            right_bounds, strict=True))
        symmetric &= (settings.rolling_probability_by_region[left_region]
                      == settings.rolling_probability_by_region[left_region + 1])
    for seed in seeds:
        if not symmetric:
            # Custom asymmetric curricula must retain their actual right-side
            # bounds/mode probabilities rather than silently mirroring the left.
            samples.append(sample_six_region_shots(
                6, settings, generator=torch.Generator().manual_seed(seed),
                regions=torch.arange(6), gravity=gravity))
            continue
        left = sample_six_region_shots(
            3, settings, generator=torch.Generator().manual_seed(seed),
            regions=torch.tensor([0, 2, 4]), gravity=gravity,
        )
        right = {k: v.clone() for k, v in left.items()}
        for key in ("start_position_e", "target_position_e", "goal_target_position_e", "velocity"):
            right[key][:, 1] *= -1
        right["region"] += 1
        samples.extend((left, right))
    return {key: torch.cat([s[key] for s in samples]) for key in samples[0]}


def launch_fixed_shots(env, env_ids, settings):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    sample = env._evaluation_shots
    if not hasattr(env, "_full_goal_shots"):
        env._full_goal_shots = {k: torch.zeros_like(v) for k, v in sample.items()}
    for key, value in sample.items():
        env._full_goal_shots[key][ids] = value[ids]
    state = torch.zeros(len(ids), 13, device=env.device)
    state[:, :3] = sample["start_position_e"][ids] + env.scene.env_origins[ids]
    state[:, 3] = 1
    state[:, 7:10] = sample["velocity"][ids]
    rolling = sample["is_rolling"][ids]
    state[rolling, 10] = -state[rolling, 8] / settings.ball_radius
    state[rolling, 11] = state[rolling, 7] / settings.ball_radius
    env.scene["ball"].write_root_state_to_sim(state, env_ids=ids)


class ArrivalTracker:
    """First inbound plane crossing (interpolated), or earlier observed contact.

    Control-rate observations cannot recover contact's exact physics substep.
    Missing references are explicit and receive a finite penalty in selection.
    Points must be ordered left hand, right hand, then any legal body blockers.
    """

    def __init__(self, ball, points, regions, plane):
        self.previous_ball, self.previous_points = ball.clone(), points.clone()
        self.regions, self.plane = regions, plane
        self.records = [None] * len(regions)

    def update(self, ball, points, contact, active, *, time_s, dt):
        previous = self.previous_ball
        crossing = (previous[:, 0] >= self.plane) & (ball[:, 0] < self.plane)
        crossing &= ball[:, 0] < previous[:, 0]
        alpha = ((self.plane - previous[:, 0]) /
                 (ball[:, 0] - previous[:, 0]).clamp(max=-1e-9)).clamp(0, 1)
        alpha = torch.where(crossing, alpha, 1.)
        for i in (active & (crossing | contact)).nonzero().flatten().tolist():
            if self.records[i] is not None:
                continue
            a = alpha[i]
            b = previous[i].lerp(ball[i], a)
            p = self.previous_points[i].lerp(points[i], a)
            error = p - b
            distances = error.norm(dim=-1)
            hand = int(self.regions[i]) % 2
            self.records[i] = {
                "reference": "plane_crossing" if crossing[i] else "first_robot_contact",
                "time_s": time_s - dt + float(a) * dt,
                "selected_hand_error_m": float(distances[hand]),
                "selected_hand_error_vector_m": error[hand].tolist(),
                "selected_hand_speed_m_s": float(
                    (points[i, hand] - self.previous_points[i, hand]).norm() / dt),
                "nearest_hand_error_m": float(distances[:2].min()),
                "nearest_blocker_error_m": float(distances.min()),
            }
        self.previous_ball, self.previous_points = ball.clone(), points.clone()


def summarize_cases(cases):
    def group(rows):
        n = len(rows)
        if not n:
            raise ValueError("Evaluation must include every motion category")
        result = {f"{name}_rate": sum(bool(r[name]) for r in rows) / n for name in (
            "save", "safe_save", "goal", "fall", "contact", "miss", "unresolved", "cutoff",
        )}
        arrivals = [r["arrival"] for r in rows if r["arrival"] is not None]
        result["arrival_coverage"] = len(arrivals) / n
        result["arrival_hand_error_m"] = sum(
            min(r["arrival"]["selected_hand_error_m"], MISSING_ERROR_M)
            if r["arrival"] else MISSING_ERROR_M for r in rows) / n
        result["arrival_blocker_error_m"] = sum(
            min(r["arrival"]["nearest_blocker_error_m"], MISSING_ERROR_M)
            if r["arrival"] else MISSING_ERROR_M for r in rows) / n
        # Either hand is legitimate. Use nearest-hand distance on 0--3,
        # nearest legal blocker on 4--5. Never count a close hand as a save.
        result["selection_arrival_error_m"] = sum(
            min(r["arrival"]["nearest_hand_error_m" if r["region"] < 4
                              else "nearest_blocker_error_m"], MISSING_ERROR_M)
            if r["arrival"] else MISSING_ERROR_M for r in rows) / n
        return result

    metrics = {name: group([r for r in cases if r["region"] == i])
               for i, name in enumerate(MOTION_NAMES)}
    metrics["macro"] = {key: sum(m[key] for m in metrics.values()) / 6
                        for key in next(iter(metrics.values()))}
    return metrics


def selection_score(metrics):
    m = metrics["macro"]
    return [m["safe_save_rate"], min(metrics[n]["safe_save_rate"] for n in MOTION_NAMES),
            -m["fall_rate"], -m["goal_rate"], -m["unresolved_rate"],
            -m["selection_arrival_error_m"]]


class GoalkeeperEvaluator:
    def __init__(self, *, training_env, config, device):
        from mjlab.managers import EventTermCfg

        self.config, self.device = deepcopy(config), device
        for name in ("validation_seeds", "holdout_seeds"):
            values = config[name]
            if (not values or any(isinstance(s, bool) or not isinstance(s, int)
                                  or s < 0 or s >= 2**32 for s in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name} must contain distinct uint32 seeds")
        if set(config["validation_seeds"]) & set(config["holdout_seeds"]):
            raise ValueError("Validation and holdout seeds must be disjoint")
        if (isinstance(config["num_envs"], bool) or not isinstance(config["num_envs"], int)
                or config["num_envs"] <= 0):
            raise ValueError("evaluation.num_envs must be a positive integer")
        if not math.isfinite(config["max_seconds"]) or config["max_seconds"] <= 0:
            raise ValueError("evaluation.max_seconds must be positive and finite")
        self.env_cfg = deepcopy(training_env.cfg)
        self.settings = self.env_cfg.events["full_goal_launch"].params["settings"]
        self.suites = {name: fixed_suite(self.settings, config[f"{name}_seeds"],
                                        gravity=-self.env_cfg.sim.mujoco.gravity[2])
                       for name in ("validation", "holdout")}
        self.env_cfg.scene.num_envs = min(config["num_envs"],
                                        max(len(s["region"]) for s in self.suites.values()))
        self.env_cfg.auto_reset = False
        self.env_cfg.seed = config["validation_seeds"][0]
        self.env_cfg.events["full_goal_launch"] = EventTermCfg(
            func=launch_fixed_shots, mode="reset", params={"settings": self.settings})
        self.manifest = {
            "schema_version": 1,
            "selection": "safe_save,worst_class_safe_save,-fall,-goal,-unresolved,-arrival_error",
            "arrival": "first_observed_contact_or_interpolated_inbound_plane_crossing",
            "arrival_selection": "nearest_hand_regions_0_3_nearest_blocker_regions_4_5",
            "missing_or_capped_arrival_error_m": MISSING_ERROR_M,
            "settings": asdict(self.settings),
            "validation_seeds": list(config["validation_seeds"]),
            "holdout_seeds": list(config["holdout_seeds"]),
            "max_seconds": config["max_seconds"],
            "environment": stable_config(self.env_cfg),
            "shots": {name: stable_config(suite) for name, suite in self.suites.items()},
        }
        self.signature = hashlib.sha256(json.dumps(
            self.manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def evaluate(self, actor):
        from mjlab.envs import ManagerBasedRlEnv

        if getattr(actor, "is_recurrent", False):
            raise ValueError("Goalkeeper evaluator currently supports feedforward actors")
        report = {"manifest": self.manifest}
        # New independent simulation each evaluation; close even on policy errors.
        env = ManagerBasedRlEnv(deepcopy(self.env_cfg), device=self.device)
        try:
            for name, shots in self.suites.items():
                cases = self._run_suite(env, actor, shots, self.config[f"{name}_seeds"])
                report[name] = {"cases": cases, "metrics": summarize_cases(cases)}
        finally:
            env.close()
        report["selection_score"] = selection_score(report["validation"]["metrics"])
        return report

    def _run_suite(self, env, actor, shots, seeds):
        robot = env.scene["robot"]
        hand_ids = [robot.site_names.index(n) for n in ("left_hand", "right_hand")]
        body_ids = [robot.body_names.index(n) for n in ("Trunk", "left_foot_link", "right_foot_link")]

        def positions():
            origin = env.scene.env_origins
            return ((env.scene["ball"].data.root_link_pos_w - origin).clone(),
                    torch.cat((robot.data.site_pos_w[:, hand_ids],
                               robot.data.body_link_pos_w[:, body_ids]), dim=1).clone()
                    - origin[:, None])

        cases = []
        count, batch = len(shots["region"]), env.num_envs
        horizon = math.ceil(self.config["max_seconds"] / env.step_dt)
        for offset in range(0, count, batch):
            real_count = min(batch, count - offset)
            # Pad the last vector batch; padded episodes never enter metrics.
            indices = torch.arange(offset, offset + batch).clamp(max=count - 1)
            env._evaluation_shots = {k: v[indices].to(env.device) for k, v in shots.items()}
            env.common_step_counter = 0
            obs, _ = env.reset(seed=(seeds[0] + offset) % 2**32)
            alive = torch.arange(batch, device=env.device) < real_count
            falls = torch.zeros(batch, dtype=torch.bool, device=env.device)
            ball, points = positions()
            tracker = ArrivalTracker(ball, points, env._evaluation_shots["region"], self.settings.intercept_x)
            for step in range(1, horizon + 1):
                action = actor(TensorDict(obs, batch_size=[batch]), stochastic_output=False)
                action[~alive] = 0
                obs, _, terminated, truncated, _ = env.step(action)
                outcome = update_goalkeeper_outcome_state(env)
                tilt_cfg = env.cfg.terminations.get("excessive_body_tilt")
                tilt_params = tilt_cfg.params if tilt_cfg is not None else {}
                falls |= alive & excessive_body_tilt(env, **tilt_params)
                ball, points = positions()
                tracker.update(ball, points, outcome.robot_contact_seen, alive,
                               time_s=step * env.step_dt, dt=env.step_dt)
                done = terminated | truncated
                finished = alive & (done | (step == horizon))
                for i in finished.nonzero().flatten().tolist():
                    saved, goal = bool(outcome.saved[i]), bool(outcome.goal_conceded[i])
                    resolved, fall = bool(outcome.shot_resolved[i]), bool(falls[i])
                    shot_index = offset + i
                    cases.append({
                        "case_index": shot_index, "seed": seeds[shot_index // 6],
                        "region": int(env._evaluation_shots["region"][i]),
                        "save": saved, "safe_save": saved and not fall,
                        "goal": goal, "fall": fall,
                        "contact": bool(outcome.robot_contact_seen[i]),
                        "miss": resolved and not saved,
                        "unresolved": not resolved and not saved,
                        "cutoff": step == horizon and not bool(done[i]),
                        "terminated": bool(terminated[i]), "timeout": bool(truncated[i]),
                        "duration_s": step * env.step_dt, "arrival": tracker.records[i],
                    })
                alive &= ~finished
                if not alive.any():
                    break
                # Capture terminal data first, then satisfy manual-reset semantics.
                # Finished lanes may run filler episodes but cannot affect records.
                if done.any():
                    obs, _ = env.reset(env_ids=done.nonzero().flatten())
        return sorted(cases, key=lambda r: r["case_index"])
