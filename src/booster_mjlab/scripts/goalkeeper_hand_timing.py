"""Read-only hand/ball timing probes for frozen goalkeeper evaluation."""

import numpy as np
import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


SIDES = ("left", "right")
AXES = ("x", "y", "z")


def add_hand_contact_sensors(cfg):
    """Separate palm/end-cap contacts from the existing whole-robot sensor."""
    cfg.scene.sensors = (*cfg.scene.sensors, *(
        ContactSensorCfg(
            name=f"{side}_hand_ball_contact",
            primary=ContactMatch(mode="subtree", entity="robot", pattern=f"{side}_hand_link"),
            secondary=ContactMatch(mode="body", entity="ball", pattern="football"),
            secondary_policy="error", fields=("found", "force"),
            reduce="maxforce", num_slots=1, history_length=cfg.decimation,
        ) for side in SIDES
    ))


def hand_timing_sample(env):
    """Synchronized post-step poses; contact evidence covers the last interval."""
    robot, ball = env.scene["robot"], env.scene["ball"]
    origin = env.scene.env_origins[0]
    target = env._full_goal_shots["target_position_e"][0]
    ball_pos = ball.data.root_link_pos_w[0] - origin
    row = {}
    for axis, value in zip(AXES, ball_pos):
        row[f"ball_{axis}"] = float(value)
    for axis, value in zip(AXES, ball.data.root_link_lin_vel_w[0]):
        row[f"ball_v{axis}"] = float(value)
    root_pos = robot.data.root_link_pos_w[0]
    root_vel = robot.data.root_link_lin_vel_w[0]
    omega = robot.data.root_link_ang_vel_w[0]
    quat = robot.data.root_link_quat_w[:1]
    for side in SIDES:
        index = robot.site_names.index(f"{side}_hand")
        pos_w = robot.data.site_pos_w[0, index]
        velocity = robot.data.site_lin_vel_w[0, index]
        offset = pos_w - root_pos
        local = quat_apply_inverse(quat, offset[None])[0]
        local_velocity = quat_apply_inverse(
            quat, (velocity - root_vel - torch.cross(omega, offset, dim=-1))[None]
        )[0]
        pos = pos_w - origin
        for stem, vector in (("", pos), ("v", velocity), ("local_", local),
                             ("local_v", local_velocity)):
            for axis, value in zip(AXES, vector):
                row[f"{side}_hand_{stem}{axis}"] = float(value)
        row[f"{side}_hand_target_distance"] = float((pos - target).norm())
        row[f"{side}_hand_ball_distance"] = float((pos - ball_pos).norm())
        contact = env.scene[f"{side}_hand_ball_contact"].data
        row[f"{side}_hand_contact"] = bool((contact.found > 0).any()) or bool(
            (contact.force_history.square().sum(-1) > 1e-12).any()
        )
        row[f"{side}_hand_contact_force_peak"] = float(contact.force_history.norm(dim=-1).max())
    return row


def sample_at_time(rows, time):
    """Linear 50Hz interpolation; do not extrapolate or interpolate contact flags."""
    if time < rows[0]["time"] or time > rows[-1]["time"]:
        return None
    times = [r["time"] for r in rows]
    keys = [k for k, v in rows[0].items() if type(v) is float]
    return {k: float(np.interp(time, times, [r[k] for r in rows])) for k in keys}


def summarize_hand_timing(rows, target, nominal_time, *, radius=.15):
    """Report timing signals, not causal proof or geometry-based contact labels.

    Actual robot contact can change the ball trajectory before the nominal
    plane. Use the earlier contact/plane event for pre-interception diagnosis;
    keep the nominal sample separately. Contact times have control-step precision.
    """
    target = np.asarray(target)
    crossing = None
    for before, after in zip(rows, rows[1:]):
        if before["ball_x"] > target[0] >= after["ball_x"]:
            fraction = (before["ball_x"] - target[0]) / (before["ball_x"] - after["ball_x"])
            crossing = float(before["time"] + fraction * (after["time"] - before["time"]))
            break
    whole_contact_time = next((r["time"] for r in rows if r.get("ball_contact", False)), None)
    contact_time = next((r["time"] for r in rows if r.get("ball_contact", False) or any(
        r.get(f"{side}_hand_contact", False) for side in SIDES
    )), None)
    events = [(t, name) for t, name in ((crossing, "ball_plane_crossing"),
                                      (contact_time, "first_robot_contact")) if t is not None]
    reference, kind = min(events) if events else (nominal_time, "nominal_only")
    reference_sample = sample_at_time(rows, reference)
    result = {
        "nominal_intercept_s": nominal_time,
        "actual_plane_crossing_s": crossing,
        "first_robot_contact_s": contact_time,
        "first_whole_robot_sensor_contact_s": whole_contact_time,
        "reference_s": reference, "reference_kind": kind,
        "reference_sample": reference_sample,
        "nominal_sample": sample_at_time(rows, nominal_time),
        "target_radius_m": radius,
        "hands": {},
    }
    for side in SIDES:
        prefix = f"{side}_hand_"
        before = [r for r in rows if r["time"] <= reference]
        entered = [r for r in rows if r[prefix + "target_distance"] < radius]
        before_entered = [r for r in before if r[prefix + "target_distance"] < radius]
        first_entry = entered[0]["time"] if entered else None
        nearest = min(before, key=lambda r: r[prefix + "target_distance"])
        h = {
            "first_target_entry_s": first_entry,
            "first_hand_contact_s": next((r["time"] for r in rows if r[prefix + "contact"]), None),
            "best_target_distance_before_reference_m": nearest[prefix + "target_distance"],
            "best_target_time_before_reference_s": nearest["time"],
            "never_entered_target_radius": not entered,
            "entered_after_reference": first_entry is not None and first_entry > reference,
        }
        if reference_sample is not None:
            pos = np.array([reference_sample[prefix + axis] for axis in AXES])
            ball = np.array([reference_sample["ball_" + axis] for axis in AXES])
            h.update(
                target_error_xyz_m=(pos - target).tolist(),
                ball_error_xyz_m=(pos - ball).tolist(),
                target_distance_at_reference_m=float(np.linalg.norm(pos - target)),
                ball_distance_at_reference_m=float(np.linalg.norm(pos - ball)),
                speed_at_reference_m_s=float(np.linalg.norm([
                    reference_sample[prefix + "v" + axis] for axis in AXES
                ])),
                local_speed_at_reference_m_s=float(np.linalg.norm([
                    reference_sample[prefix + "local_v" + axis] for axis in AXES
                ])),
                close_early_then_outside_at_reference=bool(
                    before_entered and np.linalg.norm(pos - target) >= radius
                ),
            )
        result["hands"][side] = h
    return result


def save_hand_timing_plots(trials, output):
    """Show the first seed's height, distance and velocity around ball arrival."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for pair in trials:
        if pair["seed"] != trials[0]["seed"]:
            continue
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="col", layout="constrained")
        for column, side in enumerate(SIDES):
            trial = pair[side]
            timing = trial["hand_timing"]
            nominal = timing["nominal_intercept_s"]
            rows = [trial["hand_initial_sample"], *trial["rows"]]
            rows = [r for r in rows if r["time"] <= nominal + .3]
            times = [r["time"] for r in rows]
            prefix = f"{side}_hand_"
            target = trial["shot"]["target_position_e"][0]
            axes[0, column].plot(times, [r["ball_z"] for r in rows], label="ball z")
            axes[0, column].plot(times, [r[prefix + "z"] for r in rows], label=f"{side} hand z")
            axes[0, column].axhline(target[2], color="gray", linestyle=":", label="target z")
            axes[1, column].plot(times, [r[prefix + "target_distance"] for r in rows], label="hand to fixed target")
            axes[1, column].plot(times, [r[prefix + "ball_distance"] for r in rows], label="hand site to ball center")
            axes[1, column].axhline(.15, color="gray", linestyle=":", label="target radius .15m")
            for axis in AXES:
                axes[2, column].plot(times, [r[prefix + "v" + axis] for r in rows], label=f"hand v{axis}")
            for axis, ylabel in zip(axes[:, column], ("height (m)", "distance (m)", "world velocity (m/s)")):
                axis.axvline(nominal, color="black", linestyle="--", alpha=.6, label="nominal arrival")
                contact = timing["hands"][side]["first_hand_contact_s"]
                if contact is not None and contact <= nominal + .3:
                    axis.axvline(contact, color="green", alpha=.6, label="first hand contact interval end")
                axis.set_ylabel(ylabel)
                axis.grid(alpha=.25)
                axis.legend(fontsize=7)
            axes[0, column].set_title(f"region {trial['summary']['region']}: {side} shot")
            axes[2, column].set_xlabel("time since launch (s)")
        fig.suptitle(f"Hand timing / seed {pair['seed']} / deterministic frozen actor")
        fig.savefig(output / f"hand_timing_pair_{pair['left_region']}_seed_{pair['seed']}.png", dpi=170)
        plt.close(fig)
