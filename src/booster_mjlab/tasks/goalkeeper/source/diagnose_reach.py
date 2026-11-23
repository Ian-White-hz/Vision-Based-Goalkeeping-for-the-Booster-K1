"""Why are shots missed? Hand reach, jumps and the AMP clips, per region.

For every shot (one episode per env, truth ball input, deterministic policy) it
records where the ball crosses the robot plane x=INTERCEPT_X, the highest hand,
the highest foot lift (jump) and the closest hand-ball distance before the ball
passes the robot. Missed high shots then split into "never raised the hand",
"raised but too low or too far aside" and "reached but did not stop". The clip
table gives the hand height and foot lift the AMP prior itself contains.

    uv run --no-sync python -m booster_mjlab.tasks.goalkeeper.source.diagnose_reach \
        --checkpoint logs/rsl_rl/.../model_12000.pt --num-envs 600
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path

import mjlab  # noqa: F401
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER

from .config import env_cfg, runner_cfg
from .runner import SourceRunner
from .settings import AMP_DATASET, BALL_RADIUS, GOAL_HEIGHT, GOAL_WIDTH, GOAL_X
from .settings import INTERCEPT_X
from .state import state

REGION_NAMES = ("mid-L", "mid-R", "high-L", "high-R", "low-L", "low-R")
HANDS = ("left_hand_end_ball", "right_hand_end_ball")
FEET = ("left_foot_link", "right_foot_link")
K1_XML = Path(__file__).parents[3] / "robots/booster_k1/xmls/k1.xml"
TOUCH = BALL_RADIUS + 0.05  # hand-ball centre distance counted as a touch


def clip_table():
    """Hand height and foot lift over each AMP clip (forward kinematics)."""
    m = mujoco.MjModel.from_xml_path(str(K1_XML))
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in K1_JOINT_ORDER]
    hands = [m.body(b).id for b in HANDS]
    feet = [m.body(b).id for b in FEET]
    print("AMP clips (heights above the lowest foot of frame 0):")
    print("  clip        frames  max hand z  hand |y| at max  max foot lift")
    for path in sorted(Path(AMP_DATASET).glob("*.pkl")):
        clip = pickle.load(open(path, "rb"))
        hand_z, hand_y, lift, ground = [], [], [], None
        for root_pos, root_rot, dof in zip(
            clip["root_pos"], clip["root_rot"], clip["dof_pos"]
        ):
            d.qpos[:3], d.qpos[3:7] = root_pos, np.asarray(root_rot)[[3, 0, 1, 2]]  # xyzw -> wxyz
            d.qpos[adr] = dof
            mujoco.mj_kinematics(m, d)
            fz = d.xpos[feet, 2]
            ground = fz.min() if ground is None else ground
            k = int(d.xpos[hands, 2].argmax())
            hand_z.append(d.xpos[hands[k], 2] - ground)
            hand_y.append(abs(d.xpos[hands[k], 1] - root_pos[1]))
            lift.append(fz.min() - ground)
        k = int(np.argmax(hand_z))
        print(
            f"  {path.stem:<10}  {len(hand_z):>6}  {hand_z[k]:>10.2f}  "
            f"{hand_y[k]:>15.2f}  {max(lift):>13.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    clip_table()

    dev, n = "cuda:0", args.num_envs
    env = ManagerBasedRlEnv(
        env_cfg(num_envs=n, play=True, randomize=False), device=dev
    )
    w = RslRlVecEnvWrapper(env)
    runner = SourceRunner(w, asdict(runner_cfg()), device=dev)
    runner.load(args.checkpoint, load_cfg=True)
    policy = runner.alg.actor_critic
    s = state(env)
    robot = env.scene["robot"]
    hand_ids = robot.find_bodies(HANDS, preserve_order=True)[0]
    foot_ids = robot.find_bodies(FEET, preserve_order=True)[0]

    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    obs = w.get_observations()
    orig = env.scene.env_origins
    region = s.end_regions.clone()
    nan = torch.full((n,), float("nan"), device=dev)
    zero = torch.zeros(n, dtype=torch.bool, device=dev)
    done, passed, crossed, goal, fell = (zero.clone() for _ in range(5))
    arm, deflect, jumped = zero.clone(), zero.clone(), zero.clone()
    cross_y, cross_z = nan.clone(), nan.clone()
    hand_max, lift_max, near = nan.clone(), nan.clone(), nan.clone()
    near_hand = torch.full((n, 3), float("nan"), device=dev)
    # Foot lift is measured from the lowest the feet get in the episode (the
    # robot settles a few cm after reset), so it is max - min of the lower foot.
    foot_low, foot_hi = nan.clone(), nan.clone()
    with torch.no_grad():
        for _ in range(env.max_episode_length - 1):
            obs, _, d, _ = w.step(policy.act_inference(obs["actor"]))
            alive = ~done
            ball = s.ball_states[:, :3] - orig
            bodies = robot.data.body_link_pos_w - orig[:, None]
            hands, feet = bodies[:, hand_ids], bodies[:, foot_ids]
            # Ball crossing the robot plane.
            new = alive & ~passed & (ball[:, 0] < INTERCEPT_X)
            cross_y = torch.where(new, ball[:, 1], cross_y)
            cross_z = torch.where(new, ball[:, 2], cross_z)
            # Reach and jump while the ball is still in front of the robot.
            act = alive & ~passed
            hz = hands[..., 2].max(1).values
            hand_max = torch.where(act, torch.fmax(hand_max, hz), hand_max)
            lower = feet[..., 2].min(1).values
            foot_low = torch.where(alive, torch.fmin(foot_low, lower), foot_low)
            foot_hi = torch.where(act, torch.fmax(foot_hi, lower), foot_hi)
            dist = (hands - ball[:, None]).norm(dim=-1)
            k = dist.argmin(1)
            dmin = dist.gather(1, k[:, None])[:, 0]
            closer = act & ~(dmin >= near)
            near = torch.where(closer, dmin, near)
            near_hand = torch.where(
                closer[:, None], hands[torch.arange(n, device=dev), k], near_hand
            )
            passed |= new
            # Contacts before the ball reaches the goal line.
            pre = alive & ~crossed
            arm |= pre & s.arm_touch
            deflect |= pre & s._deflected()
            jumped |= alive & s.jump_state.jumped
            # Goal line, as in vision/evaluate.py.
            g = alive & ~crossed & (ball[:, 0] < GOAL_X)
            goal |= g & (ball[:, 1].abs() < GOAL_WIDTH / 2) & (ball[:, 2] < GOAL_HEIGHT)
            crossed |= g
            fell |= alive & d.bool() & ~crossed
            done |= d.bool()

    lift_max = foot_hi - foot_low
    valid = crossed | ~fell
    kept = ~goal
    touched = near < TOUCH
    print(f"\nkept out {kept[valid].float().mean():.3f} over {int(valid.sum())} shots")
    print(
        "  arm touch = ball contact with an arm (shoulder to hand); deflected ="
        " ball velocity changed\n  (any body part); both before the goal line."
    )
    print("  region  n    arm touch  deflected  kept  kept|touch  kept|no touch  jumped")
    for i, name in enumerate(REGION_NAMES):
        m = valid & (region == i)
        if not m.any():
            continue
        t = m & arm
        u = m & ~arm
        rate = lambda x, mm: f"{x[mm].float().mean():.2f}" if mm.any() else "  - "  # noqa: E731
        print(
            f"  {name:<6} {int(m.sum()):>3}  {arm[m].float().mean():>9.2f}  "
            f"{deflect[m].float().mean():>9.2f}  {kept[m].float().mean():.2f}  "
            f"{rate(kept, t):>10}  {rate(kept, u):>13}  {jumped[m].float().mean():>6.2f}"
        )
    m = valid
    print(
        f"  all    {int(m.sum()):>3}  {arm[m].float().mean():>9.2f}  "
        f"{deflect[m].float().mean():>9.2f}  {kept[m].float().mean():.2f}  "
        f"{kept[m & arm].float().mean():>10.2f}  {kept[m & ~arm].float().mean():>13.2f}  "
        f"{jumped[m].float().mean():>6.2f}"
    )
    print("\nreach detail:")
    print(
        "  region  n    kept  touched  cross |y|  cross z  hand max z  "
        "hand-ball min  foot lift p50/p90  lifted>5cm"
    )
    for i, name in enumerate(REGION_NAMES):
        m = valid & (region == i)
        if not m.any():
            continue
        q = lambda x: x[m].nanmedian().item()  # noqa: E731
        lf = lift_max[m]
        print(
            f"  {name:<6} {int(m.sum()):>3}  {kept[m].float().mean():.2f}  "
            f"{touched[m].float().mean():>7.2f}  {q(cross_y.abs()):>9.2f}  "
            f"{q(cross_z):>7.2f}  {q(hand_max):>10.2f}  {q(near):>13.2f}  "
            f"{lf.nanquantile(0.5):>8.2f}/{lf.nanquantile(0.9):.2f}  "
            f"{(lf > 0.05).float().mean():>10.2f}"
        )

    # Missed high shots: where did the closest hand end up relative to the ball?
    high = valid & ((region == 2) | (region == 3))
    miss = high & ~kept
    if miss.any():
        print(f"\nmissed high shots ({int(miss.sum())} of {int(high.sum())}):")
        never = miss & (hand_max < 0.90)
        reached = miss & touched
        short = miss & ~never & ~reached
        print(f"  hand never above 0.90 m      {never[miss].float().mean():.2f}")
        print(f"  raised, did not reach ball   {short[miss].float().mean():.2f}")
        print(f"  within {TOUCH:.2f} m, still scored  {reached[miss].float().mean():.2f}")
        off = (near_hand - torch.stack((nan * 0, cross_y, cross_z), -1))[miss]
        print(
            "  closest hand minus ball crossing point, median dy "
            f"{(off[:, 1] * cross_y[miss].sign()).nanmedian():+.2f} m (outward +), "
            f"dz {off[:, 2].nanmedian():+.2f} m"
        )
        print("  by crossing height (missed / all high shots):")
        for lo, hi in ((0.85, 0.95), (0.95, 1.05), (1.05, 1.2)):
            for ylo, yhi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.8)):
                b = high & (cross_z >= lo) & (cross_z < hi)
                b &= (cross_y.abs() >= ylo) & (cross_y.abs() < yhi)
                if b.any():
                    print(
                        f"    z {lo:.2f}-{hi:.2f} |y| {ylo:.1f}-{yhi:.1f}: kept "
                        f"{kept[b].float().mean():.2f} (n={int(b.sum())}), hand max z "
                        f"{hand_max[b].nanmedian():.2f}, foot lift "
                        f"{lift_max[b].nanmedian():.2f}"
                    )
    env.close()


if __name__ == "__main__":
    main()
