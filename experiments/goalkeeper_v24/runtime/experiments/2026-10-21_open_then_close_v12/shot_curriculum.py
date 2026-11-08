"""Independent launch/goal-target distributions; reset sampling, never control.

The goal stays 2.4m wide at every level. Stages expand sampling rather than
changing the scene or silently excluding central or oblique shots.
"""
from dataclasses import asdict, dataclass
import torch


@dataclass(frozen=True)
class ShotRange:
    start_y_half_width: float
    target_y_half_width: float
    speed_min: float = .30
    speed_max: float = .40
    start_x_min: float = 1.8
    start_x_max: float = 2.6


RANGES = {
    'angled_near': ShotRange(.25, .35),
    'angled_half': ShotRange(.70, .65),
    'full_width_slow': ShotRange(1.4, 1.07),
    'full_width_medium': ShotRange(1.4, 1.07, .30, .60),
    'full_width_target': ShotRange(1.4, 1.07, .30, 1.0),
    'goal_gate_clear_miss_eval': ShotRange(1.4, 1.65, .40, .60, 3.0, 4.0),
    'goal_width_fast_edge_scoring_v2': ShotRange(1.4, 1.07, .50, .75, 3.0, 4.0),
    'goal_width_fast_edge_mix_v2': ShotRange(1.4, 1.65, .50, .75, 3.0, 4.0),
    'goal_gate_clear_miss_eval_v2': ShotRange(1.4, 1.65, .50, .75, 3.0, 4.0),
}


def sample_shots(count, stage, device, *, balanced_sides=None, generator=None,
                 goal_x=-.8, guard_x=0., ball_radius=.11):
    limits = RANGES[stage]
    assert count > 0 and limits.start_x_min > guard_x > goal_x
    random = torch.rand((count, 4), device=device, generator=generator)
    start = torch.zeros((count, 3), device=device)
    start[:, 0] = limits.start_x_min + random[:, 0]*(limits.start_x_max-limits.start_x_min)
    start[:, 1] = (2*random[:, 1]-1)*limits.start_y_half_width
    start[:, 2] = ball_radius + .002
    target = torch.zeros_like(start)
    target[:, 0] = goal_x
    if stage in ('goal_gate_clear_miss_eval','goal_gate_clear_miss_eval_v2'):
        u=random[:,2]
        sign=torch.where(u<.5,-1.,1.)
        target[:,1]=sign*(1.52+.13*(2*torch.abs(u-.5)))
    else:
        target[:, 1] = (2*random[:, 2]-1)*limits.target_y_half_width
    target[:, 2] = ball_radius
    fraction = (start[:, 0]-guard_x)/(start[:, 0]-goal_x)
    guard_y = start[:, 1] + fraction*(target[:, 1]-start[:, 1])
    if balanced_sides is not None:
        assert balanced_sides.shape == (count,)
        assert ((balanced_sides == -1) | (balanced_sides == 1)).all()
        # Reflect BOTH independently sampled y values to condition on interception
        # side. With equally represented +/- strata their unconditional joint
        # distribution remains independent and symmetric. No minimum |y| filter.
        reflection = torch.where(guard_y*balanced_sides < 0, -1., 1.)
        start[:, 1] *= reflection
        target[:, 1] *= reflection
        guard_y *= reflection
    speed = limits.speed_min + random[:, 3]*(limits.speed_max-limits.speed_min)
    direction = target[:, :2]-start[:, :2]
    velocity = direction/direction.norm(dim=1, keepdim=True)*speed[:, None]
    omega = torch.zeros_like(start)
    omega[:, 0] = -velocity[:, 1]/ball_radius
    omega[:, 1] = velocity[:, 0]/ball_radius
    return dict(start=start, target=target, velocity=velocity, angular_velocity=omega,
                speed=speed, guard_y=guard_y)


def describe(stage):
    r = RANGES[stage]
    return (f'前方{r.start_x_min}～{r.start_x_max}米，发球左右±{r.start_y_half_width}米；'
            f'门线目标左右±{r.target_y_half_width}米独立随机，'
            f'{r.speed_min:.2f}～{r.speed_max:.2f}米/秒；包含中央和斜向来球')


def specification(stage):
    return dict(stage=stage, **asdict(RANGES[stage]), goal_x=-.8, goal_width=2.4,
        ball_radius=.11, guard_stratification_plane_x=0.,
        sampling='independent start y and goal target y; balanced by nominal guard-plane crossing side; no central-shot exclusion')
