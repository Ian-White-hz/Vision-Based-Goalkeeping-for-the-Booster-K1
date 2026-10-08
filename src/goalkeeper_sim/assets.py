"""Simulation assets owned by the goalkeeping task."""

import mujoco

from mjlab.entity import EntityCfg

BALL_RADIUS_M = 0.09
BALL_MASS_KG = 0.25


def _ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_free_joint")
    body.add_geom(
        name="ball_collision",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS_M, 0.0, 0.0],
        mass=BALL_MASS_KG,
        rgba=[0.95, 0.25, 0.05, 1.0],
        friction=[0.7, 0.02, 0.002],
        contype=1,
        conaffinity=1,
    )
    return spec


def get_ball_cfg() -> EntityCfg:
    """Return a free rigid-body ball with an initial incoming velocity."""
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(2.0, 0.0, BALL_RADIUS_M),
            lin_vel=(-0.4, 0.0, 0.0),
        ),
        spec_fn=_ball_spec,
    )
