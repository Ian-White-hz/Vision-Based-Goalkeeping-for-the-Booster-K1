"""Source randomization ranges adapted to MuJoCo's contact representation."""

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from mjlab.envs.mdp import dr


def ball_spec():
    # Keep source mass/radius; thin-shell inertia and calibrated contact replace
    # the source URDF inertia and engine-dependent default contact response.
    return mujoco.MjSpec.from_string("""<mujoco><worldbody>
      <body name="football"><freejoint/>
        <inertial pos="0 0 0" mass="0.1" diaginertia=".000667 .000667 .000667"/>
        <geom name="football_geom" type="sphere" size=".1" rgba=".95 .95 .95 1"
              friction=".5 .005 .0001" condim="6" priority="1" solref=".04 .5"/>
      </body></worldbody></mujoco>""")


# Region marker colours (RGBA), REGIONS order: mid L/R, high L/R, low L/R.
REGION_COLORS = (
    (0.95, 0.25, 0.20, 0.80),  # mid left: red
    (0.20, 0.45, 0.95, 0.80),  # mid right: blue
    (0.95, 0.80, 0.15, 0.80),  # high left: yellow
    (0.20, 0.80, 0.30, 0.80),  # high right: green
    (0.95, 0.50, 0.10, 0.80),  # low left: orange
    (0.65, 0.30, 0.90, 0.80),  # low right: purple
)
# Display tile per region: (column, row); column 0 is robot left (+y), row 0
# is the bottom third of the goal mouth.
REGION_TILES = ((0, 1), (1, 1), (0, 2), (1, 2), (0, 0), (1, 0))


def add_region_markers(spec, x, width, height):
    """Split the goal mouth into six translucent, collision-free colour tiles.

    Display only: tiles are equal thirds/halves of the mouth and do not match
    the REGIONS bounds, which are defined at the robot plane.
    """
    body = spec.body("goal")
    w, h = width / 2, height / 3
    for i, ((col, row), rgba) in enumerate(zip(REGION_TILES, REGION_COLORS)):
        body.add_geom(
            name=f"region_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, w / 2 * (1 - 2 * col), h * (row + 0.5)],
            size=[0.003, w / 2, h / 2],
            rgba=list(rgba),
            contype=0,
            conaffinity=0,
            mass=0.0,
            # Own group so viewers can draw tiles apart from the opaque posts
            # (see booster_mjlab.viewer.TRANSLUCENT_GEOM_GROUP).
            group=2,
        )
    return spec


def ball_cfg():
    return EntityCfg(
        spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(4, 0, 1))
    )


@requires_model_fields("geom_solref")
def restitution(env, env_ids, asset_cfg: SceneEntityCfg):
    """Approximate PhysX restitution by underdamped MuJoCo contact response.

    e=exp(-pi*zeta/sqrt(1-zeta^2)) is a linear oscillator mapping, NOT
    identical solver restitution. Keep this unavoidable engine difference explicit.
    """
    if not env.cfg.source_settings.randomize:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids]
    e = torch.rand((len(env_ids), 1), device=env.device).clamp(0.001, 0.999)
    log = e.log()
    zeta = (-log / torch.sqrt(torch.pi**2 + log.square())).clamp_min(0.05)
    env.sim.model.geom_solref[env_ids[:, None], ids[None, :], 1] = zeta
    # Positive-format MuJoCo solref has k proportional to 1/(timeconst*zeta)^2.
    # Hold stiffness fixed while reducing damping; otherwise high restitution
    # accidentally makes self contacts hundreds of times stiffer.
    env.sim.model.geom_solref[env_ids[:, None], ids[None, :], 0] = 0.02 / zeta


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def body_mass(env, env_ids, asset_cfg, ranges, operation):
    if env.cfg.source_settings.randomize:
        dr.body_mass(
            env, env_ids, asset_cfg=asset_cfg, ranges=ranges, operation=operation
        )


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def body_com_offset(env, env_ids, asset_cfg, ranges):
    if env.cfg.source_settings.randomize:
        dr.body_com_offset(env, env_ids, asset_cfg=asset_cfg, ranges=ranges)


@requires_model_fields("geom_friction")
def geom_friction(env, env_ids, asset_cfg, ranges, operation, axes, shared_random):
    if env.cfg.source_settings.randomize:
        dr.geom_friction(
            env,
            env_ids,
            asset_cfg=asset_cfg,
            ranges=ranges,
            operation=operation,
            axes=axes,
            shared_random=shared_random,
        )
