"""Run user-selected shots with the frozen goalkeeper policies."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from bundle_support import bootstrap, model_path, verify_models

bootstrap()

import numpy as np
import torch
import viser
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.viewer.viser.viewer import ViserPlayViewer

import controller  # Registers the frozen handoff modules used by the actor.
import bridge
import random_goal_task
from channel_task import ChannelCaptureEnv
from fluid_control import FluidCaptureEnv
from posture_acceptance import make_cfg
from random_capture_task import raw_class
from train_bridge import config


@dataclass(frozen=True)
class Shot:
    speed: float = 0.35
    start_x: float = 3.5
    start_y: float = 0.4
    target_x: float = -0.8
    target_y: float = 0.4

    def sample(self, count, stage, device, *, guard_x=0., ball_radius=.11, **_):
        dx, dy = self.target_x - self.start_x, self.target_y - self.start_y
        distance = math.hypot(dx, dy)
        vx, vy = self.speed * dx / distance, self.speed * dy / distance
        guard_y = self.start_y + (guard_x - self.start_x) * dy / dx

        def rows(*values):
            return torch.tensor(values, device=device).expand(count, -1)

        return dict(start=rows(self.start_x, self.start_y, .112),
                    target=rows(self.target_x, self.target_y, .11),
                    velocity=rows(vx, vy),
                    angular_velocity=rows(-vy / ball_radius, vx / ball_radius, 0.),
                    speed=torch.full((count,), self.speed, device=device),
                    guard_y=torch.full((count,), guard_y, device=device))


class LiveEpisode:
    def reset(self):
        self.finished = False
        self.handoff_duration = type(self).handoff_duration
        return super().reset()

    def step(self, actions):
        result = super().step(actions)
        self.finished = bool(result[2][0])
        return result


class LiveV20(LiveEpisode, ChannelCaptureEnv):
    version = 'v20'


class LiveV24(LiveEpisode, FluidCaptureEnv):
    version = 'v24'


class ShotViewer(ViserPlayViewer):
    comparison = False
    shot_seed = 2026102961

    def setup(self):
        super().setup()
        gui = self._server.gui
        self.last_shot = None
        self.start_state = None
        self.outcomes = {}
        title = 'Same-shot comparison: v20 and v24' if self.comparison else 'Custom shot: v24'
        with gui.add_folder(title, order=-1):
            description = 'Set the initial speed, blue launch point, and yellow aim point.'
            if self.comparison:
                description += ' **Repeat with other version** reuses the shot and the starting pose.'
            gui.add_markdown(description)
            shot = Shot()
            speed = gui.add_number('Initial speed (m/s)', shot.speed, min=.10, max=1.50, step=.01)
            sx = gui.add_number('Start X (m)', shot.start_x, min=1., max=5., step=.01)
            sy = gui.add_number('Start Y (m)', shot.start_y, min=-1.8, max=1.8, step=.01)
            tx = gui.add_number('Aim X (goal line: -0.80 m)', shot.target_x, min=-1.2, max=0., step=.01)
            ty = gui.add_number('Aim Y (m)', shot.target_y, min=-1.5, max=1.5, step=.01)
            shoot = gui.add_button('Launch with v24' if self.comparison else 'Launch', color='blue')
            shoot.on_click(lambda _event: self.request_action('custom_shot', (read_shot(), 'v24')))
            if self.comparison:
                old_shoot = gui.add_button('Launch with v20')
                old_shoot.on_click(lambda _event: self.request_action('custom_shot', (read_shot(), 'v20')))
                self.repeat = gui.add_button('Repeat with other version', disabled=True)
                self.repeat.on_click(lambda _event: self.request_action('custom_shot', 'repeat'))
            self.result = gui.add_markdown('Paused. Ready to launch.')

        def read_shot():
            return Shot(speed.value, sx.value, sy.value, tx.value, ty.value)

        scene = self._server.scene
        self.start_marker = scene.add_icosphere('/shot/start', radius=.045, color=(48, 134, 244))
        self.target_marker = scene.add_icosphere('/shot/target', radius=.045, color=(244, 177, 36))
        self.aim = scene.add_line_segments('/shot/aim', points=np.zeros((1, 2, 3), dtype=np.float32),
                                           colors=(130, 150, 170), thickness=2., thickness_units='screen')
        self.show_shot(shot)
        if self._camera_overlays:
            for camera in self._camera_overlays.camera_viewers:
                camera._show_frustum_toggle.value = False

        @self._server.on_client_connect
        def set_camera(client):
            client.camera.position = (5.4, 4.6, 3.3)
            client.camera.look_at = (1.4, 0., .2)
            client.camera.up_direction = (0., 0., 1.)

    def show_shot(self, shot):
        self.start_marker.position = (shot.start_x, shot.start_y, .04)
        self.target_marker.position = (shot.target_x, shot.target_y, .04)
        self.aim.points = np.asarray([[self.start_marker.position, self.target_marker.position]], dtype=np.float32)

    def select_environment(self, version):
        self.env = self.environments[version]
        self._term_overlays.env = self.env
        self._debug_overlays.env = self.env
        if self._camera_overlays:
            for camera in self._camera_overlays.camera_viewers:
                camera._camera_sensor = self.env.unwrapped.scene[camera._camera_sensor.cfg.name]

    def _handle_custom_action(self, action, payload):
        if payload == 'repeat':
            shot = self.last_shot
            version = 'v20' if self.env.version == 'v24' else 'v24'
        elif isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], Shot):
            shot, version = payload
        else:
            return super()._handle_custom_action(action, payload)
        if shot is None:
            return True
        values = (shot.speed, shot.start_x, shot.start_y, shot.target_x, shot.target_y)
        bounds = ((.10, 1.50), (1., 5.), (-1.8, 1.8), (-1.2, 0.), (-1.5, 1.5))
        if not all(low <= value <= high for value, (low, high) in zip(values, bounds)):
            self.result.content = 'Adjust the shot parameters to the allowed range.'
            return True
        if shot != self.last_shot:
            self.outcomes.clear()
            self.start_state = None
        self.last_shot = shot
        self.select_environment(version)
        random_goal_task.sample_shots = shot.sample
        raw = self.env.unwrapped
        raw.seed(self.shot_seed)
        self.reset_environment()
        if self.comparison:
            robot = raw.scene['robot'].data
            ball = raw.scene['ball'].data
            roots = [getattr(entity, f'root_link_{field}_w').flatten()
                     for entity in (robot, ball) for field in ('pos', 'quat', 'lin_vel', 'ang_vel')]
            actual_start = torch.cat((*roots, robot.joint_pos.flatten(), robot.joint_vel.flatten())).cpu()
            if self.start_state is not None and not torch.allclose(actual_start, self.start_state, atol=1e-6, rtol=0):
                self.pause()
                self.result.content = 'Initial states differ. Launch the shot again.'
                print('INITIAL_STATE_MISMATCH', version, flush=True)
                self.start_state = None
                return True
            self.start_state = actual_start
            self.repeat.disabled = False
            self.outcomes.pop(version, None)
        self.show_shot(shot)
        self.update_result(f'{version} launched at {shot.speed:.2f} m/s.')
        self.resume()
        print('CUSTOM_SHOT', version, shot, flush=True)
        return True

    def update_result(self, status):
        rows = [status]
        if self.comparison:
            rows.extend(f'**{v}**  {self.outcomes.get(v, "Awaiting outcome.")}' for v in ('v20', 'v24'))
        self.result.content = '\n\n'.join(rows)

    def _execute_step(self):
        if self.env.finished:
            self.pause()
            return False
        stepped = super()._execute_step()
        if stepped and self.env.finished:
            self.pause()
            self._sync_ui_state()
            raw = self.env.unwrapped
            outcome = raw._pose_outcome
            if bool(outcome.success[0]):
                message = 'Strict save. Two-hand ground control held for two seconds.'
            elif bool(outcome.goal[0]):
                message = 'Goal.'
            elif bool(outcome.failed[0]):
                message = 'Physical failure. The trial ended early.'
            else:
                message = 'Timeout.'
            version = self.env.version
            contact = 'Interception contact' if bool(outcome.intercepted[0]) else 'No interception contact'
            hold = float(outcome.peak_hold[0]) * raw.step_dt
            self.outcomes[version] = f'{message} {contact}. Longest qualifying hold: {hold:.2f} s.'
            self.update_result(f'{version} trial complete.' if self.comparison else self.outcomes[version])
            print('SHOT_ENDED', version, self.outcomes[version], flush=True)
            return False
        return stepped


def make_live_environment(env_type, gait_path):
    bridge.GAIT_MODEL = gait_path
    cfg = make_cfg(1)
    cfg.auto_reset = False
    cfg.events['ball_reset'].params['stage'] = 'goal_width_fast_edge_scoring_v2'
    cfg.commands['twist'].debug_vis = False
    for sensor in cfg.scene.sensors:
        if hasattr(sensor, 'debug_vis'):
            sensor.debug_vis = False
    raw = raw_class(cfg=cfg, device='cuda:0')
    try:
        env = env_type(raw)
        raw.command_manager.get_term('twist').create_gui = lambda *args, **kwargs: None
        raw.seed(ShotViewer.shot_seed)
        env.reset()
        return env
    except Exception:
        raw.close()
        raise


@torch.inference_mode()
def main(port, comparison, smoke_steps):
    torch.set_num_threads(4)
    verify_models()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-capable GPU is required. See README.md.')

    random_goal_task.sample_shots = Shot().sample
    with ExitStack() as cleanup:
        env = make_live_environment(LiveV24, model_path('receive_gait'))
        cleanup.callback(env.close)
        environments = {'v24': env}
        runner = MjlabOnPolicyRunner(env, asdict(config()), None, 'cuda:0')
        actors = {}
        if comparison:
            runner.load(str(model_path('goalkeeper_v20')), load_cfg={'actor': True}, map_location='cuda:0')
            actors['v20'] = deepcopy(runner.alg.actor).eval()
            old_env = make_live_environment(LiveV20, model_path('lateral_gait'))
            cleanup.callback(old_env.close)
            environments['v20'] = old_env
        runner.load(str(model_path('goalkeeper_v24')), load_cfg={'actor': True}, map_location='cuda:0')
        actors['v24'] = runner.alg.actor.eval()
        if smoke_steps:
            for version, live in environments.items():
                for index in range(smoke_steps):
                    actions = actors[version](live.get_observations())
                    assert torch.isfinite(actions).all()
                    live.step(actions)
                    assert not live.finished, f'{version} ended during the startup check at step {index + 1}'
                print(f'SMOKE_OK {version}: {smoke_steps} physics and policy steps', flush=True)
            return
        label = 'K1 v20 / v24 comparison' if comparison else 'K1 v24 custom shot'
        server = viser.ViserServer(host='127.0.0.1', port=port, label=label)
        cleanup.callback(server.stop)
        server.gui.configure_theme(control_layout='collapsible', control_width='medium')
        viewer = ShotViewer(env, lambda obs: actors[viewer.env.version](obs), frame_rate=30., viser_server=server)
        viewer.comparison = comparison
        viewer.environments = environments
        viewer.pause()
        viewer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8155)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--smoke-steps', type=int, default=0,
                        help='Validate this many simulation steps without starting the viewer.')
    args = parser.parse_args()
    if not 0 <= args.smoke_steps <= 100:
        parser.error('--smoke-steps must be between 0 and 100')
    main(args.port, args.compare, args.smoke_steps)
