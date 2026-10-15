"""Geometry, reproducibility and negative outcome tests; no simulator or weights."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from goalkeeper_sim.full_goal.events import sample_shots
from goalkeeper_sim.full_goal.outcomes import FullGoalOutcome, goal_crossed
from goalkeeper_sim.full_goal.settings import FullGoalSettings


class FakeScene(dict):
    env_origins = torch.zeros((1, 3))


def fake_env():
    robot = SimpleNamespace(root_link_pos_w=torch.tensor([[0., 0., .5175]]),
                            projected_gravity_b=torch.tensor([[0., 0., -1.]]))
    ball = SimpleNamespace(root_link_pos_w=torch.tensor([[.15, 0., .11]]),
                           root_link_lin_vel_w=torch.zeros((1, 3)))
    scene = FakeScene(robot=SimpleNamespace(data=robot), ball=SimpleNamespace(data=ball))
    for name, value in (("feet_ground_contact", True), ("non_foot_ground_contact", False),
                        ("body_ball_contact", True), ("ball_ground_contact", True)):
        scene[name] = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[value]])))
    return SimpleNamespace(num_envs=1, device="cpu", scene=scene, common_step_counter=0, step_dt=.02)


class SamplingTests(unittest.TestCase):
    def test_independent_origins_reach_full_nominal_mouth(self):
        settings = FullGoalSettings()
        shots = sample_shots(10000, settings, generator=torch.Generator().manual_seed(2026101541))
        velocity = shots["velocity"]
        duration = (settings.goal_x - shots["start_x"]) / velocity[:, 0]
        hit_y = shots["start_y"] + duration * velocity[:, 1]
        self.assertTrue((duration > 0).all())
        self.assertTrue(torch.allclose(hit_y, shots["target_y"], atol=1e-5))
        self.assertTrue((hit_y.abs() + settings.ball_radius < settings.goal_width / 2).all())
        self.assertTrue(torch.allclose(velocity.norm(dim=1), shots["speed"], atol=1e-6))
        self.assertLess(float(hit_y.min()), -1.06)
        self.assertGreater(float(hit_y.max()), 1.06)
        self.assertLess(abs(float(torch.corrcoef(torch.stack((shots["start_y"], shots["target_y"])))[0, 1])), .05)
        self.assertTrue((shots["velocity"][:, 1] > .1).any())
        self.assertTrue((shots["velocity"][:, 1] < -.1).any())

    def test_seed_and_settings(self):
        settings = FullGoalSettings.from_json(Path(__file__).resolve().parents[1] / "configs/full_goal_scene.json")
        self.assertEqual(settings, FullGoalSettings())
        self.assertEqual(settings.episode_length_s, 16)
        a = sample_shots(32, settings, generator=torch.Generator().manual_seed(42))
        b = sample_shots(32, settings, generator=torch.Generator().manual_seed(42))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertEqual(sample_shots(0, settings)["velocity"].shape, (0, 2))
        for args in ({"speed_range": (0., .5)}, {"ball_radius": 2.}, {"start_x_range": (2., 1.)},
                     {"ball_mass": float("nan")}, {"ball_friction": (-1., 0., 0.)}):
            with self.assertRaises(ValueError):
                FullGoalSettings(**args)


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.env = fake_env()
        self.settings = FullGoalSettings()
        self.outcome = FullGoalOutcome(SimpleNamespace(params={"settings": self.settings}), self.env)

    def advance(self, count):
        for _ in range(count):
            self.env.common_step_counter += 1
            self.outcome.update(self.env)

    def test_whole_ball_crossing_and_wide_ball(self):
        position = torch.tensor([[-.85, 0., .11], [-.92, 0., .11], [-1., 1.5, .11]])
        self.assertEqual(goal_crossed(position, self.settings).tolist(), [False, True, False])

    def test_hold_duration_and_duplicate_calls(self):
        self.advance(19)
        self.assertFalse(self.outcome.stopped[0])
        before = self.outcome.hold.clone()
        self.outcome.update(self.env)
        self.assertTrue(torch.equal(before, self.outcome.hold))
        self.advance(1)
        self.assertTrue(self.outcome.stopped[0])

    def test_stationary_untouched_ball_is_not_success(self):
        self.env.scene["body_ball_contact"].data.found[:] = False
        self.advance(25)
        self.assertFalse(self.outcome.stopped[0])

    def test_fall_and_illegal_contact_block_success(self):
        self.env.scene["robot"].data.root_link_pos_w[:, 2] = .3
        self.advance(25)
        self.assertFalse(self.outcome.touched[0])
        self.assertFalse(self.outcome.stopped[0])
        self.assertTrue(self.outcome.failed[0])

    def test_goal_stays_failure_and_reset_retains_last_result(self):
        self.env.scene["ball"].data.root_link_pos_w[:, 0] = -1.1
        self.advance(1)
        self.env.scene["ball"].data.root_link_pos_w[:, 0] = .15
        self.advance(25)
        self.assertTrue(self.outcome.goal[0])
        self.assertFalse(self.outcome.stopped[0])
        self.outcome.reset(torch.tensor([0]))
        self.assertTrue(self.outcome.last["goal"][0])
        self.assertFalse(self.outcome.goal[0])
        self.assertFalse(self.outcome.touched[0])

    def test_airborne_and_still_moving_balls_are_not_stops(self):
        self.env.scene["ball_ground_contact"].data.found[:] = False
        self.advance(25)
        self.assertFalse(self.outcome.stopped[0])
        self.env.scene["ball_ground_contact"].data.found[:] = True
        self.env.scene["ball"].data.root_link_lin_vel_w[:, 0] = .2
        self.advance(25)
        self.assertFalse(self.outcome.stopped[0])


if __name__ == "__main__":
    unittest.main()
