"""Numerical checks for coordinate conventions, IK derivatives and data layout."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from booster_mjlab.motion.goalkeeper_retarget import (
    K1Solver,
    RetargetOptions,
    UrdfKinematics,
    load_clip,
    source_targets,
    validate_motion,
)
from booster_mjlab.motion.motion_data import MotionFile
from booster_mjlab.robots.booster_k1.k1_constants import K1_XML
from booster_mjlab.robots.booster_k1.parallel_retarget import retarget_dof_pos


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(K1_XML))
        self.solver = K1Solver(self.model, RetargetOptions())
        self.q = np.array(
            [
                0,
                0,
                -0.3,
                -1.1,
                -0.4,
                -0.5,
                -0.3,
                1.1,
                0.4,
                0.5,
                -0.4,
                0.05,
                0.1,
                0.8,
                -0.4,
                -0.05,
                -0.4,
                -0.05,
                -0.1,
                0.8,
                -0.4,
                0.05,
            ]
        )

    def test_analytic_residual_jacobian_matches_finite_difference(self):
        target, feet = self.solver.forward(self.q)

        def checked(fun, x, **kwargs):
            analytic = kwargs["jac"](x)
            epsilon = 1e-6
            numeric = np.column_stack(
                [
                    (
                        fun(x + epsilon * np.eye(len(x))[i])
                        - fun(x - epsilon * np.eye(len(x))[i])
                    )
                    / (2 * epsilon)
                    for i in range(len(x))
                ]
            )
            np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=1e-4)
            return least_squares(fun, x, **kwargs)

        with patch(
            "booster_mjlab.motion.goalkeeper_retarget.least_squares",
            side_effect=checked,
        ):
            q, converged = self.solver.solve(target, feet, self.q, self.q, self.q)
        self.assertTrue(converged)
        actual, _ = self.solver.forward(q)
        np.testing.assert_allclose(actual, target, atol=1e-5)

    def test_speed_bounds_apply_between_frames(self):
        extreme = self.q.copy()
        extreme[2] = -2.5
        target, feet = self.solver.forward(extreme)
        result, _ = self.solver.solve(target, feet, extreme, self.q, self.q)
        self.assertTrue(
            np.all(
                np.abs(result - self.q) <= self.solver.speed_limits / 30 * 0.95 + 1e-8
            )
        )

    def test_parallel_layout_closes_equalities(self):
        positions, stats = retarget_dof_pos(np.repeat(self.q[None], 3, axis=0))
        self.assertEqual(positions.shape, (3, 34))
        self.assertEqual(stats, {"Left": 0, "Right": 0})
        motion = MotionFile(
            30.0, np.tile([0, 0, 0.7], (3, 1)), np.tile([0, 0, 0, 1], (3, 1)), positions
        )
        report = validate_motion(motion, parallel=True, contact=np.zeros((3, 2), bool))
        self.assertLess(report["max_equality_residual_m"], 1e-7)
        self.assertEqual(report["max_joint_limit_violation_rad"], 0.0)


class InputTests(unittest.TestCase):
    def test_loader_rejects_nonfinite_and_wrong_quaternion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "joint_id.txt"
            mapping.write_text("0 example_joint\n")
            clip = root / "clip.pt"
            arrays = {
                "joint_position": torch.zeros(3, 1),
                "base_position": torch.zeros(3, 3),
                "base_pose": torch.zeros(3, 4),
            }
            torch.save(arrays, clip)
            with self.assertRaisesRegex(ValueError, "unit xyzw"):
                load_clip(clip, mapping)
            arrays["base_pose"][:, 3] = 1
            arrays["joint_position"][0, 0] = float("nan")
            torch.save(arrays, clip)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_clip(clip, mapping)

    def test_urdf_origin_and_axis_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.urdf"
            path.write_text(
                '<robot name="fixture"><link name="pelvis"/><link name="tip"/><joint name="hinge" type="revolute"><parent link="pelvis"/><child link="tip"/><origin xyz="1 2 3" rpy="0 0 0"/><axis xyz="0 0 1"/></joint></robot>'
            )
            actual = UrdfKinematics(path).forward({"hinge": np.pi / 2})["tip"]
            np.testing.assert_allclose(actual[:3, 3], [1, 2, 3])
            np.testing.assert_allclose(
                actual[:3, :3],
                Rotation.from_euler("z", np.pi / 2).as_matrix(),
                atol=1e-8,
            )

    def test_options_reject_invalid_rates(self):
        for rate in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                RetargetOptions(fps=rate).validate()


class PublishedDataTests(unittest.TestCase):
    def test_all_six_published_clips_match_named_fk(self):
        repo = Path(__file__).resolve().parents[2] / "Humanoid-Goalkeeper"
        if not repo.is_dir():
            self.skipTest("Optional adjacent upstream checkout not present")
        dataset = repo / "legged_gym/resources/datasets/goalkeeper"
        fk = UrdfKinematics(repo / "legged_gym/resources/robots/g1/urdf/g1_29.urdf")
        model = mujoco.MjModel.from_xml_path(str(K1_XML))
        for path in sorted(dataset.glob("*.pt")):
            with self.subTest(clip=path.stem):
                arrays, names = load_clip(path, dataset / "joint_id.txt")
                self.assertEqual(len(names), 21)
                _, rotations, targets, _, _, metadata = source_targets(
                    arrays, names, fk, model, RetargetOptions()
                )
                self.assertEqual(targets.shape[1:], (8, 3))
                self.assertLess(
                    max(row["max_m"] for row in metadata["source_fk_check"].values()),
                    0.03,
                )
                self.assertAlmostEqual(
                    np.arctan2(rotations[0, 1, 0], rotations[0, 0, 0]), 0.0, places=6
                )


if __name__ == "__main__":
    unittest.main()
