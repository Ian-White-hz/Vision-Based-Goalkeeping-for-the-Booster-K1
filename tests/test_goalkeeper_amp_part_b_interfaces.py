import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from booster_mjlab.tasks.goalkeeper.amp_interfaces import (
    MOTION_NAMES,
    GoalkeeperExpertSampler,
    amp_joint_positions,
    get_amp_motion_ids,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    serial = root / "serial"
    serial.mkdir()
    reports = []
    fps = 50.0
    for motion_id, name in enumerate(MOTION_NAMES):
        time = np.arange(20, dtype=np.float32)[:, None] / fps
        dof_pos = motion_id + time + np.arange(22, dtype=np.float32)[None] * 0.001
        payload = {
            "fps": fps,
            "root_pos": np.zeros((20, 3), dtype=np.float32),
            "root_rot": np.tile([0, 0, 0, 1], (20, 1)).astype(np.float32),
            "dof_pos": dof_pos,
        }
        with (serial / f"{name}.pkl").open("wb") as file:
            pickle.dump(payload, file)
        reports.append(
            {
                "clip": name,
                "acceptance": {"status": "kinematic_checks_passed"},
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "joint_names": list(K1_JOINT_ORDER),
                "motion_id_map": {name: i for i, name in enumerate(MOTION_NAMES)},
                "reports": reports,
            }
        )
    )

    sampler = GoalkeeperExpertSampler(serial, device="cpu")
    ids = torch.tensor([5, 0, 3, 3, 1], dtype=torch.int64)
    windows = sampler.sample(ids, frame_dt=0.02)
    assert windows.shape == (5, 2, 22)
    assert windows.dtype == torch.float32
    torch.testing.assert_close(windows[:, 1] - windows[:, 0], torch.full((5, 22), 0.02))
    assert torch.equal(torch.floor(windows[:, 0, 0]).to(torch.long), ids)
    assert (windows[:, 0] >= ids[:, None]).all()
    assert (windows[:, 1] <= ids[:, None] + 0.4).all()
    assert tuple(sampler.joint_names) == tuple(K1_JOINT_ORDER)

    try:
        sampler.sample(ids.to(torch.int32), frame_dt=0.02)
    except ValueError as error:
        assert "int64" in str(error)
    else:
        raise AssertionError("Expected invalid motion id dtype to fail")

    for invalid_ids in (
        torch.tensor([-1], dtype=torch.int64),
        torch.tensor([6], dtype=torch.int64),
    ):
        try:
            sampler.sample(invalid_ids, frame_dt=0.02)
        except ValueError as error:
            assert "[0, 5]" in str(error)
        else:
            raise AssertionError("Expected invalid motion id value to fail")

    try:
        sampler.sample(torch.tensor([0], dtype=torch.int64), frame_dt=1.0)
    except ValueError as error:
        assert "shorter" in str(error)
    else:
        raise AssertionError("Expected a clip shorter than frame_dt to fail")

    manifest_path = root / "manifest.json"
    valid_manifest = manifest_path.read_text()
    invalid_manifest = json.loads(valid_manifest)
    invalid_manifest["joint_names"] = list(reversed(K1_JOINT_ORDER))
    manifest_path.write_text(json.dumps(invalid_manifest))
    try:
        GoalkeeperExpertSampler(serial, device="cpu")
    except ValueError as error:
        assert "joint_names" in str(error)
    else:
        raise AssertionError("Expected an invalid manifest to fail")
    finally:
        manifest_path.write_text(valid_manifest)

    invalid_payload = dict(payload)
    invalid_payload["dof_pos"] = payload["dof_pos"].copy()
    invalid_payload["dof_pos"][0, 0] = np.nan
    with (serial / "lefthand.pkl").open("wb") as file:
        pickle.dump(invalid_payload, file)
    try:
        GoalkeeperExpertSampler(serial, device="cpu")
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("Expected non-finite expert data to fail")


joint_pos = torch.arange(66, dtype=torch.float32).reshape(3, 22)
fake_env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos))},
    _full_goal_shots={"region": torch.tensor([0, 3, 5], dtype=torch.int64)},
)
asset_cfg = SimpleNamespace(name="robot", joint_ids=list(range(22)))
torch.testing.assert_close(amp_joint_positions(fake_env, asset_cfg), joint_pos)
torch.testing.assert_close(
    get_amp_motion_ids(fake_env), torch.tensor([0, 3, 5], dtype=torch.int64)
)

for invalid in (True, 1.0, -1, 6):
    try:
        FullGoalSettings(fixed_motion_id=invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Expected fixed_motion_id={invalid!r} to fail")

print("Goalkeeper Part B sampler、AMP 观测和类别接口检查通过")
