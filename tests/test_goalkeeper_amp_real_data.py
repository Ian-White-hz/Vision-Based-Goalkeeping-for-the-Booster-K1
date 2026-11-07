from pathlib import Path

import torch

from booster_mjlab.tasks.goalkeeper.amp_interfaces import GoalkeeperExpertSampler


dataset = Path("artifacts/goalkeeper_k1/serial")
if not dataset.is_dir():
    print("SKIP: generated goalkeeper K1 clips are not available")
else:
    sampler = GoalkeeperExpertSampler(dataset, device="cpu")
    motion_ids = torch.arange(6, dtype=torch.int64)
    torch.manual_seed(123)
    actual = sampler.sample(motion_ids, frame_dt=0.02)

    torch.manual_seed(123)
    expected = torch.empty_like(actual)
    for motion_id, (positions, fps) in enumerate(
        zip(sampler._positions, sampler._fps, strict=True)
    ):
        duration = (positions.shape[0] - 1) / fps
        start = torch.rand(1) * (duration - 0.02)
        times = torch.stack((start, start + 0.02), dim=1)
        frame = times * fps
        lower = torch.floor(frame).long()
        upper = torch.clamp(lower + 1, max=positions.shape[0] - 1)
        alpha = (frame - lower.float()).unsqueeze(-1)
        expected[motion_id] = (
            positions[lower] * (1.0 - alpha) + positions[upper] * alpha
        )[0]

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    print("六个真实 goalkeeper clip 的 manifest、质量状态和时间插值检查通过")
