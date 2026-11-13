import torch
from types import SimpleNamespace
from booster_mjlab.tasks.goalkeeper.source.jump_state import JumpState
from booster_mjlab.tasks.goalkeeper.source.state import SourceState
from booster_mjlab.tasks.goalkeeper.source.settings import REWARD_SCALES

j = JumpState(3, "cpu")
ids = torch.arange(3)
h = torch.full((3,), 0.5125)
j.reset(ids, h)
assert not j.update(h + 0.1, torch.zeros(3, 2)).any()  # cannot start in air
j.update(h, torch.ones(3, 2) * 10)
assert not j.update(h + 0.06, torch.zeros(3, 2)).any()  # debounce
out = j.update(
    h + torch.tensor([0.06, 0.04, 0.06]),
    torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]]),
)
assert out.tolist() == [True, False, False]
s = SimpleNamespace(
    jump_state=j,
    contact_feet_indices=[0, 1],
    contact_forces=torch.zeros(3, 2, 3),
    root_states=torch.zeros(3, 13),
    end_regions=torch.tensor([2, 3, 4]),
)
s.root_states[:, 2] = h
s.contact_forces[:, :, 2] = 10
reward = SourceState._reward_successland(s)
torch.testing.assert_close(reward, torch.tensor([6.0, 0.0, 0.0]))
j.reset(torch.tensor([0]), h[:1])
assert not j.jumped.any()
assert REWARD_SCALES["postlinvel"] == 1.0
print(
    "Support, takeoff debounce, small jump, one-foot exclusion, landing, partial reset passed"
)
