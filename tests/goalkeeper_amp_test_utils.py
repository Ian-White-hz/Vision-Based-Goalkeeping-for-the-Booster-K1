"""Small CPU-only fixtures; fake physics, real RSL-RL actor/critic and PPO."""

from dataclasses import asdict
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from booster_mjlab.amp.runners.categorical_on_policy_runner import (
    CategoricalAmpOnPolicyRunner,
)
from booster_mjlab.amp.runners.categorical_runner_cfg import CategoricalAmpRunnerCfg
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER

torch.set_num_threads(1)


def create_fake_expert_sampler(*, dataset_root, device):
    sampler = FakeExpertSampler()
    sampler.dataset_root = dataset_root
    sampler.device = device
    return sampler


class FakeEnv:
    def __init__(self, *, all_done=False, timeouts=False):
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.num_actions = 22
        self.step_dt = 0.02
        self.cfg = SimpleNamespace(scale_rewards_by_dt=True)
        self.motion_ids = torch.tensor([0, 3], dtype=torch.int64)
        self.q = torch.zeros(2, 22)
        self.step_count = 0
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(2, dtype=torch.int64)
        self.max_episode_length = 50
        self.all_done = all_done
        self.timeouts = timeouts

    @property
    def unwrapped(self):
        return self

    def get_observations(self):
        return TensorDict(
            {
                "actor": torch.zeros(2, 92),
                "critic": torch.zeros(2, 95),
                "amp": self.q.clone(),
            },
            batch_size=[2],
        )

    def step(self, actions):
        assert actions.shape == (2, 22)
        self.step_count += 1
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self.q += 0.01
        dones = torch.zeros(2, dtype=torch.int64)
        if self.step_count == 1:
            self.q[1] = -0.2
            self.motion_ids[1] = 5
            dones[1] = 1
        if self.all_done:
            dones[:] = 1
        self.episode_length_buf[dones.bool()] = 0
        return (
            self.get_observations(),
            torch.tensor([0.2, -0.4]),
            dones,
            {"time_outs": dones.bool() & self.timeouts},
        )


class FakeExpertSampler:
    joint_names = K1_JOINT_ORDER

    def __init__(self):
        self.requests = []

    def sample(self, motion_ids, *, frame_dt):
        assert frame_dt == 0.02
        self.requests.append(motion_ids.clone())
        values = motion_ids.to(torch.float32) * 0.05 + 0.2
        return values[:, None, None].expand(-1, 2, 22).clone()


def read_motion_ids(env):
    return env.motion_ids


def make_runner(
    *,
    env=None,
    num_steps=4,
    epochs=1,
    minibatches=1,
    amp_hidden_dims=(16, 8),
    log_dir=None,
    save_interval=50,
    warm_start=False,
    style_weight=0.4,
    reward_scale=1.0,
    reward_mode="source_noise",
    bind_interfaces=True,
    expert_sampler_factory=None,
    motion_id_reader=None,
    obs_normalization=False,
):
    cfg = CategoricalAmpRunnerCfg()
    cfg.actor.hidden_dims = (32, 16)
    cfg.critic.hidden_dims = (32, 16)
    cfg.actor.obs_normalization = obs_normalization
    cfg.critic.obs_normalization = obs_normalization
    cfg.amp.hidden_layer_sizes = amp_hidden_dims
    cfg.num_steps_per_env = num_steps
    cfg.algorithm.num_learning_epochs = epochs
    cfg.algorithm.num_mini_batches = minibatches
    cfg.algorithm.schedule = "fixed"
    cfg.upload_model = False
    cfg.logger = "tensorboard"
    cfg.save_interval = save_interval
    cfg.warm_start = warm_start
    cfg.amp.style_weight = style_weight
    cfg.amp.reward_scale = reward_scale
    cfg.amp.reward_mode = reward_mode
    cfg.amp.expert_sampler_factory = expert_sampler_factory
    cfg.amp.motion_id_reader = motion_id_reader
    runner = CategoricalAmpOnPolicyRunner(
        env or FakeEnv(), asdict(cfg), log_dir=log_dir, device="cpu"
    )
    if not bind_interfaces:
        return runner, None

    sampler = FakeExpertSampler()
    runner.bind_amp_interfaces(sampler, read_motion_ids)
    return runner, sampler
