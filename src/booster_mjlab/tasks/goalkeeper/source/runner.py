"""mjlab runner adapter around the pinned original HIMPPO implementation."""

from pathlib import Path
import os
import inspect
import torch
from torch.utils.tensorboard import SummaryWriter
from .vendor.actor_critic import ActorCritic
from .vendor.amp import AMP
from .vendor.ppo import HIMPPO
from .vendor.normalizer import Normalizer
from .settings import MOTION_NAMES
from .state import state
from booster_mjlab.tasks.goalkeeper.amp_interfaces import GoalkeeperExpertSampler


class SourceMotion:
    """Source MotionLib's discrete first frame + randomized 0.25..1.25 speed."""

    def __init__(self, q, fps):
        self.q, self.fps = q, fps

    def get_expert_obs(self, batch_size):
        q = self.q
        n = len(q)
        # Keep the source clamp (which puts probability mass at the tail).
        t = torch.rand(batch_size, device=q.device).clamp(0, 1 - 2 / n)
        i = (t * n).floor().long()
        ratio = self.fps / 50 * (torch.rand(batch_size, device=q.device) + 0.25)
        nxt = i + ratio
        lo = nxt.floor().long().clamp_max(n - 1)
        hi = (lo + 1).clamp_max(n - 1)
        alpha = (nxt - lo)[:, None]
        return torch.cat((q[i], q[lo] * (1 - alpha) + q[hi] * alpha), -1)


class SourceRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        self.env, self.cfg, self.device = env, train_cfg, device
        self.log_dir = Path(log_dir) if log_dir else None
        self.current_learning_iteration = 0
        self.total_timesteps = 0
        actor = train_cfg["actor"]
        critic = train_cfg["critic"]
        model = ActorCritic(
            750,
            92,
            75,
            10,
            22,
            actor_hidden_dims=actor["hidden_dims"],
            critic_hidden_dims=critic["hidden_dims"],
            activation=actor["activation"],
            init_noise_std=actor["distribution_cfg"]["init_std"],
        ).to(device)
        accepted = inspect.signature(HIMPPO.__init__).parameters
        args = {k: v for k, v in train_cfg["algorithm"].items() if k in accepted}
        for k in (
            "value_smoothness_coef",
            "smoothness_upper_bound",
            "smoothness_lower_bound",
        ):
            args[k] = train_cfg[k]
        self.alg = HIMPPO(
            model,
            amp=AMP(44, 0.4, device=device),
            amp_normalizer=Normalizer(44),
            device=device,
            **args,
        )
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        if self.num_steps_per_env < 2:
            raise ValueError("Source PPO requires at least 2 steps")
        self.alg.init_storage(
            env.num_envs, self.num_steps_per_env, [750], [92], [22], [44]
        )
        self.last_losses = None

    def add_git_repo_to_log(self, path):
        pass  # source revisions are pinned in vendor/source_manifest.json

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        sampler = GoalkeeperExpertSampler(self.cfg["dataset_root"], self.device)
        self.alg.motion_buffer = {
            name: SourceMotion(q, fps)
            for name, q, fps in zip(MOTION_NAMES, sampler._positions, sampler._fps)
        }
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        writer = SummaryWriter(str(self.log_dir)) if self.log_dir else None
        episode_return = torch.zeros(self.env.num_envs, device=self.device)
        lengths = torch.zeros_like(episode_return)
        try:
            for _ in range(num_learning_iterations):
                sums = torch.zeros(3, device=self.device)
                with torch.no_grad():
                    for _ in range(self.num_steps_per_env):
                        actions = self.alg.act(obs["actor"], obs["critic"])
                        previous = obs["amp"].clone()
                        obs, task, dones, extras = self.env.step(
                            actions.to(self.env.device)
                        )
                        obs = obs.to(self.device)
                        task = task.to(self.device)
                        dones = dones.to(self.device)
                        for label, tensor in (
                            ("task", task),
                            ("actor", obs["actor"]),
                            ("critic", obs["critic"]),
                            ("amp", obs["amp"]),
                        ):
                            if not torch.isfinite(tensor).all():
                                raise FloatingPointError(
                                    f"Nonfinite {label} at control step {self.env.unwrapped.common_step_counter}"
                                )
                        window = torch.cat((previous, obs["amp"]), -1)
                        # Preserve source reset-spanning windows, unlike the tuned runner.
                        self.alg.process_amp_state(window)
                        amp = torch.zeros_like(task)
                        ids = (3 * obs["critic"][:, 78]).round().long()
                        for i, name in enumerate(MOTION_NAMES):
                            mask = ids == i
                            if mask.any():
                                amp[mask] = (
                                    self.alg.amp[name]
                                    .predict_reward(
                                        window[mask], self.alg.amp_normalizer
                                    )
                                    .squeeze(-1)
                                    * 0.5
                                )
                        total = 0.6 * task + 0.4 * amp
                        terminal = obs["critic"].clone()
                        terminal[dones.bool()] = state(
                            self.env.unwrapped
                        ).terminal_critic.to(self.device)[dones.bool()]
                        self.alg.process_env_step(total, dones, extras, terminal)
                        sums += torch.stack(
                            (task.mean() * 0.6, amp.mean() * 0.4, total.mean())
                        )
                        if writer:
                            source = state(self.env.unwrapped)
                            for region in range(6):
                                selected = ids == region
                                if selected.any():
                                    prefix = f"Region/{region}/"
                                    writer.add_scalar(
                                        prefix + "amp",
                                        (0.4 * amp[selected]).mean().item(),
                                        self.total_timesteps,
                                    )
                                    for name, values in source.rewards.items():
                                        weight = self.env.unwrapped.cfg.rewards[
                                            name
                                        ].weight
                                        writer.add_scalar(
                                            prefix + "reward_" + name,
                                            (
                                                0.6
                                                * self.env.unwrapped.step_dt
                                                * weight
                                                * values[selected]
                                            )
                                            .mean()
                                            .item(),
                                            self.total_timesteps,
                                        )
                                    for name, values in source.diagnostics.items():
                                        writer.add_scalar(
                                            prefix + name,
                                            values[selected].mean().item(),
                                            self.total_timesteps,
                                        )
                            for key, value in extras.get("log", {}).items():
                                if torch.isfinite(torch.as_tensor(value)).all():
                                    writer.add_scalar(
                                        key,
                                        float(torch.as_tensor(value).float().mean()),
                                        self.total_timesteps,
                                    )
                        episode_return += total
                        lengths += 1
                        if writer and dones.any():
                            writer.add_scalar(
                                "Train/episode_return",
                                episode_return[dones.bool()].mean().item(),
                                self.total_timesteps,
                            )
                            writer.add_scalar(
                                "Train/episode_seconds",
                                lengths[dones.bool()].mean().item()
                                * self.env.unwrapped.step_dt,
                                self.total_timesteps,
                            )
                        episode_return[dones.bool()] = 0
                        lengths[dones.bool()] = 0
                        self.total_timesteps += self.env.num_envs
                    self.alg.compute_returns(obs["critic"])
                self.last_losses = self.alg.update()
                if not all(
                    torch.isfinite(torch.as_tensor(x)).all() for x in self.last_losses
                ):
                    raise FloatingPointError(
                        f"Nonfinite source PPO/AMP update: {self.last_losses}; rollout rewards max={self.alg.storage.rewards.abs().max().item()}"
                    )
                self.current_learning_iteration += 1
                if writer:
                    for k, v in zip(
                        (
                            "value",
                            "surrogate",
                            "ball_estimation",
                            "region_classification",
                            "amp",
                            "expert",
                            "policy",
                        ),
                        self.last_losses,
                    ):
                        writer.add_scalar(
                            "Loss/" + k, float(v), self.current_learning_iteration
                        )
                    for k, v in zip(
                        ("task", "amp", "total"), sums / self.num_steps_per_env
                    ):
                        writer.add_scalar(
                            "Reward/" + k, float(v), self.current_learning_iteration
                        )
                    for name, value in state(self.env.unwrapped).rewards.items():
                        writer.add_scalar(
                            "RewardRaw/" + name,
                            value.mean().item(),
                            self.current_learning_iteration,
                        )
                    writer.add_scalar(
                        "Train/curriculum",
                        state(self.env.unwrapped).curriculumupdate,
                        self.current_learning_iteration,
                    )
                    writer.add_scalar(
                        "Train/curriculum_episode_length_ema",
                        state(self.env.unwrapped).episode_length_ema,
                        self.current_learning_iteration,
                    )
                    writer.add_scalar(
                        "Train/learning_rate",
                        self.alg.learning_rate,
                        self.current_learning_iteration,
                    )
                    if self.current_learning_iteration % self.cfg["save_interval"] == 0:
                        self.save(
                            self.log_dir / f"model_{self.current_learning_iteration}.pt"
                        )
                print(
                    f"Source K1 iteration {self.current_learning_iteration}: task/amp/total={(sums / self.num_steps_per_env).tolist()}",
                    flush=True,
                )
            if self.log_dir:
                self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
        finally:
            if writer:
                writer.close()

    def save(self, path, infos=None):
        s = state(self.env.unwrapped)
        n = self.alg.amp_normalizer
        payload = {
            "format": "k1-source-himppo-v1",
            "actor_critic": self.alg.actor_critic.state_dict(),
            "amp": {k: m.state_dict() for k, m in self.alg.amp.items()},
            "optimizer": self.alg.optimizer.state_dict(),
            "normalizer": {
                "mean": torch.from_numpy(n.mean),
                "var": torch.from_numpy(n.var),
                "count": n.count,
            },
            "iteration": self.current_learning_iteration,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.alg.learning_rate,
            "curriculum": {
                "level": s.curriculumupdate,
                "ranges": s.command_ranges,
                "startstep": s.startstep,
                "episode_length_ema": s.episode_length_ema,
            },
            "common_step_counter": self.env.unwrapped.common_step_counter,
            "last_step_counter": s.last_step_counter,
            "config": self.cfg,
            "infos": infos,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        os.replace(temp, path)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        p = torch.load(
            path, map_location=map_location or self.device, weights_only=True
        )
        if p.get("format") != "k1-source-himppo-v1":
            raise ValueError(
                "Source task requires a source HIMPPO checkpoint; old tuned PPO policies are incompatible"
            )
        self.alg.actor_critic.load_state_dict(p["actor_critic"], strict=strict)
        if load_cfg is None:
            for k, m in self.alg.amp.items():
                m.load_state_dict(p["amp"][k])
            self.alg.optimizer.load_state_dict(p["optimizer"])
            n = self.alg.amp_normalizer
            n.mean = p["normalizer"]["mean"].cpu().numpy()
            n.var = p["normalizer"]["var"].cpu().numpy()
            n.count = p["normalizer"]["count"]
            self.current_learning_iteration = p["iteration"]
            self.total_timesteps = p["total_timesteps"]
            self.alg.learning_rate = p["learning_rate"]
            s = state(self.env.unwrapped)
            s.curriculumupdate = p["curriculum"]["level"]
            s.startstep = p["curriculum"]["startstep"]
            s.episode_length_ema = p["curriculum"].get("episode_length_ema", -1.0)
            ranges = p["curriculum"]["ranges"].to(self.device)
            if ranges.shape != s.command_ranges.shape:
                raise ValueError("Full resume requires same number of environments")
            s.command_ranges.copy_(ranges)
            self.env.unwrapped.common_step_counter = p["common_step_counter"]
            s.last_step_counter = p["last_step_counter"]
            # Apply restored curriculum to new physical episodes.
            self.env.reset()
        return p.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return lambda obs: self.alg.actor_critic.act_inference(obs["actor"])
