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
from .settings import MOTION_NAMES, AMP_JOINT_IDS
from .state import state
from booster_mjlab.tasks.goalkeeper.amp_interfaces import GoalkeeperExpertSampler


# Two-frame window of the AMP joint subset.
AMP_DIM = 2 * len(AMP_JOINT_IDS)


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


class _IterationStats:
    """Per-iteration means of per-step logs, accumulated on device.

    Writing every control step cost ~17k host syncs and ~1.6 MB of event file
    per iteration; this keeps the same tags with one point per iteration.
    Region/* values are means over all (step, env) samples of that region.
    """

    def __init__(self, env, device):
        self.env, self.device = env, device
        self.names = None

    def _init(self, source):
        self.reward_names = list(source.rewards)
        self.diag_names = list(source.diagnostics)
        self.names = (
            ["amp"] + ["reward_" + n for n in self.reward_names] + self.diag_names
        )
        self.weights = torch.tensor(
            [
                0.6 * self.env.step_dt * self.env.cfg.rewards[n].weight
                for n in self.reward_names
            ],
            device=self.device,
        )
        self.region_sum = torch.zeros(6, len(self.names), device=self.device)
        self.region_count = torch.zeros(6, device=self.device)
        self.log_sum, self.log_count = {}, {}
        self.episode = torch.zeros(3, device=self.device)

    def add_step(self, source, ids, amp, total, dones, extras):
        if self.names is None:
            self._init(source)
        dev = self.device
        rewards = torch.stack(
            [source.rewards[n].to(dev).float() for n in self.reward_names], -1
        )
        columns = [0.4 * amp[:, None], rewards * self.weights]
        if self.diag_names:
            columns.append(
                torch.stack(
                    [
                        source.diagnostics.get(n, amp * 0).to(dev).float()
                        for n in self.diag_names
                    ],
                    -1,
                )
            )
        values = torch.cat(columns, -1)
        onehot = torch.nn.functional.one_hot(ids.clamp(0, 5), 6).float()
        self.region_sum += onehot.T @ values
        self.region_count += onehot.sum(0)
        for key, value in extras.get("log", {}).items():
            v = torch.as_tensor(value, device=dev).float().mean()
            ok = torch.isfinite(v)
            self.log_sum[key] = self.log_sum.get(key, 0) + torch.where(ok, v, 0.0)
            self.log_count[key] = self.log_count.get(key, 0) + ok.float()

    def add_episodes(self, episode_return, lengths, dones):
        if self.names is None:
            return
        done = dones.bool()
        self.episode += torch.stack(
            (episode_return[done].sum(), lengths[done].sum(), done.sum().float())
        )

    def write(self, writer, step):
        if self.names is None:
            return
        region_sum = self.region_sum.cpu()
        region_count = self.region_count.cpu()
        for region in range(6):
            if region_count[region] > 0:
                means = (region_sum[region] / region_count[region]).tolist()
                for name, value in zip(self.names, means):
                    writer.add_scalar(f"Region/{region}/{name}", value, step)
        for key in self.log_sum:
            count = float(self.log_count[key])
            if count > 0:
                writer.add_scalar(key, float(self.log_sum[key]) / count, step)
        ret, length, n = self.episode.tolist()
        if n > 0:
            writer.add_scalar("Train/episode_return", ret / n, step)
            writer.add_scalar(
                "Train/episode_seconds", length / n * self.env.step_dt, step
            )


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
            amp=AMP(AMP_DIM, 0.4, device=device),
            amp_normalizer=Normalizer(AMP_DIM),
            device=device,
            **args,
        )
        self.num_steps_per_env = train_cfg["num_steps_per_env"]
        if self.num_steps_per_env < 2:
            raise ValueError("Source PPO requires at least 2 steps")
        self.alg.init_storage(
            env.num_envs, self.num_steps_per_env, [750], [92], [22], [AMP_DIM]
        )
        self.last_losses = None

    def add_git_repo_to_log(self, path):
        pass  # source revisions are pinned in vendor/source_manifest.json

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        sampler = GoalkeeperExpertSampler(self.cfg["dataset_root"], self.device)
        self.alg.motion_buffer = {
            name: SourceMotion(q[:, list(AMP_JOINT_IDS)], fps)
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
                stats = _IterationStats(self.env.unwrapped, self.device)
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
                            stats.add_step(
                                state(self.env.unwrapped),
                                ids,
                                amp,
                                total,
                                dones,
                                extras,
                            )
                        episode_return += total
                        lengths += 1
                        stats.add_episodes(episode_return, lengths, dones)
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
                    stats.write(writer, self.current_learning_iteration)
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
