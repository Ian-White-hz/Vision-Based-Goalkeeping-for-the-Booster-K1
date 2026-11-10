import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from rsl_rl.utils.log_writer import LogWriter

from booster_mjlab.amp.algorithms.categorical_ppo import (
    MOTION_NAMES,
    CategoricalAmpPPO,
)
from booster_mjlab.amp.modules.categorical_collector import CategoricalAmpCollector
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)
from booster_mjlab.amp.runners.categorical_iteration import (
    run_categorical_amp_iteration,
)
from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER

from booster_mjlab.amp.utils.interface_loading import (
    resolve_amp_callable,
)


class CategoricalAmpOnPolicyRunner(BoosterOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        if self.is_distributed:
            raise NotImplementedError("Categorical AMP supports one device")
        if not isinstance(self.alg, CategoricalAmpPPO):
            raise TypeError("Use CategoricalAmpPpoAlgorithmCfg with this runner")
        self.amp_cfg = self.cfg["amp"]
        weight = self.amp_cfg["style_weight"]
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("style_weight must be finite and in [0, 1]")
        obs = self.env.get_observations().to(self.device)
        if "amp" not in obs:
            raise ValueError("Environment must provide obs['amp']")
        self.amp_model = CategoricalAmpDiscriminator(
            device=self.device,
            empirical_normalization=self.amp_cfg["empirical_normalization"],
            hidden_layer_sizes=tuple(self.amp_cfg["hidden_layer_sizes"]),
            logit_init_scale=self.amp_cfg["logit_init_scale"],
            reward_mode=self.amp_cfg.get("reward_mode", "source_noise"),
        )
        self.alg.configure_amp(
            self.amp_model,
            frame_dt=self.env.unwrapped.step_dt,
            gradient_penalty_coef=self.amp_cfg["gradient_penalty_coef"],
            trunk_weight_decay=self.amp_cfg["trunk_weight_decay"],
            head_weight_decay=self.amp_cfg["head_weight_decay"],
        )
        self.amp_updater = self.alg.amp_updater
        self.amp_collector = CategoricalAmpCollector(self.amp_model, obs["amp"])
        self._read_motion_ids = None
        self._last_iteration_summary = None
        # Completed updates, cumulative transitions and time, independent of logging.
        self.total_timesteps = 0
        self.total_time = 0.0
        self.add_git_repo_to_log(__file__)

    def bind_amp_interfaces(self, expert_sampler, read_motion_ids) -> None:
        if not callable(getattr(expert_sampler, "sample", None)):
            raise TypeError("expert_sampler must provide a callable sample()")
        if not callable(read_motion_ids):
            raise TypeError("read_motion_ids must be callable")
        names = getattr(expert_sampler, "joint_names", None)
        if names is None:
            raise ValueError("expert_sampler must expose joint_names")
        if tuple(names) != tuple(K1_JOINT_ORDER):
            raise ValueError("Expert joint order must match K1_JOINT_ORDER")
        self.amp_updater.expert_sampler = expert_sampler
        self._read_motion_ids = read_motion_ids

    def learn(self, num_learning_iterations: int, init_at_random_ep_len=False) -> None:
        if (
            not isinstance(num_learning_iterations, int)
            or isinstance(num_learning_iterations, bool)
            or num_learning_iterations <= 0
        ):
            raise ValueError("num_learning_iterations must be a positive integer")
        interval = self.cfg["save_interval"]
        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            raise ValueError("save_interval must be a positive integer")
        if self.alg.storage.step or self.alg._amp_transition is not None:
            raise RuntimeError("Start learn() at an empty rollout boundary")

        self._ensure_amp_interfaces()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # Get the observations
        obs = self.env.get_observations().to(self.device)
        self.amp_collector = CategoricalAmpCollector(self.amp_model, obs["amp"])
        target_iteration = self.current_learning_iteration + num_learning_iterations
        last_saved_iteration = None
        try:
            self.logger.init_logging_writer()
            for _ in range(num_learning_iterations):
                # Run one iteration of collection and learning, updating the AMP model and PPO policy.
                obs, summary = run_categorical_amp_iteration(
                    self.env,
                    self.alg,
                    self.amp_collector,
                    obs,
                    self._read_motion_ids,
                    num_steps=self.cfg["num_steps_per_env"],
                    style_weight=self.amp_cfg["style_weight"],
                    on_env_step=self.logger.process_env_step,
                )
                self._last_iteration_summary = summary
                self.current_learning_iteration += 1
                self.total_timesteps += (
                    self.cfg["num_steps_per_env"] * self.env.num_envs
                )
                self.total_time += sum(summary["timing"].values())
                self._log_iteration(summary, target_iteration)
                if (
                    self.logger.log_dir is not None
                    and not self.logger.disable_logs
                    and self.current_learning_iteration % self.cfg["save_interval"] == 0
                ):
                    self._save_iteration_checkpoint()
                    last_saved_iteration = self.current_learning_iteration

            if (
                self.logger.log_dir is not None
                and not self.logger.disable_logs
                and last_saved_iteration != self.current_learning_iteration
            ):
                self._save_iteration_checkpoint()
        finally:
            self._close_logging_writer()

    def _log_iteration(self, summary: dict, target_iteration: int) -> None:
        # Logger expects zero-based indices; our counter counts COMPLETED updates.
        it = self.current_learning_iteration - 1
        transitions = self.cfg["num_steps_per_env"] * self.env.num_envs
        self.logger.tot_timesteps = self.total_timesteps - transitions
        self.logger.tot_time = self.total_time - sum(summary["timing"].values())
        self.logger.log(
            it=it,
            start_it=0,  # cumulative counters make the cumulative ETA meaningful
            total_it=target_iteration,
            collect_time=summary["timing"]["collection"],
            learn_time=summary["timing"]["learning"],
            loss_dict=summary["ppo"],
            learning_rate=self.alg.learning_rate,
            action_std=self.alg.get_policy().output_std,
            rnd_weight=None,
        )
        # Keep counters correct even when log_dir=None and Logger.log() is a no-op.
        self.logger.tot_timesteps = self.total_timesteps
        self.logger.tot_time = self.total_time
        writer = self.logger.writer
        if writer is None:
            return
        for name, value in self.alg.last_ppo_diagnostics.items():
            writer.add_scalar(f"PPO/{name}", value, it)
        for name, value in summary["reward_means"].items():
            writer.add_scalar(f"AMP/step_reward/{name}", value, it)
        writer.add_scalar("AMP/valid_fraction", summary["amp_valid_fraction"], it)
        writer.add_scalar(
            "Train/completed_iterations", self.current_learning_iteration, it
        )
        writer.add_scalar("Perf/total_timesteps", self.total_timesteps, it)
        if self.amp_model.amp_normalizer is not None:
            writer.add_scalar(
                "AMP/normalizer_count", self.amp_model.amp_normalizer.count.item(), it
            )
        for motion_id, metrics in summary["discriminator"].items():
            for name, value in metrics.items():
                writer.add_scalar(
                    f"Discriminator/{MOTION_NAMES[motion_id]}/{name}", value, it
                )

    def _close_logging_writer(self) -> None:
        writer = self.logger.writer
        if writer is None:
            return
        try:
            if isinstance(writer, LogWriter):
                self.logger.stop_logging_writer()
            else:
                try:
                    writer.flush()
                finally:
                    writer.close()
        finally:
            self.logger.writer = None

    def _save_iteration_checkpoint(self) -> None:
        self.save(
            str(
                Path(self.logger.log_dir)
                / f"model_{self.current_learning_iteration}.pt"
            )
        )

    def _runner_signature(self) -> dict:
        return {
            "joint_names": list(K1_JOINT_ORDER),
            "motion_names": list(MOTION_NAMES),
            "style_weight": self.amp_cfg["style_weight"],
            "reward_version": self.amp_model.reward_version,
            "normalizer_update": "normalized_policy_then_expert",
        }

    def save(self, path: str, infos: dict | None = None) -> None:
        """Atomically save learning state at an iteration boundary.

        Unlike the velocity runner, this does not export ONNX on every save.
        Explicit export_policy_to_onnx() remains available for deployment.
        """
        if self.alg.storage.step or self.alg._amp_transition is not None:
            raise RuntimeError("Save checkpoints only at an empty rollout boundary")
        runner_state = {
            "schema_version": 1,
            "iteration_semantics": "completed_updates",
            "total_timesteps": self.total_timesteps,
            "total_time": self.total_time,
            **self._runner_signature(),
        }
        saved_infos = {
            **(infos or {}),
            "env_state": {
                "common_step_counter": int(self.env.unwrapped.common_step_counter)
            },
            "categorical_amp_runner": runner_state,
        }
        payload = self.alg.save()
        payload.update(iter=self.current_learning_iteration, infos=saved_infos)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        if self.cfg.get("upload_model", False):
            self.logger.save_model(str(target), self.current_learning_iteration)

    def load(self, path: str, load_cfg=None, strict=True, map_location=None) -> dict:
        """Resume full learning state, or warm-start Actor/Critic in a fresh runner.

        Environment poses, RNG, sampler state and rollout/History are not restored.
        Only resume with iteration=True restores the environment's common counter.
        """
        if self.alg.storage.step or self.alg._amp_transition is not None:
            raise RuntimeError("Load checkpoints only at an empty rollout boundary")
        warm_start = load_cfg is None and self.cfg.get("warm_start", False)
        if warm_start:
            if self.current_learning_iteration or self.alg.optimizer.state:
                raise RuntimeError("Warm start requires a fresh runner and optimizer")
            load_cfg = {"actor": True, "critic": True}
        restore_iteration = load_cfg is None or load_cfg.get("iteration", False)
        payload = torch.load(
            path, map_location=map_location or self.device, weights_only=True
        )
        infos = payload.get("infos") or {}
        if not isinstance(infos, dict):
            raise ValueError("Checkpoint infos must be a dictionary")
        state = infos.get("categorical_amp_runner")
        load_amp = load_cfg is None or load_cfg.get("amp", False)
        if state is not None and (restore_iteration or load_amp):
            self._validate_runner_state(state)
        if restore_iteration:
            if state is None:
                raise ValueError("Full resume requires categorical AMP runner state")
            iteration = payload.get("iter")
            if not isinstance(iteration, int) or iteration < 0:
                raise ValueError("Checkpoint must contain a nonnegative iteration")
            counter = infos.get("env_state", {}).get("common_step_counter")
            if not isinstance(counter, int) or counter < 0:
                raise ValueError("Checkpoint must contain a nonnegative common counter")
        load_iteration = self.alg.load(payload, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = payload["iter"]
            self.total_timesteps = state["total_timesteps"]
            self.total_time = state["total_time"]
            self.env.unwrapped.common_step_counter = counter
        self.logger.tot_timesteps = self.total_timesteps
        self.logger.tot_time = self.total_time
        # New physical episodes must not inherit old partial episode returns.
        self.logger.ep_extras.clear()
        self.logger.rewbuffer.clear()
        self.logger.lenbuffer.clear()
        self.logger.cur_reward_sum.zero_()
        self.logger.cur_episode_length.zero_()
        obs = self.env.get_observations().to(self.device)
        self.amp_collector = CategoricalAmpCollector(self.amp_model, obs["amp"])
        self._last_iteration_summary = None
        return infos

    def _validate_runner_state(self, state: dict) -> None:
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != 1
            or state.get("iteration_semantics") != "completed_updates"
        ):
            raise ValueError("Unsupported categorical AMP runner checkpoint format")
        for name, expected in self._runner_signature().items():
            actual = state.get(name)
            if name == "style_weight":
                matches = isinstance(actual, (int, float)) and math.isclose(
                    actual, expected, rel_tol=1e-7, abs_tol=1e-9
                )
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"Checkpoint {name} does not match current AMP setup")
        if (
            not isinstance(state.get("total_timesteps"), int)
            or state["total_timesteps"] < 0
            or not isinstance(state.get("total_time"), (int, float))
            or not math.isfinite(state["total_time"])
            or state["total_time"] < 0.0
        ):
            raise ValueError("Invalid cumulative checkpoint counters")

    def _ensure_amp_interfaces(self) -> None:
        """Lazily bind configured adapters only when learn() is invoked.

        Explicit manual binding wins. Resolve both entries before constructing
        the sampler so a broken reader path does not trigger dataset loading.
        """
        sampler = self.amp_updater.expert_sampler
        reader = self._read_motion_ids

        if sampler is not None and reader is not None:
            return

        if sampler is not None or reader is not None:
            raise RuntimeError("Incomplete AMP binding: bind both sampler and reader")

        factory_path = self.amp_cfg.get("expert_sampler_factory")
        reader_path = self.amp_cfg.get("motion_id_reader")

        if factory_path is None or reader_path is None:
            raise RuntimeError(
                "Bind AMP interfaces manually, or configure both "
                "expert_sampler_factory and motion_id_reader"
            )

        factory = resolve_amp_callable(
            factory_path, field_name="expert_sampler_factory"
        )
        reader = resolve_amp_callable(reader_path, field_name="motion_id_reader")
        sampler = factory(
            dataset_root=self.amp_cfg["dataset_root"],
            device=torch.device(self.device),
        )

        self.bind_amp_interfaces(
            expert_sampler=sampler,
            read_motion_ids=reader,
        )
