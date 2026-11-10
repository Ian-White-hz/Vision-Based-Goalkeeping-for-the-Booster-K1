# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL PPO with source Goalkeeper's joint, minibatch-local AMP update.

The PPO formulas follow the installed RSL-RL PPO. HIM estimation heads and
smoothness losses are not added here: this module ports the AMP part only.
"""

import math
from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from torch import nn

from booster_mjlab.amp.algorithms.categorical_amp import CategoricalAmpUpdater


AMP_FEATURES = "_amp_transition"
AMP_IDS = "_amp_motion_id"
AMP_VALID = "_amp_valid"
MOTION_NAMES = (
    "lefthand",
    "righthand",
    "leftjump",
    "rightjump",
    "leftstep",
    "rightstep",
)


class CategoricalAmpPPO(PPO):
    def configure_amp(
        self,
        model,
        *,
        frame_dt: float,
        gradient_penalty_coef: float = 0.5,
        trunk_weight_decay: float = 1.0e-3,
        head_weight_decay: float = 0.1,
    ) -> None:
        """Attach AMP once, before collection, and build one joint Adam."""
        if hasattr(self, "amp_model") or self.optimizer.state or self.storage.step:
            raise RuntimeError("Configure AMP exactly once, before training")
        if (
            self.is_multi_gpu
            or self.rnd
            or self.symmetry
            or self.actor.is_recurrent
            or self.critic.is_recurrent
            or self.use_mixed_precision
        ):
            raise NotImplementedError(
                "Source AMP supports feedforward, full precision, one-device PPO "
                "without RND or symmetry"
            )
        if type(self.optimizer) is not torch.optim.Adam:
            raise ValueError("Source Goalkeeper AMP requires Adam")
        if model.device != torch.device(self.device):
            raise ValueError("AMP and PPO must use the same device")
        for decay in (trunk_weight_decay, head_weight_decay):
            if not math.isfinite(decay) or decay < 0.0:
                raise ValueError("Weight decay must be finite and nonnegative")

        self.amp_model = model
        self.amp_updater = CategoricalAmpUpdater(
            model,
            None,
            frame_dt=frame_dt,
            gradient_penalty_coef=gradient_penalty_coef,
        )
        groups = [
            {
                "params": list(
                    chain(self.actor.parameters(), self.critic.parameters())
                ),
                "name": "actor_critic",
            }
        ]
        for name, discriminator in zip(MOTION_NAMES, model.discriminators):
            groups.extend(
                [
                    {
                        "params": list(discriminator.trunk.parameters()),
                        "weight_decay": trunk_weight_decay,
                        "name": f"amp_trunk_{name}",
                    },
                    {
                        "params": list(discriminator.linear.parameters()),
                        "weight_decay": head_weight_decay,
                        "name": f"amp_head_{name}",
                    },
                ]
            )
        self.optimizer = torch.optim.Adam(groups, lr=self.learning_rate)

        # Store metadata in the same TensorDict as PPO observations. RSL-RL's
        # generator then uses exactly the same permutation for all fields.
        # These keys are NOT members of the actor/critic observation groups.
        shape = (self.storage.num_transitions_per_env, self.storage.num_envs)
        self.storage.observations.set(
            AMP_FEATURES, torch.zeros(*shape, 44, device=self.device)
        )
        self.storage.observations.set(
            AMP_IDS, torch.zeros(shape, dtype=torch.int64, device=self.device)
        )
        self.storage.observations.set(
            AMP_VALID, torch.zeros(shape, dtype=torch.bool, device=self.device)
        )
        self._amp_transition = None
        self.last_amp_metrics = {}
        self.last_ppo_diagnostics = {}

    def record_amp_transition(self, features, motion_ids, valid_mask) -> None:
        if self._amp_transition is not None:
            raise RuntimeError("Previous AMP transition has not been consumed")
        self.amp_model._validate_inputs(features, motion_ids)
        if (
            features.shape[0] != self.storage.num_envs
            or valid_mask.shape != motion_ids.shape
            or valid_mask.dtype != torch.bool
            or valid_mask.device != features.device
        ):
            raise ValueError("Invalid AMP transition shape/dtype/device")
        self._amp_transition = (
            features.detach(),
            motion_ids.detach(),
            valid_mask.detach(),
        )

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self._amp_transition is None:
            raise RuntimeError("Record the AMP transition before process_env_step")
        features, motion_ids, valid_mask = self._amp_transition
        transition_obs = self.transition.observations.clone(recurse=False)
        transition_obs.set(AMP_FEATURES, features)
        transition_obs.set(AMP_IDS, motion_ids)
        transition_obs.set(AMP_VALID, valid_mask)
        self.transition.observations = transition_obs
        super().process_env_step(obs, rewards, dones, extras)
        self._amp_transition = None

    def update(self) -> dict[str, float]:
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("Call PPO update outside inference/no_grad mode")
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("Collect a complete rollout before update")
        if self.amp_updater.expert_sampler is None:
            raise RuntimeError("Bind an expert sampler before updating PPO + AMP")
        if self.num_learning_epochs <= 0 or self.num_mini_batches <= 0:
            raise ValueError("PPO epoch and minibatch counts must be positive")
        size = self.storage.num_envs * self.storage.num_transitions_per_env
        if size // self.num_mini_batches < 2:
            raise ValueError("Each PPO minibatch must contain at least two transitions")

        totals = dict(value=0.0, surrogate=0.0, entropy=0.0, amp=0.0)
        diagnostics = []
        class_sums = {}
        class_updates = {}
        self.amp_model.train()
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )
            self.actor(batch.observations, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy

            # Measure old-policy -> current-policy KL even for fixed LR runs.
            with torch.no_grad():
                kl = self.actor.get_kl_divergence(
                    batch.old_distribution_params,
                    self.actor.output_distribution_params,
                ).mean().item()
            if self.desired_kl is not None and self.schedule == "adaptive":
                if kl > self.desired_kl * 2.0:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate

            ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
            clip_fraction = (
                (ratio.detach() - 1.0).abs() > self.clip_param
            ).float().mean().item()
            advantages = batch.advantages.squeeze(-1)
            surrogate = -advantages * ratio
            clipped = -advantages * ratio.clamp(
                1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, clipped).mean()
            if self.use_clipped_value_loss:
                clipped_values = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (values - batch.returns).square(),
                    (clipped_values - batch.returns).square(),
                ).mean()
            else:
                value_loss = (batch.returns - values).square().mean()

            amp_loss, metrics = self.amp_updater.compute_loss(
                batch.observations[AMP_FEATURES],
                batch.observations[AMP_IDS],
                batch.observations[AMP_VALID],
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + amp_loss
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite PPO + AMP loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            def gradient_norm(module):
                norms = [p.grad.detach().norm() for p in module.parameters()
                         if p.grad is not None]
                return torch.stack(norms).norm().item() if norms else 0.0

            actor_norm = gradient_norm(self.actor)
            critic_norm = gradient_norm(self.critic)
            # Source clips the combined actor-critic, not discriminator grads.
            combined_norm = nn.utils.clip_grad_norm_(
                list(chain(self.actor.parameters(), self.critic.parameters())),
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            for parameter in self.amp_model.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError("Non-finite discriminator gradient")
            diagnostics.append({
                "kl": kl,
                "clip_fraction": clip_fraction,
                "actor_grad_norm_pre_clip": actor_norm,
                "critic_grad_norm_pre_clip": critic_norm,
                "combined_grad_norm_pre_clip": combined_norm.item(),
                "grad_clip_scale": min(
                    1.0, self.max_grad_norm / (combined_norm.item() + 1.0e-6)
                ),
                "learning_rate": self.learning_rate,
            })
            self.optimizer.step()

            for name, value in (
                ("value", value_loss),
                ("surrogate", surrogate_loss),
                ("entropy", entropy.mean()),
                ("amp", amp_loss),
            ):
                totals[name] += value.detach().item()
            for motion_id, values_by_name in metrics.items():
                target = class_sums.setdefault(
                    motion_id, dict.fromkeys(values_by_name, 0.0)
                )
                class_updates[motion_id] = class_updates.get(motion_id, 0) + 1
                for name, value in values_by_name.items():
                    target[name] += value

        observations = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        updates = self.num_learning_epochs * self.num_mini_batches
        self.last_amp_metrics = {
            motion_id: {
                name: value / class_updates[motion_id] for name, value in values.items()
            }
            for motion_id, values in class_sums.items()
        }
        self.last_ppo_diagnostics = {
            name: sum(row[name] for row in diagnostics) / len(diagnostics)
            for name in diagnostics[0]
        }
        self.last_ppo_diagnostics.update(
            kl_first=diagnostics[0]["kl"],
            kl_max=max(row["kl"] for row in diagnostics),
            clip_fraction_max=max(row["clip_fraction"] for row in diagnostics),
        )
        self.storage.clear()
        return {name: value / updates for name, value in totals.items()}

    def save(self) -> dict:
        saved = super().save()
        saved["categorical_amp"] = {
            "schema_version": 2,
            "model": self.amp_model.state_dict(),
            "frame_dt": self.amp_updater.frame_dt,
            "gradient_penalty_coef": self.amp_updater.gradient_penalty_coef,
        }
        return saved

    def load(
        self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True
    ) -> bool:
        load_amp = load_cfg is None or load_cfg.get("amp", False)
        load_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
        if load_optimizer and not load_amp:
            raise ValueError("Resuming the joint optimizer also requires AMP state")
        if load_amp:
            state = loaded_dict.get("categorical_amp")
            if state is None or state.get("schema_version") != 2:
                raise ValueError(
                    "Expected a source-style categorical AMP checkpoint (v2)"
                )
            for name in ("frame_dt", "gradient_penalty_coef"):
                if not math.isclose(
                    state[name], getattr(self.amp_updater, name), rel_tol=1e-7
                ):
                    raise ValueError(
                        f"Checkpoint {name} does not match current AMP setup"
                    )
            self.amp_model.load_state_dict(state["model"], strict=strict)
        return super().load(loaded_dict, load_cfg, strict)
