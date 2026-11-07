"""Source-style AMP loss for the same minibatch used by PPO.

No replay sampling or independent optimizer step is performed here.
"""

import math

import torch


class CategoricalAmpUpdater:
    def __init__(
        self,
        model,
        expert_sampler,
        *,
        frame_dt: float,
        gradient_penalty_coef: float = 0.5,
    ):
        if not math.isfinite(frame_dt) or frame_dt <= 0.0:
            raise ValueError("frame_dt must be finite and positive")
        if not math.isfinite(gradient_penalty_coef) or gradient_penalty_coef < 0.0:
            raise ValueError("gradient_penalty_coef must be finite and nonnegative")
        self.model = model
        self.expert_sampler = expert_sampler
        self.frame_dt = frame_dt
        self.gradient_penalty_coef = gradient_penalty_coef

    def compute_loss(
        self,
        features: torch.Tensor,
        motion_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, dict[str, float]]]:
        """Sum class-mean losses and update shared moments once per minibatch.

        Both batches use pre-update statistics. Updating with normalized policy
        followed by normalized expert reproduces the source's unusual behavior.
        """
        self.model._validate_inputs(features, motion_ids)
        if (
            valid_mask.shape != motion_ids.shape
            or valid_mask.dtype != torch.bool
            or valid_mask.device != features.device
        ):
            raise ValueError("valid_mask must be bool (B,) on model device")
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("Compute AMP loss outside inference/no_grad mode")
        if not valid_mask.any().item():
            return features.new_zeros(()), {}
        if self.expert_sampler is None:
            raise RuntimeError("Bind an expert sampler before updating AMP")

        features = features[valid_mask].detach()
        motion_ids = motion_ids[valid_mask]
        expert_features = torch.empty_like(features)
        for motion_id in range(6):
            mask = motion_ids == motion_id
            if not mask.any().item():
                continue
            with torch.no_grad():
                windows = self.expert_sampler.sample(
                    motion_ids[mask], frame_dt=self.frame_dt
                )
            if (
                windows.shape != (int(mask.sum().item()), 2, 22)
                or windows.dtype != torch.float32
                or windows.device != features.device
            ):
                raise ValueError(
                    "expert windows must be float32 (B, 2, 22) on model device"
                )
            expert_features[mask] = windows.flatten(start_dim=1).detach()

        # Whole-minibatch normalization precedes routing. Sub-Ds do not normalize.
        policy_normalized = self.model.normalize(features)
        expert_normalized = self.model.normalize(expert_features)
        total = features.new_zeros(())
        metrics = {}
        for motion_id, discriminator in enumerate(self.model.discriminators):
            mask = motion_ids == motion_id
            if not mask.any().item():
                continue
            policy = policy_normalized[mask]
            expert = expert_normalized[mask]
            policy_scores = discriminator(policy)
            expert_scores = discriminator(expert)
            classification, penalty = discriminator.compute_loss(
                policy_d=policy_scores,
                expert_d=expert_scores,
                sample_amp_expert=expert,
                sample_amp_policy=policy,
                lambda_=self.gradient_penalty_coef,
            )
            class_loss = classification + penalty
            total = total + class_loss
            metrics[motion_id] = {
                "classification_loss": classification.detach().item(),
                "gradient_penalty": penalty.detach().item(),
                "total_loss": class_loss.detach().item(),
                "expert_score": expert_scores.detach().mean().item(),
                "policy_score": policy_scores.detach().mean().item(),
                "sample_count": int(mask.sum().item()),
            }
        if not torch.isfinite(total).item():
            raise FloatingPointError("Non-finite AMP loss")
        # Same ordering as HIMPPO: after constructing loss, before backward.
        self.model.update_normalization(policy_normalized, expert_normalized)
        return total, metrics
