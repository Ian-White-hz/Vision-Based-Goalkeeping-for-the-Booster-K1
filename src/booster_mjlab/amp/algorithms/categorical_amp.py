import math

import torch


class CategoricalAmpUpdater:
    def __init__(
        self,
        model,
        replay,
        expert_sampler,
        optimizer: torch.optim.Optimizer,
        *,
        frame_dt: float,
        gradient_penalty_coef: float = 10.0,
    ):
        if not math.isfinite(frame_dt) or frame_dt <= 0.0:
            raise ValueError("frame_dt must be finite and positive")
        if (
            not math.isfinite(gradient_penalty_coef)
            or gradient_penalty_coef < 0.0
        ):
            raise ValueError("gradient_penalty_coef must be finite and nonnegative")
        if model.device != replay.device:
            raise ValueError("model and replay must use the same device")

        self.model = model
        self.replay = replay
        self.expert_sampler = expert_sampler
        self.optimizer = optimizer
        self.frame_dt = frame_dt
        self.gradient_penalty_coef = gradient_penalty_coef

    def update_class(
        self,
        motion_id: int,
        batch_size: int,
    ) -> dict[str,float] | None:
        if not isinstance(motion_id, int) or not 0 <= motion_id < 6:
            raise ValueError("motion_id must be an integer in [0, 5]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # None 表示没有更新，不等同于 loss 为零。
        if self.replay.counts[motion_id] == 0:
            return None

        # 实际机器人数据的特征向量，形状为 (B, 44)，选择motion_id对应的类别数据。
        policy_features = self.replay.sample(
            motion_id=motion_id,
            batch_size=batch_size,
        )

        # 采样专家数据，形状为 (B, 2, 22)，然后展平为 (B, 44)。
        requested_ids = torch.full(
            (batch_size,),
            motion_id,
            dtype=torch.int64,
            device=self.model.device,
        )

        with torch.no_grad():
            expert_windows = self.expert_sampler.sample(
                requested_ids,
                frame_dt=self.frame_dt,
            )

        if expert_windows.shape != (batch_size, 2, 22):
            raise ValueError("expert windows must have shape (B, 2, 22)")
        if (
            expert_windows.dtype != torch.float32
            or expert_windows.device != self.model.device
        ):
            raise ValueError("expert windows dtype/device mismatch")

        expert_features = expert_windows.flatten(start_dim=1)

        # 训练判别器
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True) # 清空梯度，避免累积
        # 选择对应类别的判别器
        discriminator = self.model.discriminators[motion_id]
        # 更新判别器的归一化参数，确保专家和策略特征在同一尺度上
        discriminator.update_normalization(
            expert_features,
            policy_features,
        )
        # 计算专家和策略特征的判别器输出
        expert_scores = discriminator(expert_features)
        policy_scores = discriminator(policy_features)
        # 计算分类损失和梯度惩罚（gradient penalty属于正则化项）
        classification_loss, gradient_penalty = discriminator.compute_loss(
            policy_d=policy_scores,
            expert_d=expert_scores,
            sample_amp_expert=expert_features,
            sample_amp_policy=policy_features,
            lambda_=self.gradient_penalty_coef
        )
        # 计算总损失
        total_loss = classification_loss + gradient_penalty

        if not torch.isfinite(total_loss).item():
            raise FloatingPointError("Non-finite discriminator loss")
        # 计算梯度并更新判别器参数
        total_loss.backward()

        for parameter in discriminator.parameters():
            if (
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
            ):
                raise FloatingPointError("Non-finite discriminator gradient")
        # 更新判别器参数
        self.optimizer.step()

        return {
            "classification_loss": classification_loss.detach().item(),
            "gradient_penalty": gradient_penalty.detach().item(),
            "total_loss": total_loss.detach().item(),
            "expert_score": expert_scores.detach().mean().item(),
            "policy_score": policy_scores.detach().mean().item(),
        }

    def update_all(
        self,
        batch_size: int
    ) -> dict[int, dict[str, float]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        results = {}

        for motion_id in range(6):
            metrics = self.update_class(
                motion_id=motion_id,
                batch_size=batch_size,
            )
            if metrics is not None:
                results[motion_id] = metrics

        return results

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "frame_dt": self.frame_dt,
            "gradient_penalty_coef": self.gradient_penalty_coef,
        }

    def load_state_dict(
        self, 
        state: dict,
        *,
        load_optimizer: bool = True,
    ) -> None:
        if state["schema_version"] != 1:
            raise ValueError("Unsupported schema version")

        if not math.isclose(
            state["frame_dt"],
            self.frame_dt,
            rel_tol=1e-7,
        ):
            raise ValueError("Checkpoint frame_dt does not match current setup")

        if load_optimizer and not math.isclose(
            state["gradient_penalty_coef"],
            self.gradient_penalty_coef,
            rel_tol=1e-7,
        ):
            raise ValueError("Gradient penalty coefficient changed during resume")

        self.model.load_state_dict(state["model"], strict=True)

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
