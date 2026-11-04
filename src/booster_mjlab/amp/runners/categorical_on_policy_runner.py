import math

import torch

from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.amp.algorithms.categorical_amp import (
    CategoricalAmpUpdater,
)
from booster_mjlab.amp.modules.categorical_collector import (
    CategoricalAmpCollector,
)
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)
from booster_mjlab.amp.modules.categorical_replay import (
    CategoricalAmpReplayBuffer,
)


class CategoricalAmpOnPolicyRunner(BoosterOnPolicyRunner):
    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        super().__init__(env, train_cfg, log_dir, device)

        if self.is_distributed:
            raise NotImplementedError("Categorical AMP supports one device")

        self.amp_cfg = self.cfg["amp"]

        style_weight = self.amp_cfg["style_weight"]
        if (
            not math.isfinite(style_weight)
            or not 0.0 <= style_weight <= 1.0
        ):
            raise ValueError("style_weight must be finite and in [0, 1]")

        if self.amp_cfg["batch_size"] <= 0:
            raise ValueError("AMP batch_size must be positive")

        amp_device = torch.device(self.device)
        obs = self.env.get_observations()

        if "amp" not in obs:
            raise ValueError("Environment must provide obs['amp']")

        initial_q = obs["amp"].to(amp_device)

        self.amp_model = CategoricalAmpDiscriminator(
            device=amp_device,
            empirical_normalization=self.amp_cfg["empirical_normalization"],
        )

        self.amp_replay = CategoricalAmpReplayBuffer(
            capacity_per_class=self.amp_cfg["capacity_per_class"],
            device=amp_device,
        )

        self.amp_optimizer = torch.optim.Adam(
            self.amp_model.parameters(),
            lr=self.amp_cfg["learning_rate"],
        )

        self.amp_collector = CategoricalAmpCollector(
            self.amp_model,
            self.amp_replay,
            initial_q,
        )

        self.amp_updater = CategoricalAmpUpdater(
            self.amp_model,
            self.amp_replay,
            expert_sampler=None,
            optimizer=self.amp_optimizer,
            frame_dt=self.env.unwrapped.step_dt,
            gradient_penalty_coef=self.amp_cfg["gradient_penalty_coef"],
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        raise NotImplementedError(
            "Categorical AMP learn loop is not connected yet"
        )