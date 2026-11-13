from dataclasses import dataclass, field
from typing import Literal

from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class CategoricalAmpCfg:
    style_weight: float = 0.4
    # Independent AMP ablation multiplier; does not change task reward weight.
    reward_scale: float = 1.0
    reward_mode: Literal["source_noise", "direct"] = "source_noise"
    """Source-compatible noisy reward, or score the actual window without noise."""
    gradient_penalty_coef: float = 0.5
    empirical_normalization: bool = True
    hidden_layer_sizes: tuple[int, ...] = (512, 256)
    logit_init_scale: float = 1.0
    trunk_weight_decay: float = 1.0e-3
    head_weight_decay: float = 0.1
    dataset_root: str = "artifacts/goalkeeper_k1/serial"

    expert_sampler_factory: str | None = None
    """Trusted module:object called as factory(dataset_root=..., device=...)."""
    motion_id_reader: str | None = None
    """Trusted module:object called as reader(env.unwrapped) -> int64 (N,).

    Set both entries for lazy binding on learn(), or bind both objects manually.
    Actor-only load and inference do not resolve either entry.
    """


@dataclass
class CategoricalAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "booster_mjlab.amp.algorithms.categorical_ppo:CategoricalAmpPPO"


@dataclass
class PeriodicEvaluationCfg:
    interval: int = 0
    """Completed PPO updates between evaluations; zero disables evaluation."""
    evaluator_factory: str | None = None
    validation_seeds: tuple[int, ...] = tuple(range(100, 120))
    holdout_seeds: tuple[int, ...] = tuple(range(10000, 10020))
    num_envs: int = 60
    max_seconds: float = 6.0
    evaluate_at_start: bool = True
    evaluate_at_end: bool = True


@dataclass
class CategoricalAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "CategoricalAmpOnPolicyRunner"
    load_checkpoint: str = r"model_[0-9]+\.pt$"
    """Automatic resume selects the latest numbered file, excluding model_best.pt."""
    algorithm: CategoricalAmpPpoAlgorithmCfg = field(
        default_factory=CategoricalAmpPpoAlgorithmCfg
    )
    amp: CategoricalAmpCfg = field(default_factory=CategoricalAmpCfg)
    evaluation: PeriodicEvaluationCfg = field(default_factory=PeriodicEvaluationCfg)
    warm_start: bool = False
    """Load Actor/Critic only into a fresh runner; keep a fresh AMP and Adam."""
