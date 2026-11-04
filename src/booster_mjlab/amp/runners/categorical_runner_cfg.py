from dataclasses import dataclass, field

from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg


@dataclass
class CategoricalAmpCfg:
    style_weight: float = 0.4
    capacity_per_class: int = 10_000
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    gradient_penalty_coef: float = 10.0
    empirical_normalization: bool = True

    dataset_root: str = "artifacts/goalkeeper_k1/serial"

    # Part B 提供实际入口后填写，不预设不存在的模块路径。
    expert_sampler_factory: str | None = None
    motion_id_reader: str | None = None

@dataclass
class CategoricalAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "CategoricalAmpOnPolicyRunner"

    amp: CategoricalAmpCfg = field(
        default_factory=CategoricalAmpCfg,
    )

    warm_start: bool = False