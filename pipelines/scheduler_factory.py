"""スケジューラファクトリモジュール

PipelineManager / ControlNetPipelineManager / IPAdapterPipelineManager で
共有されるスケジューラ生成ロジック
"""
from diffusers import (
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    HeunDiscreteScheduler,
)

from config import SCHEDULERS


def create_scheduler(scheduler_name: str, config=None):
    """スケジューラインスタンスを作成"""
    scheduler_value = SCHEDULERS.get(scheduler_name, "dpm_2m_karras")

    # 共通設定
    common_config = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
    }

    if config is not None:
        # 既存のパイプラインから設定を継承
        common_config = {
            "beta_start": getattr(config, 'beta_start', 0.00085),
            "beta_end": getattr(config, 'beta_end', 0.012),
            "beta_schedule": getattr(config, 'beta_schedule', "scaled_linear"),
            "num_train_timesteps": getattr(config, 'num_train_timesteps', 1000),
        }

    if scheduler_value == "ddim":
        return DDIMScheduler(
            **common_config,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1
        )
    elif scheduler_value == "euler":
        return EulerDiscreteScheduler(**common_config)
    elif scheduler_value == "euler_a":
        return EulerAncestralDiscreteScheduler(**common_config)
    elif scheduler_value == "dpm_2m":
        return DPMSolverMultistepScheduler(
            **common_config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=False
        )
    elif scheduler_value == "dpm_2m_karras":
        return DPMSolverMultistepScheduler(
            **common_config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
    elif scheduler_value == "dpm_sde":
        return DPMSolverSinglestepScheduler(
            **common_config,
            use_karras_sigmas=False
        )
    elif scheduler_value == "dpm_sde_karras":
        return DPMSolverSinglestepScheduler(
            **common_config,
            use_karras_sigmas=True
        )
    elif scheduler_value == "unipc":
        return UniPCMultistepScheduler(**common_config)
    elif scheduler_value == "lms":
        return LMSDiscreteScheduler(**common_config)
    elif scheduler_value == "pndm":
        return PNDMScheduler(**common_config)
    elif scheduler_value == "heun":
        return HeunDiscreteScheduler(**common_config)
    else:
        # デフォルトはDPM++ 2M Karras
        return DPMSolverMultistepScheduler(
            **common_config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
