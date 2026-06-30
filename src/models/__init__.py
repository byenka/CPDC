"""
src.models package initializer (CoWePS v2.4)
Mengkonsolidasikan ekspor dari trainer module dan model factory,
dengan fallback nama file trainer: trainer_base.py ↔ trainerbase.py
"""

# Model factory
from .model_factory import ModelFactory, create_model_from_config

# Trainer: coba import dari trainer_base.py, fallback ke trainerbase.py
try:
    from .trainer_base import (
        SafeTrainerMixins,
        StratifiedBatchSampler,
        TrainingMonitor,
        calculate_safe_class_weights,
        FundusDataset,
        get_train_transforms,
        get_val_transforms,
        TwoStageTrainer,
        dynamic_sampler_params,
    )
except ModuleNotFoundError:
    from .trainerbase import (
        SafeTrainerMixins,
        StratifiedBatchSampler,
        TrainingMonitor,
        calculate_safe_class_weights,
        FundusDataset,
        get_train_transforms,
        get_val_transforms,
        TwoStageTrainer,
        dynamic_sampler_params,
    )

__all__ = [
    "ModelFactory", "create_model_from_config",
    "SafeTrainerMixins", "StratifiedBatchSampler", "TrainingMonitor",
    "calculate_safe_class_weights", "FundusDataset",
    "get_train_transforms", "get_val_transforms",
    "TwoStageTrainer", "dynamic_sampler_params",
]
