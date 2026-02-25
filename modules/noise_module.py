import torch
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional, Union, Tuple, Annotated
import torch
import random
import numpy as np


class NoiseModulePlan(BaseModel):
    """Параметры шума для батча."""

    additive_scale: torch.Tensor  # Масштаб Стьюдента (B, dim)
    multiplicative_shape: torch.Tensor  # Параметр Вейбулла (B, dim)


class NoisePrior(BaseModel):
    student_df_range: Tuple[float, float]
    weibull_scale_range: Tuple[float, float]


class NoiseModule:
    def __init__(self, device: str):
        self.device = device

    def execute(self, x: torch.Tensor, plan: NoiseModulePlan) -> torch.Tensor:
        # Генерируем стандартный нормальный шум (B, T, dim)
        noise = torch.randn_like(x)
        # Масштабируем его согласно плану и добавляем к сигналу
        return x + noise * plan.additive_scale
