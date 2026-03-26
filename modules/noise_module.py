import random
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch.distributions import StudentT, Weibull
from modules.transformations import Range




class NoisePrior(BaseModel):
    additive_scale_range: Range
    additive_df_range: Range
    multiplicative_scale_range: Range
    multiplicative_shape_range: Range  #


class NoiseModulePlan(BaseModel):
    """Параметры шума для всего батча."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Student
    additive_scale: torch.Tensor  # Масштаб (B, D)
    additive_df: torch.Tensor  # Степени свободы (B, D)

    # Weibull
    multiplicative_scale: torch.Tensor  # Базовый масштаб (B, D)
    multiplicative_shape: torch.Tensor  # Параметр концентрации k (B, D)


class NoiseModule(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def execute(self, x: torch.Tensor, plan: NoiseModulePlan) -> torch.Tensor:
        """
        x: (B, T, D) - входной чистый сигнал
        """
        B, T, D = x.shape

        # Student-T
        st_df = plan.additive_df.unsqueeze(1).expand(B, T, D)
        st_scale = plan.additive_scale.unsqueeze(1).expand(B, T, D)

        dist_add = StudentT(df=st_df, loc=0.0, scale=st_scale)
        noise_add = dist_add.sample()

        # Weibull
        w_shape = plan.multiplicative_shape.unsqueeze(1).expand(B, T, D)
        w_scale = plan.multiplicative_scale.unsqueeze(1).expand(B, T, D)

        dist_mult = Weibull(scale=w_scale, concentration=w_shape)

        noise_mult_raw = dist_mult.sample()
        noise_mult = noise_mult_raw - w_scale

        return x * (1 + noise_mult) + noise_add
