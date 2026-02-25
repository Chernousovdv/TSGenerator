import torch
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional, Union, Tuple, Annotated
import torch
import random
import numpy as np


class ObservationModulePlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Маска (B, T, dim) из 0 и 1
    missing_mask: torch.Tensor


class ObservationPrior(BaseModel):
    # Единственный параметр — вероятность пропуска
    missing_rate_range: Tuple[float, float]


class ObservationModule:
    def __init__(self, device: str):
        self.device = device

    def execute(self, x: torch.Tensor, plan: ObservationModulePlan) -> torch.Tensor:
        """
        Просто добавляет индекс времени и накладывает маску.

        Args:
            x: Тензор от NoiseModule (B, T, dim).
            plan: План, содержащий только маску пропусков.

        Returns:
            Тензор (B, T, dim + 1), где канал 0 — это 0, 1, 2, ..., T-1.
        """
        B, T, D = x.shape

        # 1. Создаем сетку времени: [0, 1, ..., T-1]
        # Расширяем до (B, T, 1)
        time_steps = torch.arange(T, device=self.device).float()
        time_steps = time_steps.view(1, T, 1).expand(B, -1, -1)

        # 2. Накладываем маску пропусков (зануляем значения)
        observed_x = x * plan.missing_mask

        # 3. Собираем финальный пакет: [Time, Dim_1, Dim_2, ...]
        final_output = torch.cat([time_steps, observed_x], dim=-1)

        return final_output
