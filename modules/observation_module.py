import random
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict


class ObservationModulePlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ObservationModule:
    def __init__(self, device: str):
        self.device = device

    def execute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Просто добавляет индекс времени.

        Args:
            x: Тензор от NoiseModule (B, T, dim).

        Returns:
            Тензор (B, T, dim + 1), где канал 0 — это 0, 1, 2, ..., T-1.
        """
        B, T, D = x.shape
        time_steps = torch.arange(T, device=self.device).float()
        time_steps = time_steps.view(1, T, 1).expand(B, -1, -1)
        final_output = torch.cat([time_steps, x], dim=-1)

        return final_output
