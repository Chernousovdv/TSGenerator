import torch
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple, Annotated
import torch
import random
import numpy as np
from modules.latent import LatentModulePlan, LatentPrior, LatentDynamics
from modules.transformations import TransformModulePlan, TransformPrior, Transformations
from modules.noise_module import NoiseModulePlan, NoisePrior, NoiseModule
from modules.observation_module import (
    ObservationModulePlan,
    ObservationPrior,
    ObservationModule,
)


class GenerationPlan(BaseModel):
    """Полный пакет инструкций для generate_explicit."""

    batch_size: int
    seq_len: int
    dim: int
    latent: LatentModulePlan
    transform: TransformModulePlan
    noise: NoiseModulePlan
    observation: ObservationModulePlan

    class Config:
        arbitrary_types_allowed = True


class GeneratorConfig(BaseModel):
    """Верхнеуровневый конфиг"""

    # Параметры батча
    batch_size: int
    seq_len_range: Tuple[int, int]
    dim_range: Tuple[int, int]

    # Флаг вывода метаданных по умолчанию
    return_metadata: bool = False

    # Приоры модулей
    latent: LatentPrior
    transform: TransformPrior
    noise: NoisePrior
    observation: ObservationPrior


class TSGenerator:
    def __init__(self, config: GeneratorConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        # Инициализация модулей
        self.latent_module = LatentDynamics(device=device)
        self.transform_module = Transformations(device=device)
        self.noise_module = NoiseModule(device=device)
        self.observation_module = ObservationModule(device=device)

    def __call__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, GenerationPlan]]:
        """
        Удобный интерфейс для Online-обучения.
        Сэмплирует случайные размеры батча и данных из конфига.
        """
        return self.generate()

    def generate(
        self,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        dim: Optional[int] = None,
        seed: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, GenerationPlan]]:
        """
        ONLINE РЕЖИМ.
        Логика:
        1. Определяет B, T, dim (из аргументов или из self.config).
        2. Вызывает внутреннего сэмплера для создания GenerationPlan.
        3. Передает план в generate_explicit.
        """
        # Сэмплирование размеров, если не заданы
        B = batch_size or self._sample_int(self.config.batch_size_range)
        T = seq_len or self._sample_int(self.config.seq_len_range)
        D = dim or self._sample_int(self.config.dim_range)

        # Создание плана
        plan = self._sample_plan(B, T, D, seed)

        return self.generate_explicit(plan, return_metadata)

    def generate_explicit(
        self, plan: GenerationPlan, return_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, GenerationPlan]]:
        """
        ДЕТЕРМИНИРОВАННЫЙ РЕЖИМ.
        Выполняет цепочку модулей среднего уровня строго по плану.
        """
        # 1. Латентная динамика -> (B, T, L)
        latent_out = self.latent_module.execute(
            plan.batch_size, plan.seq_len, plan.latent
        )

        # 2. Трансформации (Граф) -> (B, T, dim)
        observed_clean = self.transform_module.execute(latent_out, plan.transform)

        # 3. Наложение шума -> (B, T, dim)
        noisy_data = self.noise_module.execute(observed_clean, plan.noise)

        # 4. Сетка времени и маскирование -> (B, T, dim + 1)
        final_output = self.observation_module.execute(noisy_data, plan.observation)

        if return_metadata:
            return final_output, plan
        return final_output

    def _sample_plan(
        self, B: int, T: int, dim: int, seed: Optional[int]
    ) -> GenerationPlan:
        """
        Логика ParameterSampler. Проходит по всем 'Prior' классам в конфиге
        и создает конкретный GenerationPlan.
        """
        pass  # Реализация логики сэмплирования параметров

    def _sample_int(self, r: Tuple[int, int], seed: Optional[int] = None) -> int:
        """
        Вспомогательный метод для сэмплирования целого числа из диапазона [min, max].

        Использует локальный генератор случайных чисел для обеспечения
        изолированной воспроизводимости.

        Args:
            r: Кортеж (min, max), определяющий границы включительно.
            seed: Опциональный сид для конкретной операции (обычно используется внутренний).

        Returns:
            Сэмплированное целое число.
        """
        low, high = r
        if low > high:
            raise ValueError(
                f"Нижняя граница ({low}) не может быть больше верхней ({high})"
            )

        return torch.randint(
            low=low, high=high + 1, size=(1,), generator=self.rng
        ).item()
