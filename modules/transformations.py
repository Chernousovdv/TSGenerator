import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Optional, Tuple, Callable

# --- 1. План и Конфигурация (Contracts) ---


class TransformModulePlan(BaseModel):
    """
    Детерминированный рецепт преобразования латентных факторов в наблюдаемые.
    Использует структуру MLP с фиксированными параметрами.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Список тензоров весов и смещений для каждого слоя
    weights: List[torch.Tensor]  # Формы: (In_i, Out_i)
    biases: List[torch.Tensor]  # Формы: (Out_i)

    # Список имен функций активации для каждого слоя
    activations: List[str]

    # Флаг использования Skip-connections от входного латентного тензора
    use_skip_connections: bool

    # Список индексов слоев, к которым нужно подмешивать исходный вход
    skip_indices: List[int]


class TransformPrior(BaseModel):
    """Приоры для сэмплирования архитектуры трансформаций."""

    complexity_range: Tuple[int, int]  # Глубина сети (кол-во слоев)
    hidden_dim_range: Tuple[int, int]  # Ширина скрытых слоев
    op_probs: Dict[str, float]  # Вероятности выбора активаций
    skip_connection_prob: float  # Вероятность создания skip-connection


# --- 2. Реализация модуля ---


class Transformations:
    """
    Модуль, выполняющий роль вычислительного графа (DAG) в форме MLP.
    Превращает (B, T, L) -> (B, T, dim).
    """

    def __init__(self, device: str):
        self.device = device

        # Реестр фиксированных активаций (атомарные операции)
        self.activation_registry: Dict[str, Callable] = {
            "identity": lambda x: x,
            "relu": torch.relu,
            "sin": torch.sin,
            "cos": torch.cos,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "abs": torch.abs,
            "softplus": torch.nn.functional.softplus,
            "square": lambda x: torch.pow(x, 2),
            "sign": torch.sign,
        }

    def execute(
        self, latent_tensor: torch.Tensor, plan: TransformModulePlan
    ) -> torch.Tensor:
        """
        Последовательно применяет слои трансформации согласно плану.

        Args:
            latent_tensor: Входной тензор от LatentDynamics (B, T, L).
            plan: Сэмплированный план с весами и типами активаций.

        Returns:
            Тензор размерности (B, T, dim).
        """
        x = latent_tensor
        input_ref = latent_tensor  # Ссылка для skip-connections

        # Проход по слоям (Register Machine Style)
        for i, (W, b, act_name) in enumerate(
            zip(plan.weights, plan.biases, plan.activations)
        ):

            # 1. Линейная проекция: (B, T, In) @ (In, Out) -> (B, T, Out)
            x = torch.matmul(x, W) + b

            # 2. Применение нелинейности из реестра
            if act_name in self.activation_registry:
                x = self.activation_registry[act_name](x)
            else:
                raise ValueError(f"Activation '{act_name}' not found in registry.")

            # 3. Обработка Skip-connections
            # Если текущий слой указан в плане как точка слияния
            if plan.use_skip_connections and i in plan.skip_indices:
                # Нам нужно убедиться, что размерности совпадают для сложения.
                # Если нет — просто конкатенируем или используем проекцию.
                # Здесь для простоты используем сложение, если Out == L.
                if x.shape[-1] == input_ref.shape[-1]:
                    x = x + input_ref
                else:
                    # Если размерности разные, можно реализовать "линейную прокидку"
                    # но в данной архитектуре мы предполагаем, что план уже учитывает это.
                    x = torch.cat([x, input_ref], dim=-1)

        return x
