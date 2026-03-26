from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field


class Range(BaseModel):
    """Интервал для равномерного сэмплирования."""

    min: float
    max: float

    def sample(self) -> float:
        """Сэмплирует случайное число из диапазона."""
        return float(torch.empty(1).uniform_(self.min, self.max).item())


class SmoothingPrior(BaseModel):
    """Диапазоны для генерации параметров сглаживания."""

    methods: List[str]  # ['moving_average', 'gaussian']
    window_range: Tuple[int, int]  # [min_w, max_w]
    std_range: Range  # Для гауссианы


class TransformationPrior(BaseModel):
    """
    Априорное распределение для генерации символьных деревьев.
    Определяет стратегию роста графа и выбор операторов.
    """

    # 1. Параметры структуры дерева
    max_depth: int  # Максимальная глубина рекурсии

    # Вероятности выбора типа узла (должны суммироваться в 1.0)
    # {'op': 0.5, 'terminal': 0.3, 'constant': 0.2}
    node_kind_probs: Dict[str, float]

    # 2. Математические операции и их веса
    # {'add': 1.0, 'sin': 0.5, 'exp': 0.2, ...}
    op_weights: Dict[str, float]

    # 3. Листья дерева (Terminals & Constants)
    # Вероятность выбора конкретного x_i обычно равномерная (1/L),
    # а для констант (ERC) задаем диапазон:
    constant_range: Range

    # 4. Локальное сглаживание
    smoothing_prob: float  # Вероятность добавить сглаживание в узел
    smoothing_prior: SmoothingPrior

    # 5. Параметры плана (пост-обработка)
    normalization_choices: List[Optional[str]]  # [None, 'z-score', 'max-min']
    output_scale_range: Range  # Для амплитуды каждого канала D


class SmoothingParams(BaseModel):
    """Параметры сглаживания для конкретного узла."""

    method: str = "moving_average"  # 'moving_average', 'gaussian', None
    window_size: int = 5  # Размер окна сглаживания
    std: Optional[float] = 1.0  # Для гауссовского сглаживания


class NodeDTO(BaseModel):
    """Атомарный узел символьного дерева."""

    kind: str  # 'op', 'terminal', 'constant' (ERC)
    value: Union[str, float]  # 'add', 'x_0', или числовое значение для ERC
    children: List["NodeDTO"] = []
    smoothing: Optional[SmoothingParams] = None  # Локальное сглаживание узла

    def to_str(self) -> str:
        """Рекурсивно превращает узел и его потомков в строку."""

        # 1. Формируем базовое строковое представление значения
        if self.kind == "terminal":
            s = str(self.value)  # Например, "x_0"
        elif self.kind == "constant":
            s = f"{float(self.value):.3f}"  # Ограничиваем знаки после запятой
        elif self.kind == "op":
            children_strs = [c.to_str() for c in self.children]

            # Красивое отображение бинарных операторов
            binary_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            if self.value in binary_map and len(children_strs) == 2:
                s = f"({children_strs[0]} {binary_map[self.value]} {children_strs[1]})"
            else:
                # Унарные операторы или функциональный вид: sin(x), exp(x)
                s = f"{self.value}({', '.join(children_strs)})"
        else:
            s = "?"

        # 2. Если на узле есть сглаживание, оборачиваем результат
        if self.smoothing:
            p = self.smoothing
            s = f"smooth_{p.method}({s}, w={p.window_size})"

        return s


class ProtectedMath:
    @staticmethod
    def div(a, b, epsilon=1e-6):
        return a / torch.where(
            torch.abs(b) > epsilon, b, torch.sign(b) * epsilon + epsilon
        )

    @staticmethod
    def log(a, epsilon=1e-6):
        return torch.log(torch.abs(a) + epsilon)

    @staticmethod
    def sqrt(a):
        return torch.sqrt(torch.abs(a))

    @staticmethod
    def exp(a, max_val=20.0):
        # Ограничение аргумента exp предотвращает взрыв до Inf
        return torch.exp(torch.clamp(a, max=-max_val, min=-max_val * 2))

    @staticmethod
    def power(a, b):
        # Реализация a^b для произвольных знаков
        return torch.pow(torch.abs(a), b)


class TransformModulePlan(BaseModel):
    """
    План трансформации для всего батча.
    Матрица [B][D], где для каждого примера в батче и каждой выходной
    размерности D определено свое дерево.
    """

    # trees[b][d] -> Корень дерева
    trees: List[List[NodeDTO]]

    # Параметры пост-обработки
    normalization: Optional[str] = "z-score"  # 'z-score', 'max-min' или None
    output_scales: List[float]  # Коэффициенты масштабирования для каждой размерности D

    def string_representation(self) -> List[List[str]]:
        """
        Возвращает матрицу строк [B][D], где каждая строка —
        это читаемая формула для конкретного канала.
        """
        return [[tree.to_str() for tree in batch_trees] for batch_trees in self.trees]


class TransformationsModule(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.math = ProtectedMath()

        # Маппинг защищенных операций
        self.ops = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": self.math.div,
            "sin": torch.sin,
            "cos": torch.cos,
            "exp": self.math.exp,
            "tanh": torch.tanh,
            "log": self.math.log,
            "abs": torch.abs,
            "pow": self.math.power,
        }

    def _post_process(
        self, x: torch.Tensor, plan: TransformModulePlan, d: int
    ) -> torch.Tensor:
        """
        Применяет нормализацию и масштабирование к сырому сигналу одного канала.
        x: (T,) - сырой сигнал из дерева.
        """
        # 1. Нормализация (если задана в плане)
        if plan.normalization == "z-score":
            # Приводим к среднему 0 и стандартному отклонению 1
            mean = x.mean()
            std = x.std()
            x = (x - mean) / (std + 1e-6)

        elif plan.normalization == "max-min":
            # Приводим к диапазону [0, 1]
            x_min = x.min()
            x_max = x.max()
            x = (x - x_min) / (x_max - x_min + 1e-6)

        # 2. Финальное масштабирование
        # Умножаем на коэффициент амплитуды для данного измерения d
        scale = plan.output_scales[d]
        return x * scale

    def _apply_smoothing(
        self, x: torch.Tensor, params: SmoothingParams
    ) -> torch.Tensor:
        """Применяет локальное сглаживание к тензору (T,)."""
        if params is None or params.method is None:
            return x

        T = x.shape[0]
        # Превращаем (T,) в (1, 1, T) для conv1d
        x_reshaped = x.view(1, 1, -1)
        w = params.window_size

        if params.method == "moving_average":
            kernel = torch.ones((1, 1, w), device=self.device) / w
        elif params.method == "gaussian":
            # Упрощенная генерация гауссианы
            steps = torch.linspace(-(w // 2), w // 2, w, device=self.device)
            kernel = torch.exp(-(steps**2) / (2 * params.std**2))
            kernel = (kernel / kernel.sum()).view(1, 1, -1)

        # Padding для сохранения длины T
        padding = w // 2
        smoothed = F.conv1d(x_reshaped, kernel, padding=padding)

        # Если из-за четного окна длина изменилась, подрезаем до T
        return smoothed.view(-1)[:T]

    def _evaluate_node(self, node: NodeDTO, latent_data: torch.Tensor) -> torch.Tensor:
        """Рекурсивное вычисление узла."""
        # Вычисление базового значения узла
        if node.kind == "terminal":
            idx = int(node.value.split("_")[1])
            res = latent_data[:, idx]

        elif node.kind == "constant":
            # ERC (Ephemeral Random Constant)
            res = torch.full(
                (latent_data.shape[0],), float(node.value), device=self.device
            )

        elif node.kind == "op":
            children_res = [self._evaluate_node(c, latent_data) for c in node.children]
            res = self.ops[node.value](*children_res)

        # Локальное сглаживание
        if node.smoothing:
            res = self._apply_smoothing(res, node.smoothing)

        return res

    def execute(
        self, latent_dynamics: torch.Tensor, plan: TransformModulePlan
    ) -> torch.Tensor:
        """TODO docstring"""
        B, T, L = latent_dynamics.shape
        D = len(plan.trees[0])
        output = torch.zeros((B, T, D), device=self.device)

        for b in range(B):
            for d in range(D):
                tree_root = plan.trees[b][d]
                # Рекурсивный расчет
                raw_signal = self._evaluate_node(tree_root, latent_dynamics[b])

                # Финальная нормализация и масштабирование (из плана)
                output[b, :, d] = self._post_process(raw_signal, plan, d)

        return output
