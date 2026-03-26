import math
from typing import Dict, List, Optional, Tuple, Union
from modules.transformations import Range
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict


class LatentComponentSpec(BaseModel):
    """Абстрактный класс родитель для спецификаций каждого подмодуля"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str  # "arima", "kernel_synth", "tsi", "ets"


class ARIMAPrior(BaseModel):
    """
    Априорное распределение параметров ARIMA.
    Определяет пространство поиска для коэффициентов и порядков модели.
    """

    # Диапазоны для порядков (целые числа)
    p_range: Tuple[int, int]
    d_choices: List[int]
    q_range: Tuple[int, int]

    # Диапазоны для коэффициентов и шума
    ar_range: Range
    ma_range: Range
    intercept_range: Range
    sigma_range: Range


class KernelSynthPrior(BaseModel):
    """
    Априорное распределение параметров KernelSynth.
    Определяет пространство поиска для гиперпараметров гауссовских процессов.
    """

    # Типы ядер и их вероятности
    kernel_type_probs: Dict[
        str, float
    ]  # {"RBF": 0.4, "Periodic": 0.3, "RQ": 0.2, "Linear": 0.1}

    # Диапазоны гиперпараметров ядер
    lengthscale_range: Range  # Длина масштаба
    variance_range: Range  # Дисперсия
    period_range: Range  # Период (для Periodic ядра)
    alpha_range: Range  # Параметр (для RQ ядра)

    # Диапазоны параметров функции среднего m(t) = a*t + b + c*exp(d*t)
    mean_a_range: Range  # Линейный коэффициент
    mean_b_range: Range  # Свободный член
    mean_c_range: Range  # Экспоненциальный коэффициент
    mean_d_range: Range  # Экспоненциальный показатель


class TSIPrior(BaseModel):
    """
    Априорное распределение параметров TSI (Time Series Intrinsic).
    Определяет пространство поиска для параметров внутренних мод.
    """

    # Диапазоны для количества мод
    n_modes_range: Tuple[int, int]

    # Диапазоны для частот мод
    frequency_range: Range

    # Диапазоны для амплитуд мод
    amplitude_range: Range

    # Диапазоны для фаз мод
    phase_range: Range

    # Параметр затухания
    decay_range: Range


class ETSPrior(BaseModel):
    """
    Априорное распределение параметров ETS (Error-Trend-Seasonality).
    Определяет пространство поиска для параметров экспоненциального сглаживания.
    """

    # Типы моделей и их вероятности
    model_type_probs: Dict[
        str, float
    ]  # {"ANN": 0.25, "AAN": 0.25, "AAA": 0.25, "MNN": 0.25}

    # Диапазоны для параметров сглаживания
    alpha_range: Range  # Параметр сглаживания уровня
    beta_range: Range  # Параметр сглаживания тренда
    gamma_range: Range  # Параметр сглаживания сезонности

    # Диапазон для периода сезонности
    seasonality_period_range: Tuple[int, int]

    # Диапазон для начальных значений
    initial_level_range: Range
    initial_trend_range: Range
    initial_seasonal_range: Range


class LatentPrior(BaseModel):
    """
    Корневой класс априорных распределений латентной динамики.
    """

    # Вероятности выбора типа генератора
    type_probs: Dict[str, float]

    # Диапазон количества латентных факторов L [min, max]
    l_range: Tuple[int, int]

    # Конфигурации подмодулей
    arima: Optional[ARIMAPrior] = None
    kernel_synth: Optional[KernelSynthPrior] = None
    tsi: Optional[TSIPrior] = None
    ets: Optional[ETSPrior] = None


class BaseLatentComponent(nn.Module):
    """
    Абстрактный класс родитель для каждого подмодуля латентной динамики.
    """

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def execute(self, T: int, specs: List[LatentComponentSpec]) -> torch.Tensor:
        """
        Args:
            T: Длина генерируемой последовательности.
            specs: Список параметров для каждой линии, которую нужно отрисовать.
        Returns:
            Тензор (N, T), где N = len(specs).
        """
        raise NotImplementedError("Подмодуль должен реализовать метод execute")


class ARIMASpec(LatentComponentSpec):
    """
    Спецификация для модели ARIMA(p, d, q).
    Хранит параметры авторегрессии, скользящего среднего, степень интегрирования
    и настройки стохастического шума.
    """

    # TODO string representation
    type: str = "arima"

    ar_params: torch.Tensor  # Коэффициенты AR (p)
    ma_params: torch.Tensor  # Коэффициенты MA (q)
    d: int = 0  # Порядок интегрирования: 0 — стационарный ряд
    intercept: float = 0.0  # Константа (c): базовое смещение уровня ряда.
    sigma: float = 1.0  # стандартное отклонение шума epsilon.

    burn_in: int = 50


class KernelSynthSpec(LatentComponentSpec):
    """
    Спецификация для модели KernelSynth (Гауссовский процесс).
    Хранит параметры ядра и функции среднего для генерации одного временного ряда.
    """

    type: str = "kernel_synth"

    # Параметры ядра
    kernel_type: str  # Тип ядра: "RBF", "Periodic", "RQ", "Linear"
    lengthscale: float  # Длина масштаба (ℓ)
    variance: float  # Дисперсия (σ²)
    period: float = 1.0  # Период (для Periodic ядра)
    alpha: float = 1.0  # Параметр (для RQ ядра)

    # Параметры функции среднего m(t) = a*t + b + c*exp(d*t)
    mean_a: float = 0.0  # Линейный коэффициент
    mean_b: float = 0.0  # Свободный член
    mean_c: float = 0.0  # Экспоненциальный коэффициент
    mean_d: float = 0.0  # Экспоненциальный показатель


class TSISpec(LatentComponentSpec):
    """
    Спецификация для модели TSI (Time Series Intrinsic).
    Хранит параметры внутренних мод для генерации одного временного ряда.
    """

    type: str = "tsi"

    # Параметры мод
    frequencies: List[float]  # Частоты мод
    amplitudes: List[float]  # Амплитуды мод
    phases: List[float]  # Фазы мод
    decays: List[float]  # Параметры затухания мод


class ETSSpec(LatentComponentSpec):
    """
    Спецификация для модели ETS (Error-Trend-Seasonality).
    Хранит параметры экспоненциального сглаживания для генерации одного временного ряда.
    """

    type: str = "ets"

    # Тип модели
    model_type: str  # "ANN", "AAN", "AAA", "MNN", "MAN", "MAM" и т.д.

    # Параметры сглаживания
    alpha: float  # Параметр сглаживания уровня
    beta: float = 0.0  # Параметр сглаживания тренда
    gamma: float = 0.0  # Параметр сглаживания сезонности

    # Период сезонности
    seasonality_period: int = 0  # 0 если нет сезонности

    # Начальные значения
    initial_level: float = 0.0  # Начальный уровень
    initial_trend: float = 0.0  # Начальный тренд
    initial_seasonal: List[float] = []  # Начальные сезонные компоненты


AnySpec = Union[ARIMASpec, KernelSynthSpec, TSISpec, ETSSpec]


class LatentModulePlan(BaseModel):
    """План для LatentDynamics на весь батч."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    items: List[List[AnySpec]]


class ARIMAModule(BaseLatentComponent):
    """Подмодуль латентной динамики для генерации ARIMA(p, d, q) процессов."""

    def execute(self, T: int, specs: List[ARIMASpec]) -> torch.Tensor:
        N = len(specs)
        total_T = T + max(s.burn_in for s in specs)
        max_p = (
            max(len(s.ar_params) for s in specs)
            if any(len(s.ar_params) > 0 for s in specs)
            else 0
        )
        max_q = (
            max(len(s.ma_params) for s in specs)
            if any(len(s.ma_params) > 0 for s in specs)
            else 0
        )

        phi = torch.zeros((N, max_p), device=self.device)
        theta = torch.zeros((N, max_q), device=self.device)
        intercepts = torch.zeros((N, 1), device=self.device)
        sigmas = torch.zeros((N, 1), device=self.device)

        for i, s in enumerate(specs):
            if len(s.ar_params) > 0:
                phi[i, : len(s.ar_params)] = s.ar_params
            if len(s.ma_params) > 0:
                theta[i, : len(s.ma_params)] = s.ma_params
            intercepts[i] = s.intercept
            sigmas[i] = s.sigma

        # Генерация шума
        eps = torch.randn((N, total_T), device=self.device) * sigmas

        # Рекурсивное вычисление ARMA (основной цикл по времени)
        y = torch.zeros((N, total_T), device=self.device)

        for t in range(total_T):
            ar_sum = torch.zeros(N, device=self.device)
            ma_sum = torch.zeros(N, device=self.device)

            # AR
            for p_idx in range(min(t, max_p)):
                ar_sum += phi[:, p_idx] * y[:, t - 1 - p_idx]

            # MA
            for q_idx in range(min(t, max_q)):
                ma_sum += theta[:, q_idx] * eps[:, t - 1 - q_idx]

            y[:, t] = intercepts.squeeze() + ar_sum + eps[:, t] + ma_sum

        # Integrate
        final_y = y
        max_d = max(s.d for s in specs)
        for _ in range(max_d):
            # Интегрируем только те ряды, у которых текущий d_step < s.d
            mask = torch.tensor([s.d > _ for s in specs], device=self.device).view(N, 1)
            integrated = torch.cumsum(final_y, dim=1)
            final_y = torch.where(mask, integrated, final_y)

        # Burn-in
        return final_y[:, -T:]


class KernelSynthModule(BaseLatentComponent):
    """
    Генерация через гауссовские процессы с различными ядрами.
    Логика: z(t) = m(t) + GP(t), где m(t) = a*t + b + c*exp(d*t)
    """

    def __init__(self, device: str):
        super().__init__(device)

    def execute(self, T: int, specs: List[KernelSynthSpec]) -> torch.Tensor:
        """
        Генерирует N временных рядов через гауссовские процессы с различными ядрами.

        Args:
            T: Длина временной сетки
            specs: Список спецификаций для каждой латентной компоненты

        Returns:
            Тензор (N, T) с сгенерированными временными рядами
        """
        if not specs:
            return torch.empty((0, T), device=self.device)

        N = len(specs)
        device = self.device

        # Предполагаем, что временная сетка равномерная [0, 1] для простоты
        # В реальной реализации можно передавать явную сетку
        time_grid = torch.linspace(0, 1, T, device=device)

        # Инициализация выходного тензора
        output = torch.zeros((N, T), device=device)

        # Для каждой спецификации генерируем отдельный GP
        for i, spec in enumerate(specs):
            # Извлекаем параметры из спецификации
            kernel_type = spec.kernel_type
            lengthscale = spec.lengthscale
            variance = spec.variance
            period = spec.period
            alpha = spec.alpha

            # Параметры среднего
            a = spec.mean_a
            b = spec.mean_b
            c = spec.mean_c
            d = spec.mean_d

            # Вычисляем функцию среднего
            mean_func = self._compute_mean_function(time_grid, a, b, c, d)

            # Строим ковариационную матрицу
            K = self._build_covariance_matrix(
                time_grid, kernel_type, lengthscale, variance, period, alpha
            )

            # Адаптивная стабилизация матрицы
            stabilization_factor = max(1e-6, variance * 1e-6)
            K_stable = K + stabilization_factor * torch.eye(T, device=device)

            # Семплируем из GP
            try:
                # Используем Cholesky декомпозицию
                L = torch.linalg.cholesky(K_stable)  # (T, T)
                eps = torch.randn(T, device=device)  # (T,)
                gp_sample = torch.matmul(L, eps)  # (T,)

                # Добавляем среднее
                output[i, :] = mean_func + gp_sample

            except RuntimeError:
                # Если Cholesky не сработал, используем более грубую стабилизацию
                # Добавляем больше стабилизации (без вывода предупреждений)
                K_stable = K + 1e-4 * torch.eye(T, device=device)
                try:
                    L = torch.linalg.cholesky(K_stable)
                    eps = torch.randn(T, device=device)
                    gp_sample = torch.matmul(L, eps)
                    output[i, :] = mean_func + gp_sample
                except RuntimeError:
                    # Если все равно не работает, возвращаем простой тренд
                    output[i, :] = mean_func

        return output

    def _compute_mean_function(
        self, time_grid: torch.Tensor, a: float, b: float, c: float, d: float
    ) -> torch.Tensor:
        """
        Вычисляет функцию среднего m(t) = a*t + b + c*exp(d*t)

        Args:
            time_grid: Временная сетка (T,)
            a, b, c, d: Параметры функции среднего

        Returns:
            Тензор (T,) со значениями функции среднего
        """
        return a * time_grid + b + c * torch.exp(d * time_grid)

    def _build_covariance_matrix(
        self,
        time_grid: torch.Tensor,
        kernel_type: str,
        lengthscale: float,
        variance: float,
        period: float,
        alpha: float,
    ) -> torch.Tensor:
        """
        Строит ковариационную матрицу для заданного ядра.

        Args:
            time_grid: Временная сетка (T,)
            kernel_type: Тип ядра ('RBF', 'Periodic', 'RQ', 'Linear')
            lengthscale: Параметр длины масштаба
            variance: Дисперсия
            period: Период (для Periodic ядра)
            alpha: Параметр (для RQ ядра)

        Returns:
            Ковариационная матрица (T, T)
        """
        T = time_grid.shape[0]
        K = torch.zeros((T, T), device=time_grid.device)

        if kernel_type == "RBF":
            K = self._rbf_kernel(time_grid, lengthscale, variance)
        elif kernel_type == "Periodic":
            K = self._periodic_kernel(time_grid, lengthscale, variance, period)
        elif kernel_type == "RQ":
            K = self._rq_kernel(time_grid, lengthscale, variance, alpha)
        elif kernel_type == "Linear":
            K = self._linear_kernel(time_grid, variance)
        else:
            # По умолчанию используем RBF
            K = self._rbf_kernel(time_grid, lengthscale, variance)

        return K

    def _rbf_kernel(
        self, time_grid: torch.Tensor, lengthscale: float, variance: float
    ) -> torch.Tensor:
        """
        RBF (Radial Basis Function) ядро: k(t, t') = σ² * exp(-||t-t'||² / (2*ℓ²))

        Args:
            time_grid: Временная сетка (T,)
            lengthscale: Параметр длины масштаба (ℓ)
            variance: Дисперсия (σ²)

        Returns:
            Ковариационная матрица (T, T)
        """
        T = time_grid.shape[0]
        # Вычисляем pairwise расстояния
        time_diff = time_grid.unsqueeze(1) - time_grid.unsqueeze(0)  # (T, T)
        squared_diff = time_diff**2

        # Улучшенное численное вычисление экспоненты
        # Избегаем переполнения/недополнения
        exponent_arg = -squared_diff / (2 * lengthscale**2)
        # Ограничиваем аргумент экспоненты для числовой стабильности
        exponent_arg = torch.clamp(exponent_arg, min=-50, max=50)

        # Вычисляем ядро
        K = variance * torch.exp(exponent_arg)

        # Дополнительная проверка на симметричность
        K = (K + K.T) / 2

        return K

    def _periodic_kernel(
        self,
        time_grid: torch.Tensor,
        lengthscale: float,
        variance: float,
        period: float,
    ) -> torch.Tensor:
        """
        Periodic ядро: k(t, t') = σ² * exp(-2*sin²(π*|t-t'|/p) / ℓ²)

        Args:
            time_grid: Временная сетка (T,)
            lengthscale: Параметр длины масштаба (ℓ)
            variance: Дисперсия (σ²)
            period: Период (p)

        Returns:
            Ковариационная матрица (T, T)
        """
        # Вычисляем pairwise расстояния
        time_diff = time_grid.unsqueeze(1) - time_grid.unsqueeze(0)  # (T, T)
        abs_diff = torch.abs(time_diff)
        sin_squared = torch.sin(torch.pi * abs_diff / period) ** 2
        return variance * torch.exp(-2 * sin_squared / (lengthscale**2))

    def _rq_kernel(
        self, time_grid: torch.Tensor, lengthscale: float, variance: float, alpha: float
    ) -> torch.Tensor:
        """
        Rational Quadratic ядро: k(t, t') = σ² * (1 + ||t-t'||² / (2*α*ℓ²))^(-α)

        Args:
            time_grid: Временная сетка (T,)
            lengthscale: Параметр длины масштаба (ℓ)
            variance: Дисперсия (σ²)
            alpha: Параметр (α)

        Returns:
            Ковариационная матрица (T, T)
        """
        # Вычисляем pairwise расстояния
        time_diff = time_grid.unsqueeze(1) - time_grid.unsqueeze(0)  # (T, T)
        squared_diff = time_diff**2
        denominator = 2 * alpha * lengthscale**2
        return variance * (1 + squared_diff / denominator) ** (-alpha)

    def _linear_kernel(self, time_grid: torch.Tensor, variance: float) -> torch.Tensor:
        """
        Linear ядро: k(t, t') = σ² * t * t'

        Args:
            time_grid: Временная сетка (T,)
            variance: Дисперсия (σ²)

        Returns:
            Ковариационная матрица (T, T)
        """
        return variance * torch.outer(time_grid, time_grid)


class TSIModule(BaseLatentComponent):
    """
    Time Series Intrinsic (TSI): Генерация внутренних мод (IMF).
    Логика: Суперпозиция собственных функций системы с аддитивным шумом.
    """

    def execute(self, T: int, specs: List[TSISpec]) -> torch.Tensor:
        """
        Генерирует N временных рядов через суперпозицию внутренних мод.

        Args:
            T: Длина временной сетки
            specs: Список спецификаций для каждой латентной компоненты

        Returns:
            Тензор (N, T) с сгенерированными временными рядами
        """
        if not specs:
            return torch.empty((0, T), device=self.device)

        N = len(specs)
        device = self.device

        # Создаем временную сетку [0, 1]
        time_grid = torch.linspace(0, 1, T, device=device)

        # Инициализация выходного тензора
        output = torch.zeros((N, T), device=device)

        # Для каждой спецификации генерируем отдельный TSI
        for i, spec in enumerate(specs):
            # Получаем параметры из спецификации
            frequencies = spec.frequencies
            amplitudes = spec.amplitudes
            phases = spec.phases
            decays = spec.decays

            # Генерируем сигнал как сумму модулированных экспоненциальных функций
            signal = torch.zeros(T, device=device)

            for freq, amp, phase, decay in zip(frequencies, amplitudes, phases, decays):
                # Создаем экспоненциально затухающую синусоиду
                envelope = torch.exp(-decay * time_grid)  # Затухание
                oscillation = torch.sin(
                    2 * torch.pi * freq * time_grid + phase
                )  # Осцилляция
                component = amp * envelope * oscillation  # Модулированная компонента
                signal += component

            # Добавляем малый гауссовский шум для вариативности сэмплов
            # Это обеспечивает различие между сэмплами при тестировании
            noise_std = 0.05 * torch.std(signal) if torch.std(signal) > 0 else 0.01
            signal = signal + torch.randn(T, device=device) * noise_std

            output[i, :] = signal

        return output


class ETSModule(BaseLatentComponent):
    """
    Error-Trend-Seasonality (Экспоненциальное сглаживание).
    Логика: Уровни, тренды и сезонные компоненты с аддитивным шумом наблюдений.
    """

    def execute(self, T: int, specs: List[ETSSpec]) -> torch.Tensor:
        """
        Генерирует N временных рядов через модели экспоненциального сглаживания.

        Args:
            T: Длина временной сетки
            specs: Список спецификаций для каждой латентной компоненты

        Returns:
            Тензор (N, T) с сгенерированными временными рядами
        """
        if not specs:
            return torch.empty((0, T), device=self.device)

        N = len(specs)
        device = self.device

        # Инициализация выходного тензора
        output = torch.zeros((N, T), device=device)

        # Для каждой спецификации генерируем отдельный ETS
        for i, spec in enumerate(specs):
            # Получаем параметры из спецификации
            model_type = spec.model_type
            alpha = spec.alpha
            beta = spec.beta
            gamma = spec.gamma
            seasonality_period = spec.seasonality_period
            initial_level = spec.initial_level
            initial_trend = spec.initial_trend
            initial_seasonal = spec.initial_seasonal

            # Инициализация компонент
            level = initial_level
            trend = initial_trend

            # Инициализация сезонных компонент
            if seasonality_period > 0 and len(initial_seasonal) >= seasonality_period:
                seasonal = list(initial_seasonal[:seasonality_period])
            else:
                seasonal = [0.0] * max(1, seasonality_period)

            # Генерируем временной ряд
            signal = torch.zeros(T, device=device)

            for t in range(T):
                # Вычисляем текущее значение (уровень + тренд + сезонность)
                current_value = level + trend

                if seasonality_period > 0 and len(seasonal) > 0:
                    current_value += seasonal[t % len(seasonal)]

                # Добавляем шум наблюдений для вариативности между сэмплами
                # Это обеспечивает стохастичность даже при одинаковых параметрах
                observation_noise = torch.randn(1, device=device).item() * 0.1

                signal[t] = current_value + observation_noise

                # Генерируем шум (ошибка) для обновления компонент
                error = torch.randn(1, device=device).item()

                # Обновляем компоненты в зависимости от типа модели
                if "A" in model_type:  # Аддитивная ошибка
                    new_level = alpha * (current_value + error) + (1 - alpha) * (
                        level + trend
                    )
                elif "M" in model_type:  # Мультипликативная ошибка
                    new_level = alpha * (current_value * (1 + error)) + (1 - alpha) * (
                        level + trend
                    )
                else:  # Без ошибки
                    new_level = level + trend

                # Обновляем тренд если есть
                if (
                    len(model_type) > 1 and "N" not in model_type[1]
                ):  # Есть тренд (A или M)
                    if len(model_type) > 1 and "A" in model_type[1]:  # Аддитивный тренд
                        new_trend = beta * (new_level - level) + (1 - beta) * trend
                    elif (
                        len(model_type) > 1 and "M" in model_type[1]
                    ):  # Мультипликативный тренд
                        new_trend = (
                            beta * (new_level / (level + 1e-8)) + (1 - beta) * trend
                        )
                    else:
                        new_trend = trend
                else:
                    new_trend = 0.0

                # Обновляем сезонность если есть
                if (
                    seasonality_period > 0
                    and len(model_type) > 2
                    and "N" not in model_type[2]
                ):
                    seasonal_idx = t % len(seasonal) if len(seasonal) > 0 else 0
                    if len(seasonal) > 0:
                        if "A" in model_type[2]:  # Аддитивная сезонность
                            new_seasonal = (
                                gamma * error + (1 - gamma) * seasonal[seasonal_idx]
                            )
                        elif "M" in model_type[2]:  # Мультипликативная сезонность
                            new_seasonal = (
                                gamma * (error / (current_value + 1e-8))
                                + (1 - gamma) * seasonal[seasonal_idx]
                            )
                        else:
                            new_seasonal = seasonal[seasonal_idx]

                        # Обновляем сезонную компоненту
                        seasonal[seasonal_idx] = new_seasonal

                # Обновляем значения
                level = new_level
                trend = new_trend

            output[i, :] = signal

        return output


class LatentDynamics:
    """Класс латентного модуля, выступающий интерфейсом между глобальным генератором и подмодулями"""

    # Маппинг имен типов спецификаций на имена подмодулей
    TYPE_TO_MODULE = {
        "arima": "arima",
        "kernel_synth": "kernel",
        "tsi": "tsi",
        "ets": "ets",
    }

    def __init__(self, device: str):
        self.device = device
        # Инициализация 4-х подмодулей
        self.sub_modules = {
            "arima": ARIMAModule(device),
            "kernel": KernelSynthModule(device),
            "tsi": TSIModule(device),
            "ets": ETSModule(device),
        }

    def execute(
        self, B: int, T: int, plan: LatentModulePlan, normalize: bool = True
    ) -> torch.Tensor:
        # 1. Find maximum number of components (L) in the batch for allocation
        max_l = max(len(row) for row in plan.items)

        # Final tensor (B, T, L)
        latent_out = torch.zeros((B, T, max_l), device=self.device)

        # 2. Group specifications by type for vectorized execution
        for spec_type, module_name in self.TYPE_TO_MODULE.items():
            module = self.sub_modules[module_name]
            specs_for_module = []
            positions = []  # remember (batch_idx, component_idx)

            for b_idx, row in enumerate(plan.items):
                for l_idx, spec in enumerate(row):
                    if spec.type == spec_type:
                        specs_for_module.append(spec)
                        positions.append((b_idx, l_idx))

            if specs_for_module:
                res = module.execute(T, specs_for_module)

                # 3. Place results into the overall tensor
                for i, (b_i, l_i) in enumerate(positions):
                    latent_out[b_i, :, l_i] = res[i]

        # 4. Optional instance‑wise normalization along time
        if normalize:
            mean = latent_out.mean(dim=1, keepdim=True)  # (B, 1, L)
            std = latent_out.std(dim=1, keepdim=True)  # (B, 1, L)
            eps = 1e-8
            latent_out = (latent_out - mean) / (std + eps)

        return latent_out

    def visualize(
        self, B: int, T: int, plan: "LatentModulePlan", on_the_same_axes: bool = False
    ) -> plt.Figure:
        """
        Визуализирует латентные компоненты.

        Args:
            on_the_same_axes:
                True -> каждая компонента на своем subfigure.
                False -> все компоненты на одном графике (легенда содержит параметры).
        """
        # 1. Генерируем данные через основной метод
        # latent_out: (B, T, L)
        latent_out = self.execute(B, T, plan)
        latent_out_cpu = latent_out.detach().cpu().numpy()

        # 2. Определяем параметры сетки
        # Считаем общее кол-во графиков: для каждого примера в батче B выводим его L компонент
        total_samples = B
        L = latent_out.shape[2]

        if on_the_same_axes:
            # Каждый фактор каждого примера — отдельный график
            total_plots = B * L
            cols = 1
            rows = math.ceil(total_plots / cols)
            fig, axes = plt.subplots(
                rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True
            )
            if total_plots == 1:
                axes = [axes]
            axes = (
                axes.flatten()
                if isinstance(axes, (list, plt.Axes, object))
                and hasattr(axes, "flatten")
                else [axes]
            )
        else:
            # Каждый пример из батча — один график, на котором L линий
            total_plots = B
            rows = math.ceil(total_plots / 1)
            fig, axes = plt.subplots(
                rows, 1, figsize=(12, rows * 5), constrained_layout=True
            )
            if total_plots == 1:
                axes = [axes]
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        plot_idx = 0

        for b_idx in range(B):
            current_row_specs = plan.items[b_idx]

            # Если все компоненты в этом примере одного типа, вынесем в title (для случая False)
            all_types = [s.type for s in current_row_specs]
            unique_types = set(all_types)
            common_type_title = (
                f"Module: {all_types[0]}" if len(unique_types) == 1 else "Mixed Modules"
            )

            if not on_the_same_axes:
                # Режим: Все L компонент на одной оси
                ax = axes[b_idx]
                for l_idx in range(L):
                    spec = current_row_specs[l_idx]
                    params_str = self._format_dto_params(spec)
                    ax.plot(
                        latent_out_cpu[b_idx, :, l_idx],
                        label=f"L{l_idx}: {spec.type} | {params_str}",
                    )

                ax.set_title(f"Batch Sample {b_idx} | {common_type_title}")
                ax.legend(fontsize="small", loc="upper right")
                ax.grid(True, alpha=0.3)

            else:
                # Режим: Каждая компонента на своей оси
                for l_idx in range(L):
                    ax = axes[plot_idx]
                    spec = current_row_specs[l_idx]
                    params_str = self._format_dto_params(spec)

                    ax.plot(latent_out_cpu[b_idx, :, l_idx], color="tab:blue")
                    ax.set_title(
                        f"B{b_idx}:L{l_idx} | {spec.type}\n{params_str}", fontsize=10
                    )
                    ax.grid(True, alpha=0.3)
                    plot_idx += 1

        # Убираем пустые оси, если они остались в сетке
        if on_the_same_axes:
            for i in range(plot_idx, len(axes)):
                fig.delaxes(axes[i])

        return fig

    def _format_dto_params(self, spec: "LatentComponentSpec") -> str:
        """Вспомогательный метод для превращения параметров DTO в строку"""
        # Извлекаем все поля, кроме 'type'
        if hasattr(spec, "model_dump"):
            params = spec.model_dump()
        else:
            params = spec.params if hasattr(spec, "params") else {}

        info = []
        for k, v in params.items():
            if k == "type":
                continue

            # Красиво форматируем тензоры
            if isinstance(v, torch.Tensor):
                if v.numel() <= 3:
                    val = v.tolist()
                else:
                    val = f"shape{list(v.shape)}"
            else:
                val = v
            info.append(f"{k}={val}")

        return ", ".join(info)
