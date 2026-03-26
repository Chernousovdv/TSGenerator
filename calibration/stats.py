"""
statistics.py — одномерные статистики временных рядов.

Интерфейс
---------
Все функции принимают массив формы (B, T).

Большинство функций возвращают (B,) — по одному скаляру на ряд.
Исключение: `acf` возвращает (B, len(lags)) — по одному значению на лаг.

B — количество рядов (батч или произвольная выборка).
T — длина временного ряда.

Функции не знают ничего о генераторе — они работают с голым np.ndarray.
Это позволяет переиспользовать их для:
  - выхода LatentDynamics        : tensor (B, T, L) → нарезать по L
  - выхода TSGenerator           : tensor (B, T, D+1) → нарезать по D
  - реальных датасетов           : np.ndarray (B, T)

Зависимости
-----------
  numpy        — обязательно
  statsmodels  — только для adf_statistic
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_input(x: np.ndarray) -> np.ndarray:
    """
    Привести вход к float64 и проверить форму (B, T).

    Parameters
    ----------
    x : array-like, shape (B, T)

    Returns
    -------
    np.ndarray, shape (B, T), dtype float64

    Raises
    ------
    ValueError
        Если размерность не равна 2.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(
            f"Ожидается массив формы (B, T), получено shape={x.shape}. "
            "Если у вас одномерный ряд, добавьте батч-измерение: x[np.newaxis, :]"
        )
    # Заменяем ±inf на NaN: статистики на взорвавшихся рядах
    # бессмысленны, а inf ломает numpy-операции (subtract, histogram).
    x = np.where(np.isinf(x), np.nan, x)
    return x


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Поэлементное деление; там где b == 0 возвращает NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(b == 0, np.nan, a / b)
    return result


# ---------------------------------------------------------------------------
# 1. mean
# ---------------------------------------------------------------------------

def mean(x: np.ndarray) -> np.ndarray:
    """
    Среднее значение ряда (уровень / смещение).

    Что измеряет
    ------------
    Уровень ряда без учёта тренда. Для рядов с трендом среднее смещено
    к центру тренда. Для центрированных стационарных рядов ≈ 0.

    Связь с параметрами генератора
    ------------------------------
    ARIMA.intercept    → E[X] = intercept / (1 - sum(ar_params))
    KernelSynth.mean_b → прямой вертикальный сдвиг функции среднего
    TSI                → по конструкции ≈ 0 (синусоиды без смещения)

    Parameters
    ----------
    x : np.ndarray, shape (B, T)

    Returns
    -------
    np.ndarray, shape (B,)

    Examples
    --------
    >>> x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 6.0]])
    >>> mean(x)
    array([2., 2.])
    >>> mean(np.full((1, 100), 5.0))
    array([5.])
    """
    x = _validate_input(x)
    return x.mean(axis=1)


# ---------------------------------------------------------------------------
# 2. std
# ---------------------------------------------------------------------------

def std(x: np.ndarray) -> np.ndarray:
    """
    Стандартное отклонение ряда (амплитуда колебаний).

    Что измеряет
    ------------
    Масштаб ряда. Критично для проверки согласованности амплитуд
    разных латентных компонент перед их смешиванием в TransformationModule:
    компонента с std=10 будет подавлять компоненту с std=0.1.

    Связь с параметрами генератора
    ------------------------------
    ARIMA.sigma        → основной вклад в амплитуду
    KernelSynth.variance → дисперсия GP σ²; std ≈ sqrt(variance)
    TSI.amplitudes     → прямое управление амплитудой каждой моды
    ETS.initial_level  → косвенно через накопление уровня

    Parameters
    ----------
    x : np.ndarray, shape (B, T)

    Returns
    -------
    np.ndarray, shape (B,)
        Несмещённая оценка (ddof=1). Константный ряд → 0.0.

    Examples
    --------
    >>> std(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    array([1.58113883])
    >>> std(np.full((1, 10), 7.0))
    array([0.])
    """
    x = _validate_input(x)
    return x.std(axis=1, ddof=1)


# ---------------------------------------------------------------------------
# 3. acf
# ---------------------------------------------------------------------------

def acf(
    x: np.ndarray,
    lags: Sequence[int] = (1, 2, 5, 10),
) -> np.ndarray:
    """
    Выборочная автокорреляционная функция на заданных лагах.

    Что измеряет
    ------------
    Силу и знак линейной зависимости между значениями ряда,
    разделёнными на k шагов. Ключевые ориентиры:
      lag=1 : «память» между соседними точками.
              AR(1) с φ=0.9       → acf(1) ≈ 0.9
              случайное блуждание → acf(1) ≈ 1
              белый шум           → acf(1) ≈ 0
      lag=k : для AR(1)  acf(k) ≈ φ^k  (экспоненциальное затухание)

    Связь с параметрами генератора
    ------------------------------
    ARIMA.ar_params  → acf(1) ≈ ar_params[0] при q=0, d=0
    ARIMA.d≥1        → acf на всех лагах стремится к 1
    KernelSynth      → lengthscale управляет скоростью убывания acf

    Parameters
    ----------
    x    : np.ndarray, shape (B, T)
    lags : sequence of int, default (1, 2, 5, 10)
        Лаги. Все значения должны быть положительными и < T.

    Returns
    -------
    np.ndarray, shape (B, len(lags))
        Значения в [-1, 1].
        Константный ряд (нулевая дисперсия) → NaN по всем лагам.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> white = rng.standard_normal((1, 1000))
    >>> acf(white, lags=[1, 5])   # ≈ [[0.0, 0.0]]
    """
    x = _validate_input(x)
    lags = list(lags)
    B, T = x.shape

    xc = x - x.mean(axis=1, keepdims=True)       # (B, T) центрированный
    denom = (xc ** 2).sum(axis=1)                 # (B,)

    result = np.empty((B, len(lags)), dtype=np.float64)
    for j, k in enumerate(lags):
        if k <= 0 or k >= T:
            raise ValueError(
                f"Лаг {k} вне допустимого диапазона [1, T-1={T - 1}]."
            )
        numer = (xc[:, k:] * xc[:, :-k]).sum(axis=1)   # (B,)
        result[:, j] = _safe_divide(numer, denom)

    return result


# ---------------------------------------------------------------------------
# 4. adf_statistic
# ---------------------------------------------------------------------------

def adf_statistic(
    x: np.ndarray,
    regression: str = "ct",
    autolag: str = "AIC",
) -> np.ndarray:
    """
    ADF-статистика — тест на наличие единичного корня (нестационарность).

    Что измеряет
    ------------
    Нестационарность типа случайного блуждания (интегрированный процесс).
    Чем меньше значение (например, -5), тем сильнее стационарность.
    Значение около 0 или выше — признак нестационарности.

    Связь с параметрами генератора
    ------------------------------
    ARIMA.d=0         → ожидается отрицательная статистика (стационарный)
    ARIMA.d=1         → статистика ≈ 0 или выше (случайное блуждание)
    ARIMA.d=2         → статистика ещё выше
    KernelSynth mean_a≠0 / mean_c≠0 → тренд; значение зависит от regression

    Parameters
    ----------
    x          : np.ndarray, shape (B, T)
    regression : str, default 'ct'
        Детерминированный компонент в тестовой регрессии:
          'n'   — без константы и тренда
          'c'   — только константа
          'ct'  — константа + линейный тренд  ← рекомендуется по умолчанию
          'ctt' — константа + линейный + квадратичный тренд
        Выбор влияет на критические значения теста. 'ct' покрывает
        общий случай с неизвестным трендом.
    autolag    : str, default 'AIC'
        Критерий выбора числа лагов в тестовой регрессии:
        'AIC', 'BIC', 't-stat' или None.

    Returns
    -------
    np.ndarray, shape (B,)
        Тестовая статистика (float). Не p-value.
        Вырожденный или константный ряд → NaN.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> white = rng.standard_normal((3, 500))
    >>> adf_statistic(white)                    # << 0 (стационарный)
    >>> adf_statistic(np.cumsum(white, axis=1)) # ≈ 0 (случайное блуждание)
    """
    from statsmodels.tsa.stattools import adfuller

    x = _validate_input(x)
    B = x.shape[0]
    result = np.empty(B, dtype=np.float64)

    for b in range(B):
        try:
            stat, *_ = adfuller(x[b], regression=regression, autolag=autolag)
            result[b] = stat
        except Exception:
            result[b] = np.nan

    return result


# ---------------------------------------------------------------------------
# 5. permutation_entropy
# ---------------------------------------------------------------------------

# Все перестановки порядка m строятся один раз при первом вызове
# и кешируются по значению m.
_PERM_INDEX_CACHE: dict[int, dict[tuple, int]] = {}


def _get_perm_index(m: int) -> dict[tuple, int]:
    """Словарь перестановка → индекс, кешируется по m."""
    if m not in _PERM_INDEX_CACHE:
        _PERM_INDEX_CACHE[m] = {
            perm: idx for idx, perm in enumerate(permutations(range(m)))
        }
    return _PERM_INDEX_CACHE[m]


def permutation_entropy(
    x: np.ndarray,
    m: int = 3,
    tau: int = 1,
) -> np.ndarray:
    """
    Нормированная перестановочная энтропия.

    Что измеряет
    ------------
    Динамическую сложность ряда через распределение порядковых паттернов.
    0 — полная регулярность (монотонный ряд),
    1 — максимальная сложность (белый шум).

    Хорошо разделяет типы латентных компонент:
      TSI (регулярные синусоиды)          → низкая PE
      KernelSynth/RBF (большой lengthscale) → низкая PE (гладкий GP)
      ARIMA (d=0, малые AR)              → высокая PE (близко к шуму)
      ARIMA (d=1)                        → средняя PE (случайное блуждание)
      ETS с трендом                      → низкая PE

    Алгоритм
    --------
    1. Строятся все подвекторы длины m с шагом tau:
       v_t = (x[t], x[t+tau], ..., x[t+(m-1)*tau])
    2. Для каждого v_t определяется порядковый паттерн (argsort).
    3. Считаются частоты m! возможных паттернов → p_i.
    4. PE = -sum(p_i * log2(p_i)) / log2(m!)

    Parameters
    ----------
    x   : np.ndarray, shape (B, T)
    m   : int, default 3
        Порядок (длина паттерна). При m=3 — 6 паттернов,
        устойчиво при малых T. При m=5 — 120 паттернов, нужен T >> 120.
    tau : int, default 1
        Лаг между элементами паттерна.

    Returns
    -------
    np.ndarray, shape (B,)
        Значения в [0, 1]. При T <= m*tau → NaN.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> permutation_entropy(rng.standard_normal((1, 1000)))  # ≈ 1.0
    >>> mono = np.arange(1000, dtype=float)[np.newaxis, :]
    >>> permutation_entropy(mono)                            # ≈ 0.0
    """
    x = _validate_input(x)
    B, T = x.shape

    min_len = m * tau
    if T <= min_len:
        return np.full(B, np.nan)

    perm_index = _get_perm_index(m)
    n_perms = len(perm_index)      # m!
    log_n = np.log2(n_perms)
    result = np.empty(B, dtype=np.float64)

    # Матрица индексов окон: shape (n_windows, m)
    n_windows = T - (m - 1) * tau
    offsets = np.arange(m) * tau                            # (m,)
    window_idx = np.arange(n_windows)[:, None] + offsets    # (n_windows, m)

    for b in range(B):
        if np.any(np.isnan(x[b])):
            result[b] = np.nan
            continue
        windows = x[b][window_idx]                          # (n_windows, m)
        patterns = tuple(
            map(tuple, np.argsort(windows, axis=1, kind="stable"))
        )
        counts = np.zeros(n_perms, dtype=np.float64)
        for pat in patterns:
            counts[perm_index[pat]] += 1.0

        probs = counts / counts.sum()
        nonzero = probs > 0
        entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))
        result[b] = entropy / log_n if log_n > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# 6. mann_kendall_z
# ---------------------------------------------------------------------------

def mann_kendall_z(x: np.ndarray) -> np.ndarray:
    """
    Z-статистика теста Манна-Кендалла на монотонный тренд.

    Что измеряет
    ------------
    Наличие монотонного (не обязательно линейного) тренда.
    Z > 0  — возрастающий тренд
    Z < 0  — убывающий тренд
    |Z| > 1.96 → значимо на уровне 0.05

    Связь с параметрами генератора
    ------------------------------
    ARIMA.d=1 + intercept≠0    → сильный монотонный тренд, высокий |Z|
    KernelSynth.mean_a≠0       → монотонный линейный тренд
    KernelSynth.mean_c≠0       → монотонный экспоненциальный тренд
    ETS с трендом (AAN, AAA)   → возрастающий тренд

    Алгоритм
    --------
    S    = sum_{i<j} sign(x[j] - x[i])
    Var(S) = T*(T-1)*(2*T+5) / 18
    Z    = (S - sign(S)) / sqrt(Var(S))   (continuity correction)

    Примечание: тест предполагает независимость наблюдений.
    При сильной автокорреляции |Z| может быть завышен.

    Parameters
    ----------
    x : np.ndarray, shape (B, T)

    Returns
    -------
    np.ndarray, shape (B,)
        Z-статистика. При T < 4 → NaN.

    Examples
    --------
    >>> t = np.linspace(0, 1, 200)[np.newaxis, :]
    >>> mann_kendall_z(t)   # >> 1.96  (сильный возрастающий тренд)
    >>> mann_kendall_z(-t)  # << -1.96 (сильный убывающий тренд)
    >>> rng = np.random.default_rng(0)
    >>> mann_kendall_z(rng.standard_normal((1, 500)))  # ≈ 0
    """
    x = _validate_input(x)
    B, T = x.shape

    if T < 4:
        return np.full(B, np.nan)

    var_s = T * (T - 1) * (2 * T + 5) / 18.0
    sqrt_var_s = np.sqrt(var_s)
    result = np.empty(B, dtype=np.float64)

    for b in range(B):
        xi = x[b]
        if np.any(np.isnan(xi)):
            result[b] = np.nan
            continue
        # Векторизованный подсчёт S через верхнетреугольные разности
        diff = xi[None, :] - xi[:, None]                   # (T, T)
        S = np.sign(diff[np.triu_indices(T, k=1)]).sum()

        if S > 0:
            result[b] = (S - 1.0) / sqrt_var_s
        elif S < 0:
            result[b] = (S + 1.0) / sqrt_var_s
        else:
            result[b] = 0.0

    return result


# ---------------------------------------------------------------------------
# 7. roughness
# ---------------------------------------------------------------------------

def roughness(x: np.ndarray) -> np.ndarray:
    """
    Относительная шероховатость ряда: std(diff(x)) / std(x).

    Что измеряет
    ------------
    Высокочастотное содержание ряда относительно его общей амплитуды.
    Хорошо разделяет типы латентных компонент:
      KernelSynth/RBF (большой lengthscale) → малое (гладкий GP)
      KernelSynth/Periodic (малый period)   → среднее
      TSI (высокие частоты)                 → большое
      ARIMA (большая sigma, малые AR)       → большое (близко к шуму)
      ARIMA (большие AR / d=1)             → малое (гладкое блуждание)

    Формула
    -------
    R = std(Δx) / std(x),   Δx_t = x_t - x_{t-1}
    std вычисляется с ddof=1.

    Parameters
    ----------
    x : np.ndarray, shape (B, T)

    Returns
    -------
    np.ndarray, shape (B,)
        Значения ≥ 0. Константный ряд (std=0) → NaN.

    Examples
    --------
    >>> t = np.linspace(0, 2 * np.pi, 500)
    >>> smooth = np.sin(t)[np.newaxis, :]
    >>> roughness(smooth)                              # малое значение
    >>> rng = np.random.default_rng(0)
    >>> roughness(rng.standard_normal((1, 500)))       # большое значение
    """
    x = _validate_input(x)
    dx = np.diff(x, axis=1)                  # (B, T-1)
    std_dx = dx.std(axis=1, ddof=1)          # (B,)
    std_x = x.std(axis=1, ddof=1)            # (B,)
    return _safe_divide(std_dx, std_x)


# ---------------------------------------------------------------------------
# 8. forecastability
# ---------------------------------------------------------------------------

def forecastability(
    x: np.ndarray,
    detrend: bool = True,
) -> np.ndarray:
    """
    Предсказуемость ряда через спектральную энтропию (Goerg, 2013).

    Что измеряет
    ------------
    Концентрацию энергии в частотной области. Высокая концентрация
    (доминирующие частоты) → высокая предсказуемость. Равномерный
    спектр (белый шум) → нулевая предсказуемость.

    Диапазон [0, 1]:
      ≈ 1   — идеальная синусоида (вся энергия в одной частоте)
      ≈ 0   — белый шум (энергия равномерно распределена по частотам)

    Хорошо разделяет типы латентных компонент:
      TSI (синусоиды)                        → высокая (~0.8–1.0)
      KernelSynth/Periodic                   → высокая (~0.6–0.9)
      KernelSynth/RBF (большой lengthscale)  → средняя (~0.3–0.6)
      ETS с трендом                          → средняя (после детрендинга)
      ARIMA (d=0, малые AR)                  → низкая (~0.0–0.2)

    Алгоритм
    --------
    1. Опционально удаляем линейный тренд (scipy.signal.detrend).
    2. Применяем окно Ханна для подавления спектральных утечек.
    3. Вычисляем одностороннюю PSD: p_i = |FFT(x)[i]|², i ∈ [1, T//2].
       DC-компонента (i=0) исключается — она кодирует среднее, не структуру.
    4. Нормируем: p_i /= sum(p_i)  →  вероятностное распределение.
    5. Спектральная энтропия: H = -sum(p_i * log(p_i))
       H_max = log(T//2)  — энтропия белого шума (равномерный спектр).
    6. Forecastability = 1 - H / H_max

    Параметры
    ---------
    detrend : bool, default True
        Удалять ли линейный тренд перед вычислением.
        Рекомендуется True: тренд создаёт большую DC-подобную компоненту
        на низких частотах и искажает оценку предсказуемости осциллирующей
        части ряда.

    Parameters
    ----------
    x       : np.ndarray, shape (B, T)
    detrend : bool, default True

    Returns
    -------
    np.ndarray, shape (B,)
        Значения в [0, 1]. Константный ряд → NaN.
        При T < 4 → NaN (недостаточно частотных бинов).

    Examples
    --------
    >>> T = 500
    >>> t = np.linspace(0, 1, T)
    >>> sine  = np.sin(2 * np.pi * 5 * t)[np.newaxis, :]
    >>> white = np.random.default_rng(0).standard_normal((1, T))
    >>> forecastability(sine)    # ≈ 1.0
    >>> forecastability(white)   # ≈ 0.0
    """
    from scipy.signal import detrend as sp_detrend

    x = _validate_input(x)
    B, T = x.shape
    n_freq = T // 2   # число положительных частот (без DC)

    if n_freq < 2:
        return np.full(B, np.nan)

    H_max = np.log(n_freq)   # энтропия равномерного спектра
    result = np.empty(B, dtype=np.float64)

    # Окно Ханна: подавляет спектральные утечки на краях
    hann = np.hanning(T)                    # (T,)

    for b in range(B):
        xi = x[b]
        if np.any(np.isnan(xi)):
            result[b] = np.nan
            continue

        # 1. Линейный детрендинг
        xi_d = sp_detrend(xi) if detrend else xi.copy()

        # 2. Взвешивание окном Ханна
        xi_w = xi_d * hann

        # 3. FFT → одностороння PSD, без DC (индекс 0)
        fft_vals = np.fft.rfft(xi_w)        # (T//2 + 1,)
        psd = np.abs(fft_vals[1:n_freq + 1]) ** 2   # (n_freq,)

        psd_sum = psd.sum()
        if psd_sum == 0:
            result[b] = np.nan
            continue

        # 4. Нормировка → вероятностное распределение
        p = psd / psd_sum

        # 5. Спектральная энтропия (только ненулевые вероятности)
        nonzero = p > 0
        H = -np.sum(p[nonzero] * np.log(p[nonzero]))

        # 6. Forecastability = 1 - H / H_max
        result[b] = 1.0 - H / H_max if H_max > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# 9. fft_mean
# ---------------------------------------------------------------------------

def fft_mean(
    x: np.ndarray,
    detrend: bool = True,
) -> np.ndarray:
    """
    Среднее значение односторонней нормированной PSD (FFT Mean).

    Что измеряет
    ------------
    Среднюю «энергию» ряда в частотной области после нормировки на
    длину ряда. Позволяет сравнивать спектральную интенсивность рядов
    разной длины.

    В отличие от forecastability (форма спектра), fft_mean характеризует
    абсолютный уровень спектральной мощности. Полезен для обнаружения
    рядов с аномально высокой энергией.

    Связь с параметрами генератора
    ------------------------------
    KernelSynth.variance   → прямо управляет fft_mean
    ARIMA.sigma            → основной вклад в fft_mean
    TSI.amplitudes         → прямое управление амплитудой мод

    Формула
    -------
    fft_mean = mean(|FFT(x)[1 : T//2 + 1]|² / T)

    DC-компонента (i=0) исключается. Деление на T нормирует по длине ряда
    (теорема Парсеваля: sum(|FFT|²) / T = sum(x²)).

    Parameters
    ----------
    x       : np.ndarray, shape (B, T)
    detrend : bool, default True
        Удалять ли линейный тренд перед вычислением.

    Returns
    -------
    np.ndarray, shape (B,)
        Значения ≥ 0. Константный ряд → 0.0.
        Ряд с NaN → NaN.

    Examples
    --------
    >>> T = 500
    >>> t = np.linspace(0, 1, T)
    >>> low_amp  = np.sin(2 * np.pi * 5 * t)[np.newaxis, :]          # амплитуда 1
    >>> high_amp = (5 * np.sin(2 * np.pi * 5 * t))[np.newaxis, :]    # амплитуда 5
    >>> fft_mean(low_amp)   # малое значение
    >>> fft_mean(high_amp)  # в 25 раз больше (амплитуда² ∝ мощности)
    """
    from scipy.signal import detrend as sp_detrend

    x = _validate_input(x)
    B, T = x.shape
    n_freq = T // 2

    if n_freq < 1:
        return np.full(B, np.nan)

    result = np.empty(B, dtype=np.float64)

    for b in range(B):
        xi = x[b]
        if np.any(np.isnan(xi)):
            result[b] = np.nan
            continue

        xi_d = sp_detrend(xi) if detrend else xi.copy()

        fft_vals = np.fft.rfft(xi_d)                        # (T//2 + 1,)
        psd = np.abs(fft_vals[1:n_freq + 1]) ** 2 / T      # нормировка Парсеваля
        result[b] = psd.mean()

    return result


# ---------------------------------------------------------------------------
# 10. seasonality_strength  /  trend_strength
# ---------------------------------------------------------------------------

# Набор периодов как в S² / Bahrpeyma et al. (2021)
_STL_PERIODS = (4, 7, 12, 24, 52)


def _stl_strengths(xi: np.ndarray, period: int) -> tuple[float, float]:
    """
    Возвращает (seasonality_strength, trend_strength) для одного ряда
    через STL-декомпозицию с заданным периодом.

    Формулы (Bahrpeyma 2021, Wang et al. S²):
        FS = max(0, 1 - Var(R) / Var(S + R))
        FT = max(0, 1 - Var(R) / Var(T + R))

    Возвращает (nan, nan) если декомпозиция невозможна.
    """
    from statsmodels.tsa.seasonal import STL

    T = len(xi)
    # STL требует T >= 2 * period
    if T < 2 * period:
        return np.nan, np.nan

    try:
        stl = STL(xi, period=period, robust=True)
        res = stl.fit()
    except Exception:
        return np.nan, np.nan

    R = res.resid
    S = res.seasonal
    Tr = res.trend

    var_R = np.var(R, ddof=1)
    var_SR = np.var(S + R, ddof=1)
    var_TR = np.var(Tr + R, ddof=1)

    fs = max(0.0, 1.0 - var_R / var_SR) if var_SR > 0 else 0.0
    ft = max(0.0, 1.0 - var_R / var_TR) if var_TR > 0 else 0.0
    return fs, ft


def seasonality_strength(
    x: np.ndarray,
    periods: tuple[int, ...] = _STL_PERIODS,
) -> np.ndarray:
    """
    Сила сезонности через STL-декомпозицию (максимум по набору периодов).

    Что измеряет
    ------------
    Долю дисперсии ряда, объяснённую сезонной компонентой.
    Диапазон [0, 1]: 0 — нет сезонности, 1 — идеально периодический ряд.

    Алгоритм (как в S² и Bahrpeyma et al. 2021)
    -------------------------------------------
    Для каждого периода m ∈ periods:
        Запустить STL(x, period=m, robust=True)
        FS(m) = max(0, 1 - Var(R) / Var(S + R))
    Вернуть max(FS(m)) по всем m.

    Брать максимум по периодам — стандартный подход в S²: мы не знаем
    истинный период, поэтому берём наилучшее покрытие из кандидатов.

    Связь с параметрами генератора
    ------------------------------
    KernelSynth/Periodic    → высокая (period попадает в candidates)
    ETS с seasonality_period → высокая если period в candidates
    TSI (низкие частоты)    → средняя
    ARIMA (d=0)             → низкая
    ARIMA (d=1)             → низкая (тренд не сезонность)

    Parameters
    ----------
    x       : np.ndarray, shape (B, T)
    periods : tuple[int, ...], default (4, 7, 12, 24, 52)
        Кандидаты периодов. STL пропускает период если T < 2 * period.

    Returns
    -------
    np.ndarray, shape (B,)
        Значения в [0, 1]. NaN если ни один период не применим.

    Examples
    --------
    >>> T = 200
    >>> t = np.arange(T)
    >>> seasonal = np.sin(2 * np.pi * t / 12)[np.newaxis, :]  # период 12
    >>> seasonality_strength(seasonal)   # ≈ высокое значение
    >>> rng = np.random.default_rng(0)
    >>> white = rng.standard_normal((1, T))
    >>> seasonality_strength(white)      # ≈ низкое значение
    """
    x = _validate_input(x)
    B = x.shape[0]
    result = np.empty(B, dtype=np.float64)

    for b in range(B):
        xi = x[b]
        if np.any(np.isnan(xi)):
            result[b] = np.nan
            continue

        best = np.nan
        for m in periods:
            fs, _ = _stl_strengths(xi, m)
            if np.isfinite(fs):
                best = fs if np.isnan(best) else max(best, fs)

        result[b] = best

    return result


def trend_strength(
    x: np.ndarray,
    periods: tuple[int, ...] = _STL_PERIODS,
) -> np.ndarray:
    """
    Сила тренда через STL-декомпозицию (максимум по набору периодов).

    Что измеряет
    ------------
    Долю дисперсии ряда, объяснённую трендовой компонентой.
    Диапазон [0, 1]: 0 — нет тренда, 1 — ряд полностью объяснён трендом.

    Алгоритм
    --------
    Аналогично seasonality_strength, но берётся FT:
        FT(m) = max(0, 1 - Var(R) / Var(T + R))
    Результат — max(FT(m)) по всем периодам.

    Замечание: trend_strength слабо зависит от выбора периода по сравнению
    с seasonality_strength, поэтому максимум по периодам здесь менее важен,
    но используется для консистентности.

    Связь с параметрами генератора
    ------------------------------
    ARIMA.d ≥ 1             → высокая
    KernelSynth mean_a ≠ 0  → высокая (линейный тренд)
    KernelSynth mean_c ≠ 0  → высокая (экспоненциальный тренд)
    ETS с трендом (AAN/AAA) → высокая
    TSI (затухающие моды)   → средняя
    ARIMA (d=0, малые AR)   → низкая

    Parameters
    ----------
    x       : np.ndarray, shape (B, T)
    periods : tuple[int, ...], default (4, 7, 12, 24, 52)

    Returns
    -------
    np.ndarray, shape (B,)
        Значения в [0, 1]. NaN если ни один период не применим.

    Examples
    --------
    >>> T = 200
    >>> t = np.arange(T, dtype=float)
    >>> trend = (t / T)[np.newaxis, :]
    >>> trend_strength(trend)   # ≈ высокое значение
    >>> rng = np.random.default_rng(0)
    >>> white = rng.standard_normal((1, T))
    >>> trend_strength(white)   # ≈ низкое значение
    """
    x = _validate_input(x)
    B = x.shape[0]
    result = np.empty(B, dtype=np.float64)

    for b in range(B):
        xi = x[b]
        if np.any(np.isnan(xi)):
            result[b] = np.nan
            continue

        best = np.nan
        for m in periods:
            _, ft = _stl_strengths(xi, m)
            if np.isfinite(ft):
                best = ft if np.isnan(best) else max(best, ft)

        result[b] = best

    return result