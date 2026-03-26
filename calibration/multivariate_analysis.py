"""
multivariate_analysis.py — Анализ многомерных временных рядов.

Модуль расширяет возможности калибровки генератора синтетических данных:

1. Вычисление многомерных статистик (сигнатуры, кросс-корреляции,
   площади Леви, ранг корреляционной матрицы)
2. Сжатие ультрамногомерных данных через случайные проекции
3. Сравнение распределений многомерных рядов (Wasserstein, MMD)

Задача — сравнить синтетические данные с реальными по многомерным
характеристикам и понять, корректно ли работает генератор.

Использование
-------------
    from calibration.multivariate_analysis import (
        MultivariateAnalyzer,
        SignatureConfig,
        CrossCorrelationConfig,
        ProjectionConfig,
    )

    analyzer = MultivariateAnalyzer()

    # Базовый анализ
    results = analyzer.analyze_batch(
        synthetic_data,  # (B, T, D)
        real_data,       # (B, T, D)
        signature_config=SignatureConfig(depth=3),
        cross_corr_config=CrossCorrelationConfig(max_lag=10),
    )

    # Для ультрамногомерных данных (D > 10)
    results = analyzer.analyze_with_projections(
        synthetic_data,
        real_data,
        proj_dim=5,
        n_projections=10,
    )

Бэкенды сигнатур
-----------------
Модуль автоматически выбирает доступный бэкенд:
    1. iisignature  — pip install iisignature (или conda install -c conda-forge iisignature)
    2. signatory    — pip install signatory (требует PyTorch)
    3. esig         — pip install esig
    4. manual       — встроенная реализация (без внешних зависимостей, только полная сигнатура)

Если iisignature не ставится через pip:
    - conda install -c conda-forge iisignature
    - git clone https://github.com/bottler/iisignature.git && cd iisignature && pip install .
    - Или используйте backend='manual' / backend='esig'
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from math import gcd
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


# =============================================================================
# Конфигурации
# =============================================================================


@dataclass
class SignatureConfig:
    """
    Конфигурация для вычисления сигнатур.

    Parameters
    ----------
    depth : int, default 3
        Глубина сигнатуры.
        Рекомендации: D<=5 → depth=3, D<=10 → depth=2, D>10 → depth=2.
    use_log_signature : bool, default True
        Использовать ли лог-сигнатуру (компактнее) вместо полной.
    normalize : bool, default True
        Z-score нормализация перед вычислением.
    backend : str, default 'auto'
        Бэкенд: 'iisignature', 'signatory', 'esig', 'manual', или 'auto'.
    """

    depth: int = 3
    use_log_signature: bool = True
    normalize: bool = True
    backend: str = "auto"


@dataclass
class CrossCorrelationConfig:
    """
    Конфигурация для кросс-корреляционного анализа.

    Parameters
    ----------
    max_lag : int, default 10
        Максимальный лаг для кросс-корреляции.
    compute_full_matrix : bool, default False
        Вычислять ли полную матрицу (D, D, 2*max_lag+1) или только среднее.
    """

    max_lag: int = 10
    compute_full_matrix: bool = False


@dataclass
class ProjectionConfig:
    """
    Конфигурация для случайных проекций.

    Parameters
    ----------
    proj_dim : int, default 5
        Размерность после проекции.
    n_projections : int, default 10
        Число случайных проекций для усреднения.
    seed : int, default 42
        Сид для воспроизводимости.
    """

    proj_dim: int = 5
    n_projections: int = 10
    seed: int = 42


# =============================================================================
# Вспомогательные функции
# =============================================================================


def _validate_3d(x: np.ndarray) -> np.ndarray:
    """Проверка и приведение к (B, T, D)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[np.newaxis, :, :]
    if x.ndim != 3:
        raise ValueError(f"Ожидается (B, T, D) или (T, D), получено shape={x.shape}")
    x = np.where(np.isinf(x), np.nan, x)
    return x


def _prepare_path(
    xi: np.ndarray, normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Подготовка пути (T, D) к вычислению сигнатуры.

    Parameters
    ----------
    xi : np.ndarray, shape (T, D)
    normalize : bool
        Z-score нормализация.

    Returns
    -------
    np.ndarray или None (если путь вырожден).
    """
    if np.any(np.isnan(xi)):
        return None
    xi = xi.astype(np.float64)
    if xi.shape[0] < 2:
        return None
    if normalize:
        mu = xi.mean(axis=0, keepdims=True)
        sigma = xi.std(axis=0, ddof=1, keepdims=True)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        xi = (xi - mu) / sigma
    if np.any(np.isnan(xi)) or np.any(np.isinf(xi)):
        return None
    return xi


def _sig_length(D: int, depth: int) -> int:
    """
    Число компонент полной сигнатуры: sum_{k=1}^{depth} D^k.
    """
    return sum(D**k for k in range(1, depth + 1))


def _mobius(n: int) -> int:
    """Функция Мёбиуса μ(n)."""
    if n == 1:
        return 1
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count > 1:
                return 0
            factors.append(d)
        d += 1
    if temp > 1:
        factors.append(temp)
    return (-1) ** len(factors)


def _witt_necklace(D: int, k: int) -> int:
    """
    Размерность k-й компоненты свободной алгебры Ли (формула Витта).

    dim(L_k) = (1/k) * sum_{d | k} μ(k/d) * D^d
    """
    if k == 0:
        return 0
    total = 0
    for d in range(1, k + 1):
        if k % d == 0:
            total += _mobius(k // d) * (D**d)
    return total // k


def _logsig_length(D: int, depth: int) -> int:
    """
    Число компонент лог-сигнатуры (размерность свободной алгебры Ли до уровня depth).

    Используется формула Витта: sum_{k=1}^{depth} (1/k) sum_{d|k} μ(k/d) D^d.
    """
    return sum(_witt_necklace(D, k) for k in range(1, depth + 1))


def _output_length(D: int, depth: int, use_log: bool) -> int:
    """Длина выходного вектора сигнатуры или лог-сигнатуры."""
    if use_log:
        return _logsig_length(D, depth)
    else:
        return _sig_length(D, depth)


# =============================================================================
# Бэкенды сигнатур
# =============================================================================


def _detect_backend() -> str:
    """Автоматическое определение доступного бэкенда."""
    for name in ("iisignature", "signatory", "esig"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return "manual"


def _compute_sig_iisignature(
    x: np.ndarray,
    depth: int,
    normalize: bool,
    use_log: bool,
) -> np.ndarray:
    """Бэкенд: iisignature."""
    import iisignature

    B, T, D = x.shape
    s = iisignature.prepare(D, depth)

    if use_log:
        K = iisignature.logsiglength(D, depth)
    else:
        K = iisignature.siglength(D, depth)

    result = np.full((B, K), np.nan, dtype=np.float64)

    for b in range(B):
        path = _prepare_path(x[b], normalize=normalize)
        if path is None:
            continue
        try:
            if use_log:
                result[b] = iisignature.logsig(path, s)
            else:
                result[b] = iisignature.sig(path, depth)
        except Exception as e:
            warnings.warn(
                f"iisignature: ошибка batch {b}: {e}", stacklevel=3
            )

    return result


def _compute_sig_esig(
    x: np.ndarray,
    depth: int,
    normalize: bool,
    use_log: bool,
) -> np.ndarray:
    """Бэкенд: esig."""
    import esig

    B, T, D = x.shape

    if use_log:
        K = _logsig_length(D, depth)
    else:
        K = _sig_length(D, depth)

    result = np.full((B, K), np.nan, dtype=np.float64)

    for b in range(B):
        path = _prepare_path(x[b], normalize=normalize)
        if path is None:
            continue
        try:
            if use_log:
                sig_vec = esig.stream2logsig(path, depth)
            else:
                sig_vec = esig.stream2sig(path, depth)
            # esig может вернуть вектор с нулевым элементом (1) в начале для полной сигнатуры
            if not use_log and len(sig_vec) == K + 1:
                sig_vec = sig_vec[1:]
            result[b, : len(sig_vec)] = sig_vec[:K]
        except Exception as e:
            warnings.warn(f"esig: ошибка batch {b}: {e}", stacklevel=3)

    return result


def _compute_sig_signatory(
    x: np.ndarray,
    depth: int,
    normalize: bool,
    use_log: bool,
) -> np.ndarray:
    """Бэкенд: signatory (PyTorch)."""
    import torch
    import signatory

    B, T, D = x.shape

    paths = []
    valid_mask = np.ones(B, dtype=bool)
    for b in range(B):
        path = _prepare_path(x[b], normalize=normalize)
        if path is None:
            valid_mask[b] = False
            paths.append(np.zeros((T, D), dtype=np.float64))
        else:
            paths.append(path)

    tensor = torch.tensor(np.stack(paths), dtype=torch.float64)

    try:
        if use_log:
            out = signatory.logsignature(tensor, depth=depth).numpy()
        else:
            out = signatory.signature(tensor, depth=depth).numpy()
    except Exception as e:
        warnings.warn(f"signatory: ошибка — {e}", stacklevel=3)
        K = _output_length(D, depth, use_log)
        return np.full((B, K), np.nan, dtype=np.float64)

    K = out.shape[1]
    result = np.full((B, K), np.nan, dtype=np.float64)
    result[valid_mask] = out[valid_mask]
    return result


def _manual_signature_single(path: np.ndarray, depth: int) -> np.ndarray:
    """
    Вычисление полной сигнатуры одного пути итеративно.

    path : shape (T, D) — подготовленный путь.

    Рекурсия (дискретная):
        S^{i1,...,ik}_{0,t+1} = S^{i1,...,ik}_{0,t}
                                + S^{i1,...,ik-1}_{0,t} * dX^{ik}_{t}

    Реализуем обновление уровней от высшего к низшему на каждом шаге,
    чтобы не перезатирать данные.
    """
    T, D = path.shape
    increments = np.diff(path, axis=0)  # (T-1, D)

    # prev_levels[k] — текущее значение сигнатуры уровня k, shape (D^k,)
    prev_levels: Dict[int, np.ndarray] = {}
    for k in range(1, depth + 1):
        prev_levels[k] = np.zeros(D**k, dtype=np.float64)

    for t_idx in range(T - 1):
        dX = increments[t_idx]  # (D,)

        # Обновляем от высшего уровня к низшему
        for k in range(depth, 1, -1):
            # S^{I, j} += S^{I} ⊗ dX
            outer = np.outer(prev_levels[k - 1], dX).ravel()
            prev_levels[k] = prev_levels[k] + outer

        # Уровень 1
        prev_levels[1] = prev_levels[1] + dX

    return np.concatenate([prev_levels[k] for k in range(1, depth + 1)])


def _compute_sig_manual(
    x: np.ndarray,
    depth: int,
    normalize: bool,
    use_log: bool,
) -> np.ndarray:
    """
    Ручная реализация (без внешних библиотек).

    Поддерживается только полная сигнатура.
    Если запрошена лог-сигнатура — выдаётся предупреждение
    и считается полная.
    """
    B, T, D = x.shape

    actual_use_log = False
    if use_log:
        warnings.warn(
            "Ручной бэкенд не поддерживает лог-сигнатуру. "
            "Вычисляется полная сигнатура.",
            stacklevel=3,
        )

    K = _sig_length(D, depth)
    result = np.full((B, K), np.nan, dtype=np.float64)

    for b in range(B):
        path = _prepare_path(x[b], normalize=normalize)
        if path is None:
            continue
        try:
            result[b] = _manual_signature_single(path, depth)
        except Exception as e:
            warnings.warn(f"manual: ошибка batch {b}: {e}", stacklevel=3)

    return result


# Реестр бэкендов
_BACKENDS = {
    "iisignature": _compute_sig_iisignature,
    "signatory": _compute_sig_signatory,
    "esig": _compute_sig_esig,
    "manual": _compute_sig_manual,
}


# =============================================================================
# Основная функция вычисления сигнатуры
# =============================================================================


def compute_signature(
    x: np.ndarray,
    depth: int = 3,
    normalize: bool = True,
    use_log_signature: bool = True,
    backend: str = "auto",
    config: Optional[SignatureConfig] = None,
) -> np.ndarray:
    """
    Сигнатура или лог-сигнатура многомерного ряда.

    Parameters
    ----------
    x : np.ndarray, shape (B, T, D) или (T, D)
        Входные данные.
    depth : int
        Глубина сигнатуры.
    normalize : bool
        Z-score нормализация.
    use_log_signature : bool
        Использовать лог-сигнатуру (компактнее).
    backend : str
        'iisignature', 'signatory', 'esig', 'manual', или 'auto'.
    config : SignatureConfig, optional
        Если задан, параметры берутся из конфига (приоритет над аргументами).

    Returns
    -------
    np.ndarray, shape (B, K)
        K зависит от D, depth и типа сигнатуры.
    """
    if config is not None:
        depth = config.depth
        normalize = config.normalize
        use_log_signature = config.use_log_signature
        backend = config.backend

    x = _validate_3d(x)
    B, T, D = x.shape

    if backend == "auto":
        backend = _detect_backend()

    if backend not in _BACKENDS:
        raise ValueError(
            f"Неизвестный бэкенд: '{backend}'. "
            f"Доступные: {list(_BACKENDS.keys())}"
        )

    # Пробуем запрошенный бэкенд, при ошибке — fallback на manual
    try:
        return _BACKENDS[backend](x, depth, normalize, use_log_signature)
    except ImportError:
        warnings.warn(
            f"Бэкенд '{backend}' не установлен. Переключаемся на 'manual'.",
            stacklevel=2,
        )
        return _BACKENDS["manual"](x, depth, normalize, use_log_signature)


# =============================================================================
# Кросс-корреляции
# =============================================================================


def compute_cross_correlations(
    x: np.ndarray,
    max_lag: int = 10,
    compute_full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Кросс-корреляции между каналами при разных лагах.

    Parameters
    ----------
    x : np.ndarray, shape (B, T, D)
    max_lag : int
        Максимальный лаг.
    compute_full : bool
        Полная матрица корреляций по лагам или только скалярное среднее.

    Returns
    -------
    Если compute_full=False:
        np.ndarray, shape (B, 1) — средняя абсолютная кросс-корреляция
        (усреднение по всем парам каналов и всем лагам).
    Если compute_full=True:
        (mean_corr, corr_tensor) где:
        - mean_corr : shape (B, 1)
        - corr_tensor : shape (B, D, D, 2*max_lag+1)
    """
    x = _validate_3d(x)
    B, T, D = x.shape

    mean_result = np.full((B, 1), np.nan, dtype=np.float64)
    n_lags = 2 * max_lag + 1

    if D < 2:
        if compute_full:
            return mean_result, np.full((B, 1, 1, n_lags), np.nan)
        return mean_result

    full_tensor = (
        np.full((B, D, D, n_lags), np.nan, dtype=np.float64)
        if compute_full
        else None
    )

    for b in range(B):
        xi = x[b]  # (T, D)
        if np.any(np.isnan(xi)):
            continue

        # Z-score нормализация каждого канала
        mu = xi.mean(axis=0, keepdims=True)
        sigma = xi.std(axis=0, ddof=1, keepdims=True)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        xi_norm = (xi - mu) / sigma

        all_abs_corr = []

        for lag_idx, lag in enumerate(range(-max_lag, max_lag + 1)):
            if lag >= 0:
                x_part = xi_norm[: T - lag] if lag > 0 else xi_norm
                y_part = xi_norm[lag:] if lag > 0 else xi_norm
            else:
                x_part = xi_norm[-lag:]
                y_part = xi_norm[: T + lag]

            n_overlap = x_part.shape[0]
            if n_overlap < 2:
                continue

            # Корреляционная матрица при данном лаге
            # corr[i, j] = (1/n) * sum_t x_i(t) * y_j(t)  (уже z-scored)
            corr_matrix = (x_part.T @ y_part) / n_overlap  # (D, D)

            if compute_full and full_tensor is not None:
                full_tensor[b, :, :, lag_idx] = corr_matrix

            # Собираем верхний треугольник (пары i < j) для среднего
            idx_upper = np.triu_indices(D, k=1)
            all_abs_corr.append(np.abs(corr_matrix[idx_upper]))

        if all_abs_corr:
            mean_result[b, 0] = np.mean(np.concatenate(all_abs_corr))

    if compute_full:
        return mean_result, full_tensor
    return mean_result


# =============================================================================
# Ранг матрицы корреляций
# =============================================================================


def compute_correlation_rank(
    x: np.ndarray,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Нормированный ранг матрицы корреляций.

    Parameters
    ----------
    x : np.ndarray, shape (B, T, D)
    tol : float
        Порог для ненулевых собственных значений (относительно максимального).

    Returns
    -------
    np.ndarray, shape (B, 1)
        rank(Corr) / D для каждого элемента батча.
    """
    x = _validate_3d(x)
    B, T, D = x.shape
    result = np.full((B, 1), np.nan, dtype=np.float64)

    if D < 2:
        result[:] = 1.0
        return result

    for b in range(B):
        xi = x[b]
        if np.any(np.isnan(xi)):
            continue
        try:
            corr = np.corrcoef(xi.T)  # (D, D)
            eigvals = np.linalg.eigvalsh(corr)
            max_eig = eigvals.max()
            if max_eig < 1e-15:
                continue
            threshold = tol * max_eig
            rank = int((eigvals > threshold).sum())
            result[b, 0] = rank / D
        except np.linalg.LinAlgError:
            pass

    return result


def compute_correlation_condition_and_eff_rank(
    x: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Condition number и эффективный ранг матрицы корреляций.

    Parameters
    ----------
    x : np.ndarray, shape (B, T, D)
    tol : float
        Порог для малых собственных значений.

    Returns
    -------
    (condition_number, effective_rank) : Tuple[np.ndarray, np.ndarray]
        condition_number : shape (B, 1) - число обусловленности
        effective_rank : shape (B, 1) - эффективный ранг
    """
    x = _validate_3d(x)
    B, T, D = x.shape
    cond_result = np.full((B, 1), np.nan, dtype=np.float64)
    eff_rank_result = np.full((B, 1), np.nan, dtype=np.float64)

    if D < 2:
        cond_result[:] = 1.0
        eff_rank_result[:] = 1.0
        return cond_result, eff_rank_result

    for b in range(B):
        xi = x[b]
        if np.any(np.isnan(xi)):
            continue
        try:
            corr = np.corrcoef(xi.T)  # (D, D)
            eigvals = np.linalg.eigvalsh(corr)
            eigvals = np.sort(eigvals)[::-1]  # По убыванию

            max_eig = eigvals.max()
            min_eig = eigvals[eigvals > tol * max_eig].min() if np.any(eigvals > tol * max_eig) else tol

            # Condition number
            cond_result[b, 0] = max_eig / min_eig if min_eig > 0 else float('inf')

            # Effective rank: exp(entropy) = exp(-sum(p_i * log(p_i)))
            # где p_i = lambda_i / sum(lambda_j)
            total = eigvals.sum()
            if total > 0:
                p = eigvals / total
                p = p[p > 0]  # Убираем нули
                entropy = -np.sum(p * np.log(p + 1e-12))
                eff_rank_result[b, 0] = np.exp(entropy)
        except np.linalg.LinAlgError:
            pass

    return cond_result, eff_rank_result


# =============================================================================
# Площади Леви
# =============================================================================


def compute_levy_area(
    x: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Площади Леви (антисимметричная часть уровня-2 сигнатуры).

    Для пары каналов (i, j), i < j:
        A^{i,j} = (1/2) * (S^{i,j} - S^{j,i})
                = (1/2) * sum_{r<s} (dX^i_r * dX^j_s - dX^j_r * dX^i_s)

    Parameters
    ----------
    x : np.ndarray, shape (B, T, D)
    normalize : bool
        Z-score нормализация.

    Returns
    -------
    np.ndarray, shape (B, D*(D-1)//2)
    """
    x = _validate_3d(x)
    B, T, D = x.shape
    K = D * (D - 1) // 2
    result = np.full((B, K), np.nan, dtype=np.float64)

    if D < 2:
        return result

    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]

    for b in range(B):
        path = _prepare_path(x[b], normalize=normalize)
        if path is None:
            continue

        dx = np.diff(path, axis=0)  # (T-1, D)
        # Кумулятивная сумма приращений = путь - путь[0], но нам нужны
        # S^{i,j} = sum_{r<s} dX^i_r dX^j_s
        # = sum_s ( sum_{r<s} dX^i_r ) * dX^j_s
        # = sum_s  cumsum_i(s-1) * dX^j_s

        n_steps = dx.shape[0]
        # cumsum[t, i] = sum_{r=0}^{t-1} dX^i_r  (сдвинутая кумулятивная сумма)
        cumsum_dx = np.zeros_like(dx)
        cumsum_dx[0] = 0.0
        if n_steps > 1:
            cumsum_dx[1:] = np.cumsum(dx[:-1], axis=0)

        for k, (i, j) in enumerate(pairs):
            # S^{i,j} = sum_s cumsum_i[s] * dX^j[s]
            s_ij = np.sum(cumsum_dx[:, i] * dx[:, j])
            s_ji = np.sum(cumsum_dx[:, j] * dx[:, i])
            result[b, k] = 0.5 * (s_ij - s_ji)

    return result


# =============================================================================
# Дополнительные сигнатурные статистики
# =============================================================================


def compute_signature_energy_by_level(
    sig: np.ndarray,
    D: int,
    depth: int,
) -> np.ndarray:
    """
    Энергия сигнатуры по уровням.

    Вычисляет ||sig_level_k||^2 для каждого уровня k=1..depth.

    Parameters
    ----------
    sig : np.ndarray, shape (B, K)
        Сигнатура (полная, не лог).
    D : int
        Размерность исходного пространства.
    depth : int
        Глубина сигнатуры.

    Returns
    -------
    np.ndarray, shape (B, depth)
        Энергия по уровням [E_1, E_2, ..., E_depth].
    """
    B = sig.shape[0]
    energies = np.zeros((B, depth), dtype=np.float64)
    idx = 0

    for k in range(1, depth + 1):
        level_dim = D ** k
        level_sig = sig[:, idx:idx + level_dim]
        # Энергия = сумма квадратов компонент уровня
        energies[:, k - 1] = np.sum(level_sig ** 2, axis=1)
        idx += level_dim

    return energies


def compute_signature_entropy(sig: np.ndarray) -> np.ndarray:
    """
    Энтропия сигнатуры.

    H = -Σ p_i log(p_i), где p_i = |sig_i| / Σ|sig_j|

    Parameters
    ----------
    sig : np.ndarray, shape (B, K)
        Сигнатура.

    Returns
    -------
    np.ndarray, shape (B,)
        Энтропия для каждого элемента батча.
    """
    B = sig.shape[0]
    abs_sig = np.abs(sig)
    sum_abs = abs_sig.sum(axis=1, keepdims=True) + 1e-12
    p = abs_sig / sum_abs

    # Энтропия
    entropy = -np.sum(p * np.log(p + 1e-12), axis=1)

    # Нормализация на максимальную энтропию (log(K))
    K = sig.shape[1]
    entropy_normalized = entropy / np.log(K) if K > 1 else entropy

    return entropy_normalized


def compute_signature_rotation_number(levy_area: np.ndarray, D: int) -> np.ndarray:
    """
    Число вращения (Rotation Number) через площади Леви.

    RN = (1/π) * sqrt(Σ A^{i,j}²)  # сумма по всем парам (i,j)

    Parameters
    ----------
    levy_area : np.ndarray, shape (B, D*(D-1)//2)
        Площади Леви.
    D : int
        Размерность пространства.

    Returns
    -------
    np.ndarray, shape (B,)
        Число вращения для каждого элемента батча.
    """
    # Сумма квадратов всех площадей
    sum_sq = np.sum(levy_area ** 2, axis=1)
    rotation_number = np.sqrt(sum_sq) / np.pi
    return rotation_number


def compute_signature_brownian_distance(
    sig: np.ndarray,
    D: int,
    depth: int,
    T: float = 1.0,
) -> np.ndarray:
    """
    Расстояние до сигнатуры стандартного броуновского движения.

    Для БД:
    - Уровень 1: E[S^i] = 0
    - Уровень 2: E[S^{i,j}] = 0 для i≠j, E[S^{i,i}] = T/2

    Parameters
    ----------
    sig : np.ndarray, shape (B, K)
        Сигнатура.
    D : int
        Размерность пространства.
    depth : int
        Глубина сигнатуры.
    T : float
        Время наблюдения (по умолчанию 1.0).

    Returns
    -------
    np.ndarray, shape (B,)
        Расстояние до БД для каждого элемента батча.
    """
    B = sig.shape[0]
    distances = np.zeros(B, dtype=np.float64)
    idx = 0

    for k in range(1, depth + 1):
        level_dim = D ** k
        level_sig = sig[:, idx:idx + level_dim]

        if k == 1:
            # Уровень 1: ожидаем 0
            distances += np.sum(level_sig ** 2, axis=1)
        elif k == 2:
            # Уровень 2: диагональ = T/2, остальное = 0
            for i in range(D):
                for j in range(D):
                    flat_idx = i * D + j
                    if i == j:
                        # Диагональ: отклонение от T/2
                        distances += (level_sig[:, flat_idx] - T / 2) ** 2
                    else:
                        # Не диагональ: отклонение от 0
                        distances += level_sig[:, flat_idx] ** 2
        else:
            # Уровни 3+: ожидаем 0
            distances += np.sum(level_sig ** 2, axis=1)

        idx += level_dim

    return np.sqrt(distances)


def compute_signature_gram_eigenvalues(
    sig: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Топ-k собственных значений матрицы Грама сигнатур.

    G = S @ S.T, где S — матрица сигнатур батча.

    Parameters
    ----------
    sig : np.ndarray, shape (B, K)
        Сигнатура.
    k : int
        Число собственных значений.

    Returns
    -------
    np.ndarray, shape (B, k)
        Топ-k собственных значений (повторяется для каждого элемента).
    """
    B, K = sig.shape

    # Матрица Грама
    G = sig @ sig.T  # (B, B)

    # Собственные значения
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.sort(eigvals)[::-1]  # По убыванию

    # Берём топ-k
    top_k = eigvals[:k]

    # Возвращаем для каждого элемента (одинаково)
    return np.tile(top_k, (B, 1))


def compute_signature_predictability(
    sig_past: np.ndarray,
    sig_future: np.ndarray,
    max_lag: int = 5,
) -> float:
    """
    Мера предсказуемости через сигнатуры.

    Вычисляет R² между сигнатурой прошлого и будущего:
    S_future ≈ f(S_past)

    Parameters
    ----------
    sig_past : np.ndarray, shape (B, K)
        Сигнатура прошлого.
    sig_future : np.ndarray, shape (B, K)
        Сигнатура будущего.
    max_lag : int
        Максимальный лаг для кросс-корреляции.

    Returns
    -------
    float
        R² ∈ [0, 1]. 1 = высокая предсказуемость.
    """
    # Простая линейная регрессия
    from scipy.stats import linregress

    # Усредняем по компонентам сигнатуры
    past_mean = np.mean(sig_past, axis=1)
    future_mean = np.mean(sig_future, axis=1)

    # Линейная регрессия
    slope, intercept, r_value, p_value, std_err = linregress(
        past_mean, future_mean
    )

    return r_value ** 2


# =============================================================================
# Сигнатурный лосс (MMD)
# =============================================================================


def signature_mmd_loss(
    x_real: np.ndarray,
    x_fake: np.ndarray,
    depth: int = 3,
    normalize: bool = True,
    use_log_signature: bool = True,
    backend: str = "auto",
    config: Optional[SignatureConfig] = None,
) -> float:
    r"""
    Сигнатурный MMD-лосс (разность средних сигнатур).

    .. math::
        \mathcal{L}_{\mathrm{Sig}}
        = \left\| \frac{1}{B_1}\sum_{i} S(X_i)
                - \frac{1}{B_2}\sum_{j} S(Y_j) \right\|^2

    Parameters
    ----------
    x_real : np.ndarray, shape (B1, T, D)
    x_fake : np.ndarray, shape (B2, T, D)

    Returns
    -------
    float
        Значение лосса (>= 0, равно 0 ⟺ средние сигнатуры совпадают).
    """
    sig_real = compute_signature(
        x_real,
        depth=depth,
        normalize=normalize,
        use_log_signature=use_log_signature,
        backend=backend,
        config=config,
    )
    sig_fake = compute_signature(
        x_fake,
        depth=depth,
        normalize=normalize,
        use_log_signature=use_log_signature,
        backend=backend,
        config=config,
    )

    # Убираем NaN-строки
    sig_real = sig_real[~np.any(np.isnan(sig_real), axis=1)]
    sig_fake = sig_fake[~np.any(np.isnan(sig_fake), axis=1)]

    if len(sig_real) == 0 or len(sig_fake) == 0:
        return float("nan")

    mean_real = sig_real.mean(axis=0)
    mean_fake = sig_fake.mean(axis=0)

    return float(np.sum((mean_real - mean_fake) ** 2))


def signature_kernel_mmd_loss(
    x_real: np.ndarray,
    x_fake: np.ndarray,
    depth: int = 3,
    normalize: bool = True,
    use_log_signature: bool = True,
    backend: str = "auto",
    config: Optional[SignatureConfig] = None,
) -> float:
    r"""
    Полный сигнатурный MMD-лосс с линейным ядром.

    .. math::
        \mathrm{MMD}^2_k(\mu, \nu)
        = \mathbb{E}[k(X,X')] - 2\,\mathbb{E}[k(X,Y)] + \mathbb{E}[k(Y,Y')]

    где :math:`k(X,Y) = \langle S(X), S(Y) \rangle` (линейное ядро на сигнатурах).

    Используется несмещённая оценка.

    Parameters
    ----------
    x_real : np.ndarray, shape (B1, T, D)
    x_fake : np.ndarray, shape (B2, T, D)

    Returns
    -------
    float
    """
    sig_real = compute_signature(
        x_real,
        depth=depth,
        normalize=normalize,
        use_log_signature=use_log_signature,
        backend=backend,
        config=config,
    )
    sig_fake = compute_signature(
        x_fake,
        depth=depth,
        normalize=normalize,
        use_log_signature=use_log_signature,
        backend=backend,
        config=config,
    )

    sig_real = sig_real[~np.any(np.isnan(sig_real), axis=1)]
    sig_fake = sig_fake[~np.any(np.isnan(sig_fake), axis=1)]

    if len(sig_real) < 2 or len(sig_fake) < 2:
        return float("nan")

    n, m = len(sig_real), len(sig_fake)

    # Gram-матрицы
    K_xx = sig_real @ sig_real.T  # (n, n)
    K_yy = sig_fake @ sig_fake.T  # (m, m)
    K_xy = sig_real @ sig_fake.T  # (n, m)

    # Несмещённая оценка MMD^2
    mmd2 = (
        (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
        + (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
        - 2.0 * K_xy.sum() / (n * m)
    )

    return float(max(mmd2, 0.0))


# =============================================================================
# Проекционный анализ
# =============================================================================


def analyze_with_projections(
    synthetic: np.ndarray,
    real: np.ndarray,
    proj_dim: int = 5,
    n_projections: int = 10,
    depth: int = 3,
    seed: int = 42,
    use_log_signature: bool = True,
    backend: str = "auto",
) -> Dict[str, np.ndarray]:
    """
    Анализ через случайные проекции для ультрамногомерных данных.

    Вместо работы с D-мерным пространством (D >> 10) проецируем данные
    на случайные подпространства размерности proj_dim и вычисляем
    сигнатуры там. Усредняем по n_projections проекциям.

    Parameters
    ----------
    synthetic : np.ndarray, shape (B_s, T, D)
    real : np.ndarray, shape (B_r, T, D)
    proj_dim : int
        Размерность проекции.
    n_projections : int
        Число проекций.
    depth : int
        Глубина сигнатуры.
    seed : int
        Сид для воспроизводимости.
    use_log_signature : bool
    backend : str

    Returns
    -------
    dict
    """
    synthetic = _validate_3d(synthetic)
    real = _validate_3d(real)

    B_s, T, D = synthetic.shape
    B_r, T_r, D_r = real.shape

    if D != D_r:
        raise ValueError(f"D не совпадает: synthetic={D}, real={D_r}")

    rng = np.random.default_rng(seed)

    # Случайные ортогональные проекции (через QR)
    projections = []
    for _ in range(n_projections):
        M = rng.standard_normal((D, proj_dim))
        Q, _ = np.linalg.qr(M)
        projections.append(Q[:, :proj_dim])  # (D, proj_dim)

    K = _output_length(proj_dim, depth, use_log_signature)

    synth_sig_mean = np.full((n_projections, K), np.nan)
    real_sig_mean = np.full((n_projections, K), np.nan)
    synth_sig_std = np.full((n_projections, K), np.nan)
    real_sig_std = np.full((n_projections, K), np.nan)
    wasserstein_dists = np.full((n_projections, K), np.nan)

    for p_idx, P in enumerate(projections):
        synth_proj = synthetic @ P  # (B_s, T, proj_dim)
        real_proj = real @ P  # (B_r, T, proj_dim)

        try:
            s_sigs = compute_signature(
                synth_proj,
                depth=depth,
                normalize=True,
                use_log_signature=use_log_signature,
                backend=backend,
            )
            r_sigs = compute_signature(
                real_proj,
                depth=depth,
                normalize=True,
                use_log_signature=use_log_signature,
                backend=backend,
            )

            synth_sig_mean[p_idx] = np.nanmean(s_sigs, axis=0)
            synth_sig_std[p_idx] = np.nanstd(s_sigs, axis=0)
            real_sig_mean[p_idx] = np.nanmean(r_sigs, axis=0)
            real_sig_std[p_idx] = np.nanstd(r_sigs, axis=0)

            for k in range(K):
                sv = s_sigs[:, k]
                rv = r_sigs[:, k]
                sv = sv[np.isfinite(sv)]
                rv = rv[np.isfinite(rv)]
                if len(sv) > 0 and len(rv) > 0:
                    wasserstein_dists[p_idx, k] = wasserstein_distance(sv, rv)

        except Exception as e:
            warnings.warn(
                f"Проекция {p_idx}: ошибка — {e}", stacklevel=2
            )

    return {
        "synthetic_sig_mean": synth_sig_mean,
        "real_sig_mean": real_sig_mean,
        "synthetic_sig_std": synth_sig_std,
        "real_sig_std": real_sig_std,
        "wasserstein_distances": wasserstein_dists,
        "mean_wasserstein": float(np.nanmean(wasserstein_dists)),
        "projection_matrices": np.stack(projections),
    }


# =============================================================================
# Перебор 5-мерных комбинаций
# =============================================================================


def analyze_all_5d_combinations(
    synthetic: np.ndarray,
    real: np.ndarray,
    depth: int = 3,
    seed: int = 42,
    use_log_signature: bool = True,
    backend: str = "auto",
    max_combinations: int = 500,
) -> Dict[str, np.ndarray]:
    """
    Анализ 5-мерных комбинаций каналов.

    Для D-мерных данных перебираем сочетания из 5 каналов
    и вычисляем сигнатуры для каждого. Если C(D,5) > max_combinations,
    берём случайную подвыборку.

    Parameters
    ----------
    synthetic : np.ndarray, shape (B_s, T, D)
    real : np.ndarray, shape (B_r, T, D)
    depth : int
    seed : int
    use_log_signature : bool
    backend : str
    max_combinations : int
        Максимальное число комбинаций (для больших D).

    Returns
    -------
    dict
    """
    synthetic = _validate_3d(synthetic)
    real = _validate_3d(real)

    B_s, T, D = synthetic.shape
    B_r, T_r, D_r = real.shape

    if D != D_r:
        raise ValueError(f"D не совпадает: synthetic={D}, real={D_r}")
    if D < 5:
        raise ValueError(f"D должно быть >= 5, получено {D}")

    all_combinations = list(combinations(range(D), 5))
    rng = np.random.default_rng(seed)

    if len(all_combinations) > max_combinations:
        indices = rng.choice(
            len(all_combinations), size=max_combinations, replace=False
        )
        selected_combinations = [all_combinations[i] for i in sorted(indices)]
    else:
        selected_combinations = all_combinations

    n_comb = len(selected_combinations)
    K = _output_length(5, depth, use_log_signature)

    result_synth_sigs = np.full((n_comb, B_s, K), np.nan)
    result_real_sigs = np.full((n_comb, B_r, K), np.nan)
    wasserstein_dists = np.full((n_comb, K), np.nan)
    mean_w_per_comb = np.full(n_comb, np.nan)

    for c_idx, comb in enumerate(selected_combinations):
        comb_list = list(comb)
        synth_sub = synthetic[:, :, comb_list]
        real_sub = real[:, :, comb_list]

        try:
            s_sigs = compute_signature(
                synth_sub,
                depth=depth,
                normalize=True,
                use_log_signature=use_log_signature,
                backend=backend,
            )
            r_sigs = compute_signature(
                real_sub,
                depth=depth,
                normalize=True,
                use_log_signature=use_log_signature,
                backend=backend,
            )

            result_synth_sigs[c_idx] = s_sigs
            result_real_sigs[c_idx] = r_sigs

            for k in range(K):
                sv = s_sigs[:, k]
                rv = r_sigs[:, k]
                sv = sv[np.isfinite(sv)]
                rv = rv[np.isfinite(rv)]
                if len(sv) > 0 and len(rv) > 0:
                    wasserstein_dists[c_idx, k] = wasserstein_distance(sv, rv)

            mean_w_per_comb[c_idx] = np.nanmean(wasserstein_dists[c_idx])

        except Exception as e:
            warnings.warn(
                f"Комбинация {c_idx} {comb}: ошибка — {e}", stacklevel=2
            )

    return {
        "combinations": selected_combinations,
        "synthetic_sigs": result_synth_sigs,
        "real_sigs": result_real_sigs,
        "wasserstein_distances": wasserstein_dists,
        "mean_wasserstein_per_combination": mean_w_per_comb,
        "overall_mean_wasserstein": float(np.nanmean(mean_w_per_comb)),
    }


# =============================================================================
# Класс-анализатор
# =============================================================================


class MultivariateAnalyzer:
    """
    Анализатор многомерных временных рядов.

    Предоставляет единый интерфейс для:
    1. Вычисления многомерных статистик
    2. Сравнения синтетических и реальных данных
    3. Анализа через случайные проекции
    4. Перебора 5-мерных комбинаций
    5. Вычисления сигнатурного MMD-лосса
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._backend = _detect_backend()

    @property
    def backend(self) -> str:
        return self._backend

    def analyze_batch(
        self,
        synthetic: np.ndarray,
        real: np.ndarray,
        signature_config: Optional[SignatureConfig] = None,
        cross_corr_config: Optional[CrossCorrelationConfig] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Базовый анализ многомерных рядов.

        Parameters
        ----------
        synthetic : np.ndarray, shape (B_s, T, D)
        real : np.ndarray, shape (B_r, T, D)
        signature_config : SignatureConfig, optional
        cross_corr_config : CrossCorrelationConfig, optional

        Returns
        -------
        dict с результатами.
        """
        synthetic = _validate_3d(synthetic)
        real = _validate_3d(real)

        B_s, T, D = synthetic.shape
        B_r, T_r, D_r = real.shape

        if D != D_r:
            raise ValueError(f"D не совпадает: synthetic={D}, real={D_r}")

        if signature_config is None:
            signature_config = SignatureConfig(
                depth=min(3, max(2, 5 - D // 5)),
                backend=self._backend,
            )
        if cross_corr_config is None:
            cross_corr_config = CrossCorrelationConfig()

        results: Dict[str, np.ndarray] = {}

        # 1. Сигнатуры
        try:
            synth_sig = compute_signature(
                synthetic, config=signature_config
            )
            real_sig = compute_signature(real, config=signature_config)

            results["synthetic_signature"] = synth_sig
            results["real_signature"] = real_sig

            K = synth_sig.shape[1]
            sig_wasserstein = np.zeros(K)
            for k in range(K):
                sv = synth_sig[:, k]
                rv = real_sig[:, k]
                sv = sv[np.isfinite(sv)]
                rv = rv[np.isfinite(rv)]
                if len(sv) > 0 and len(rv) > 0:
                    sig_wasserstein[k] = wasserstein_distance(sv, rv)

            results["signature_wasserstein"] = sig_wasserstein
            results["mean_signature_wasserstein"] = float(
                np.nanmean(sig_wasserstein)
            )

        except Exception as e:
            warnings.warn(f"Ошибка вычисления сигнатур: {e}", stacklevel=2)

        # 2. Сигнатурный MMD-лосс
        try:
            results["signature_mmd_simple"] = signature_mmd_loss(
                real, synthetic, config=signature_config
            )
            results["signature_mmd_kernel"] = signature_kernel_mmd_loss(
                real, synthetic, config=signature_config
            )
        except Exception as e:
            warnings.warn(f"Ошибка вычисления MMD: {e}", stacklevel=2)

        # 3. Кросс-корреляции
        try:
            synth_cc = compute_cross_correlations(
                synthetic,
                max_lag=cross_corr_config.max_lag,
                compute_full=cross_corr_config.compute_full_matrix,
            )
            real_cc = compute_cross_correlations(
                real,
                max_lag=cross_corr_config.max_lag,
                compute_full=cross_corr_config.compute_full_matrix,
            )

            if cross_corr_config.compute_full_matrix:
                synth_cc_mean, synth_cc_full = synth_cc
                real_cc_mean, real_cc_full = real_cc
                results["synthetic_cross_corr_full"] = synth_cc_full
                results["real_cross_corr_full"] = real_cc_full
            else:
                synth_cc_mean = synth_cc
                real_cc_mean = real_cc

            results["synthetic_cross_corr"] = synth_cc_mean
            results["real_cross_corr"] = real_cc_mean

            sv = synth_cc_mean[:, 0]
            rv = real_cc_mean[:, 0]
            sv = sv[np.isfinite(sv)]
            rv = rv[np.isfinite(rv)]
            if len(sv) > 0 and len(rv) > 0:
                results["cross_corr_wasserstein"] = float(
                    wasserstein_distance(sv, rv)
                )

        except Exception as e:
            warnings.warn(
                f"Ошибка вычисления кросс-корреляций: {e}", stacklevel=2
            )

        # 4. Ранг корреляционной матрицы
        try:
            results["synthetic_rank_ratio"] = compute_correlation_rank(synthetic)
            results["real_rank_ratio"] = compute_correlation_rank(real)
        except Exception as e:
            warnings.warn(f"Ошибка вычисления ранга: {e}", stacklevel=2)

        # 5. Площади Леви
        try:
            results["synthetic_levy_area"] = compute_levy_area(synthetic)
            results["real_levy_area"] = compute_levy_area(real)
        except Exception as e:
            warnings.warn(
                f"Ошибка вычисления площадей Леви: {e}", stacklevel=2
            )

        # 6. Дополнительные сигнатурные метрики
        try:
            # Энергия по уровням
            synth_energy = compute_signature_energy_by_level(
                synth_sig, D, signature_config.depth
            )
            real_energy = compute_signature_energy_by_level(
                real_sig, D, signature_config.depth
            )
            results["synthetic_signature_energy"] = synth_energy
            results["real_signature_energy"] = real_energy

            # Энтропия сигнатуры
            synth_entropy = compute_signature_entropy(synth_sig)
            real_entropy = compute_signature_entropy(real_sig)
            results["synthetic_signature_entropy"] = synth_entropy
            results["real_signature_entropy"] = real_entropy

            # Число вращения (из площадей Леви)
            if "synthetic_levy_area" in results:
                synth_rotation = compute_signature_rotation_number(
                    results["synthetic_levy_area"], D
                )
                real_rotation = compute_signature_rotation_number(
                    results["real_levy_area"], D
                )
                results["synthetic_rotation_number"] = synth_rotation
                results["real_rotation_number"] = real_rotation

            # Расстояние до броуновского движения
            synth_brownian = compute_signature_brownian_distance(
                synth_sig, D, signature_config.depth
            )
            real_brownian = compute_signature_brownian_distance(
                real_sig, D, signature_config.depth
            )
            results["synthetic_brownian_distance"] = synth_brownian
            results["real_brownian_distance"] = real_brownian

            # Собственные значения Грама
            synth_gram = compute_signature_gram_eigenvalues(synth_sig, k=3)
            real_gram = compute_signature_gram_eigenvalues(real_sig, k=3)
            results["synthetic_gram_eigenvalues"] = synth_gram
            results["real_gram_eigenvalues"] = real_gram

        except Exception as e:
            warnings.warn(
                f"Ошибка вычисления доп. сигнатурных метрик: {e}", stacklevel=2
            )

        # 7. Дополнительные метрики кросс-корреляций
        try:
            # Кросс-корреляции по лагам (полная матрица)
            synth_cc_full = compute_cross_correlations(
                synthetic, max_lag=cross_corr_config.max_lag, compute_full=True
            )[1]  # (B, D, D, 2*max_lag+1)
            real_cc_full = compute_cross_correlations(
                real, max_lag=cross_corr_config.max_lag, compute_full=True
            )[1]
            results["synthetic_cross_corr_by_lag"] = synth_cc_full
            results["real_cross_corr_by_lag"] = real_cc_full

        except Exception as e:
            warnings.warn(
                f"Ошибка вычисления кросс-корреляций по лагам: {e}", stacklevel=2
            )

        # 8. Дополнительные метрики ранга
        try:
            # Condition number и effective rank
            synth_cond, synth_eff_rank = compute_correlation_condition_and_eff_rank(
                synthetic
            )
            real_cond, real_eff_rank = compute_correlation_condition_and_eff_rank(real)
            results["synthetic_condition_number"] = synth_cond
            results["real_condition_number"] = real_cond
            results["synthetic_effective_rank"] = synth_eff_rank
            results["real_effective_rank"] = real_eff_rank

        except Exception as e:
            warnings.warn(
                f"Ошибка вычисления condition number: {e}", stacklevel=2
            )

        # 9. Общая площадь Леви
        try:
            synth_total_levy = np.sum(
                np.abs(results["synthetic_levy_area"]), axis=1, keepdims=True
            )
            real_total_levy = np.sum(
                np.abs(results["real_levy_area"]), axis=1, keepdims=True
            )
            results["synthetic_total_levy_area"] = synth_total_levy
            results["real_total_levy_area"] = real_total_levy

        except Exception as e:
            warnings.warn(
                f"Ошибка вычисления общей площади Леви: {e}", stacklevel=2
            )

        return results

    def analyze_with_projections(
        self,
        synthetic: np.ndarray,
        real: np.ndarray,
        proj_dim: int = 5,
        n_projections: int = 10,
        depth: int = 3,
    ) -> Dict[str, np.ndarray]:
        """Анализ через случайные проекции."""
        return analyze_with_projections(
            synthetic,
            real,
            proj_dim=proj_dim,
            n_projections=n_projections,
            depth=depth,
            seed=self.seed,
            backend=self._backend,
        )

    def analyze_all_5d_combinations(
        self,
        synthetic: np.ndarray,
        real: np.ndarray,
        depth: int = 3,
        max_combinations: int = 500,
    ) -> Dict[str, np.ndarray]:
        """Анализ 5-мерных комбинаций."""
        return analyze_all_5d_combinations(
            synthetic,
            real,
            depth=depth,
            seed=self.seed,
            backend=self._backend,
            max_combinations=max_combinations,
        )

    def compute_summary(
        self,
        results: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Сводные метрики из результатов анализа.

        Returns
        -------
        dict с ключами:
            - mean_signature_wasserstein
            - signature_mmd_simple
            - signature_mmd_kernel
            - cross_corr_wasserstein
            - rank_ratio_diff
            - levy_area_wasserstein
            - overall_multivariate_distance
        """
        summary: Dict[str, float] = {}

        # Сигнатуры
        if "mean_signature_wasserstein" in results:
            summary["mean_signature_wasserstein"] = float(
                results["mean_signature_wasserstein"]
            )
        elif "mean_wasserstein" in results:
            summary["mean_signature_wasserstein"] = float(
                results["mean_wasserstein"]
            )

        # MMD
        for key in ("signature_mmd_simple", "signature_mmd_kernel"):
            if key in results:
                summary[key] = float(results[key])

        # Кросс-корреляции
        if "cross_corr_wasserstein" in results:
            summary["cross_corr_wasserstein"] = float(
                results["cross_corr_wasserstein"]
            )

        # Ранг
        if "synthetic_rank_ratio" in results and "real_rank_ratio" in results:
            sr = np.nanmean(results["synthetic_rank_ratio"])
            rr = np.nanmean(results["real_rank_ratio"])
            summary["rank_ratio_diff"] = float(abs(sr - rr))

        # Площади Леви
        if "synthetic_levy_area" in results and "real_levy_area" in results:
            sl = results["synthetic_levy_area"]
            rl = results["real_levy_area"]
            K = sl.shape[1]
            levy_w = np.zeros(K)
            for k in range(K):
                sv = sl[:, k]
                rv = rl[:, k]
                sv = sv[np.isfinite(sv)]
                rv = rv[np.isfinite(rv)]
                if len(sv) > 0 and len(rv) > 0:
                    levy_w[k] = wasserstein_distance(sv, rv)
            summary["levy_area_wasserstein"] = float(np.nanmean(levy_w))

        # Общая метрика
        metric_values = [
            v
            for k, v in summary.items()
            if k != "overall_multivariate_distance" and np.isfinite(v)
        ]
        summary["overall_multivariate_distance"] = (
            float(np.mean(metric_values)) if metric_values else float("nan")
        )

        return summary


# =============================================================================
# Визуализация многомерных метрик
# =============================================================================


def plot_multivariate_comparison(
    synthetic_metrics: dict,
    real_metrics: dict,
    save_path: str | Path,
    n_bins: int = 30,
    figsize_per_cell: tuple[float, float] = (4.0, 3.0),
) -> plt.Figure:
    """
    Построение гистограмм многомерных метрик для синтетических и реальных данных.
    
    Parameters
    ----------
    synthetic_metrics : dict
        Результаты analyze_batch для синтетических данных.
    real_metrics : dict
        Результаты analyze_batch для реальных данных.
    save_path : str | Path
        Путь для сохранения графика.
    n_bins : int
        Число бинов для гистограмм.
    figsize_per_cell : tuple[float, float]
        Размер одной ячейки в дюймах.
    
    Returns
    -------
    plt.Figure
    """
    # Метрики для визуализации (массивы значений)
    metrics_to_plot = {
        "signature_components": ("synthetic_signature", "real_signature", "Signature components"),
        "cross_corr": ("synthetic_cross_corr", "real_cross_corr", "Cross-correlation"),
        "rank_ratio": ("synthetic_rank_ratio", "real_rank_ratio", "Correlation rank ratio"),
        "levy_area": ("synthetic_levy_area", "real_levy_area", "Levy area components"),
    }
    
    # Подсчёт числа графиков
    n_plots = 0
    for key, (synth_key, real_key, title) in metrics_to_plot.items():
        synth_data = synthetic_metrics.get(synth_key)
        real_data = real_metrics.get(real_key)
        if synth_data is not None or real_data is not None:
            n_plots += 1
    
    if n_plots == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    n_cols = min(n_plots, 3)
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False
    )
    axes_flat = axes.flatten()
    
    plot_idx = 0
    for key, (synth_key, real_key, title) in metrics_to_plot.items():
        synth_data = synthetic_metrics.get(synth_key)
        real_data = real_metrics.get(real_key)
        
        if synth_data is None and real_data is None:
            continue
        
        ax = axes_flat[plot_idx]
        plot_idx += 1
        
        # Собираем все данные для определения диапазона
        all_vals = []
        if synth_data is not None:
            synth_flat = synth_data[~np.isnan(synth_data)].ravel()
            all_vals.extend(synth_flat)
        if real_data is not None:
            real_flat = real_data[~np.isnan(real_data)].ravel()
            all_vals.extend(real_flat)
        
        if len(all_vals) == 0:
            ax.text(0.5, 0.5, "No finite values", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
            continue
        
        lo, hi = np.percentile(all_vals, [2, 98])
        pad = (hi - lo) * 0.05 if hi > lo else abs(hi) * 0.1
        lo, hi = lo - pad, hi + pad
        
        # Гистограммы
        if synth_data is not None:
            synth_flat = synth_data[~np.isnan(synth_data)].ravel()
            ax.hist(synth_flat, bins=n_bins, range=(lo, hi), 
                   alpha=0.5, density=True, color='steelblue', 
                   label='synthetic', edgecolor='none')
        
        if real_data is not None:
            real_flat = real_data[~np.isnan(real_data)].ravel()
            ax.hist(real_flat, bins=n_bins, range=(lo, hi),
                   alpha=0.5, density=True, color='coral',
                   label='real', edgecolor='none')
        
        # KDE (если есть данные)
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(lo, hi, 200)
            
            if synth_data is not None and len(synth_flat) > 5:
                kde = gaussian_kde(synth_flat)
                ax.plot(xs, kde(xs), color='steelblue', linewidth=2, linestyle='-')
            
            if real_data is not None and len(real_flat) > 5:
                kde = gaussian_kde(real_flat)
                ax.plot(xs, kde(xs), color='coral', linewidth=2, linestyle='-')
        except Exception:
            pass
        
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('value')
        ax.set_ylabel('density')
        ax.legend(fontsize=8, loc='upper right')
        
        # Статистики
        stats_text = []
        if synth_data is not None:
            stats_text.append(f"synth: μ={np.nanmean(synth_data):.3g}, σ={np.nanstd(synth_data):.3g}")
        if real_data is not None:
            stats_text.append(f"real: μ={np.nanmean(real_data):.3g}, σ={np.nanstd(real_data):.3g}")
        
        ax.text(0.98, 0.97, '\n'.join(stats_text),
               transform=ax.transAxes, ha='right', va='top',
               fontsize=7, color='gray',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Скрыть пустые ячейки
    for ax in axes_flat[plot_idx:]:
        ax.set_visible(False)
    
    fig.suptitle("Multivariate Metrics Comparison", fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# Тестирование
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ МНОГОМЕРНОГО АНАЛИЗА")
    print("=" * 70)

    backend = _detect_backend()
    print(f"\nОбнаруженный бэкенд сигнатур: {backend}")

    # --- Тест формул длин сигнатур ---
    print("\n--- Длины сигнатур ---")
    for D in (2, 3, 5):
        for depth in (2, 3):
            sl = _sig_length(D, depth)
            lsl = _logsig_length(D, depth)
            print(f"  D={D}, depth={depth}: sig_length={sl}, logsig_length={lsl}")

    # --- Тест ручной сигнатуры ---
    print("\n--- Тест ручной сигнатуры (прямая линия в R^2) ---")
    # Для прямой X(t) = (t, 2t), сигнатура уровня 1: (1, 2)
    # Уровень 2: S^{11}=0.5, S^{12}=1.0, S^{21}=1.0, S^{22}=2.0
    t = np.linspace(0, 1, 100)
    straight_path = np.stack([t, 2 * t], axis=-1)  # (100, 2)
    straight_path = straight_path[np.newaxis]  # (1, 100, 2)
    sig_manual = compute_signature(
        straight_path, depth=2, normalize=False,
        use_log_signature=False, backend="manual"
    )
    print(f"  Путь: (t, 2t), depth=2")
    print(f"  Сигнатура: {sig_manual[0]}")
    print(f"  Ожидается: [1.0, 2.0, 0.5, 1.0, 1.0, 2.0]")

    # --- Основной тест ---
    print("\n--- Основной тест: сравнение распределений ---")
    B, T, D = 50, 100, 3
    X_real = np.cumsum(np.random.randn(B, T, D) * 0.1, axis=1)
    X_similar = np.cumsum(np.random.randn(B, T, D) * 0.1, axis=1)
    X_different = np.cumsum(np.random.randn(B, T, D) * 2.0 + 5.0, axis=1)

    analyzer = MultivariateAnalyzer(seed=42)

    print(f"\n  Бэкенд: {analyzer.backend}")
    print(f"  Данные: B={B}, T={T}, D={D}")

    # Базовый анализ
    results_similar = analyzer.analyze_batch(X_similar, X_real)
    results_different = analyzer.analyze_batch(X_different, X_real)

    summary_similar = analyzer.compute_summary(results_similar)
    summary_different = analyzer.compute_summary(results_different)

    print(f"\n  Сводка (similar vs real):")
    for k, v in summary_similar.items():
        print(f"    {k}: {v:.6f}")

    print(f"\n  Сводка (different vs real):")
    for k, v in summary_different.items():
        print(f"    {k}: {v:.6f}")

    d_sim = summary_similar.get("overall_multivariate_distance", float("nan"))
    d_dif = summary_different.get("overall_multivariate_distance", float("nan"))
    print(f"\n  different >> similar? {'ДА ✓' if d_dif > d_sim else 'НЕТ ✗'}")

    # --- Тест сигнатурного лосса ---
    print("\n--- Сигнатурный MMD-лосс ---")
    mmd_self = signature_kernel_mmd_loss(X_real, X_real, depth=3, backend=backend)
    mmd_sim = signature_kernel_mmd_loss(X_real, X_similar, depth=3, backend=backend)
    mmd_dif = signature_kernel_mmd_loss(X_real, X_different, depth=3, backend=backend)

    print(f"  real vs real:      {mmd_self:.6f}  (≈ 0)")
    print(f"  real vs similar:   {mmd_sim:.6f}")
    print(f"  real vs different: {mmd_dif:.6f}")
    print(f"  different >> similar? {'ДА ✓' if mmd_dif > mmd_sim else 'НЕТ ✗'}")

    # --- Тест проекций ---
    print("\n--- Тест проекций (D=10) ---")
    D_big = 10
    X_real_big = np.cumsum(np.random.randn(B, T, D_big) * 0.1, axis=1)
    X_fake_big = np.cumsum(np.random.randn(B, T, D_big) * 0.5, axis=1)

    proj_results = analyzer.analyze_with_projections(
        X_fake_big, X_real_big, proj_dim=4, n_projections=5, depth=2
    )
    print(f"  mean_wasserstein: {proj_results['mean_wasserstein']:.6f}")

    print("\n" + "=" * 70)
    print("ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 70)
