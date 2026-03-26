"""
multivariate_statistics.py — статистики для многомерных временных рядов.

Интерфейс
---------
Все функции принимают массив формы (B, T, D) и возвращают (B, K),
где K — размерность выходного вектора статистики.

B — количество многомерных рядов (батч).
T — длина временного ряда.
D — число каналов (размерность пространства пути).

Это позволяет переиспользовать функции для:
  - выхода TSGenerator           : tensor (B, T, D+1)[..., 1:] → (B, T, D)
  - реальных многомерных данных  : np.ndarray (B, T, D)

Зависимости
-----------
  numpy      — обязательно
  iisignature — для signature / log_signature
               pip install iisignature

Размерность сигнатуры
---------------------
При D каналах и глубине m:
  sig:     siglength(D, m)    = (D^(m+1) - 1) / (D - 1) - 1
  logsig:  logsiglength(D, m)

Пример для D=3, m=3:
  sig:    3 + 9 + 27 = 39 компонент
  logsig: 3 + 3 + 8  = 14 компонент  (значительно компактнее)

Практические рекомендации по глубине
-------------------------------------
  D ≤  5, m=3  — стандартный выбор для калибровки генератора
  D ≤ 10, m=2  — если D большой
  D > 10, m=2  — и рассмотреть проекцию на случайный подпространство dim=5
"""

from __future__ import annotations

from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_input_3d(x: np.ndarray) -> np.ndarray:
    """
    Привести вход к float64 и проверить форму (B, T, D).

    Parameters
    ----------
    x : array-like, shape (B, T, D)

    Returns
    -------
    np.ndarray, shape (B, T, D), dtype float64

    Raises
    ------
    ValueError если размерность не равна 3.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(
            f"Ожидается массив формы (B, T, D), получено shape={x.shape}. "
            "Если у вас один ряд — добавьте батч-измерение: x[np.newaxis, ...]"
        )
    # Заменяем ±inf на NaN
    x = np.where(np.isinf(x), np.nan, x)
    return x


def _sig_length(D: int, depth: int) -> int:
    """Число компонент в сигнатуре глубины depth для D-мерного пути."""
    if D == 1:
        return depth
    return (D ** (depth + 1) - 1) // (D - 1) - 1


def _logsig_length(D: int, depth: int) -> int:
    """Число компонент в лог-сигнатуре глубины depth для D-мерного пути."""
    try:
        import iisignature
        return iisignature.logsiglength(D, depth)
    except ImportError:
        raise ImportError("pip install iisignature")


# ---------------------------------------------------------------------------
# Предобработка пути
# ---------------------------------------------------------------------------

def _prepare_path(xi: np.ndarray) -> np.ndarray | None:
    """
    Предобработка одного пути (T, D) перед вычислением сигнатуры.

    1. Если есть NaN — возвращает None (путь пропускается).
    2. Z-score нормализация каждого канала: делает пути сравнимыми
       независимо от абсолютного масштаба.
    3. Приведение к float64.

    Нормализация здесь важна: сигнатура чувствительна к масштабу,
    поэтому без нормализации каналы с большой амплитудой будут доминировать
    в итерированных интегралах высших порядков.
    """
    if np.any(np.isnan(xi)):
        return None
    xi = xi.astype(np.float64)
    mu  = xi.mean(axis=0, keepdims=True)
    sig = xi.std(axis=0, ddof=1, keepdims=True)
    sig = np.where(sig == 0, 1.0, sig)   # константный канал → не делим на 0
    return (xi - mu) / sig


# ---------------------------------------------------------------------------
# 1. signature — сигнатура пути
# ---------------------------------------------------------------------------

def signature(
    x: np.ndarray,
    depth: int = 3,
) -> np.ndarray:
    """
    Сигнатура многомерного временного ряда до заданной глубины.

    Что измеряет
    ------------
    Сигнатура — иерархическое описание формы пути в R^D через итерированные
    интегралы. Уровень k содержит все интегралы вида:

        S^{i_1,...,i_k} = ∫_{s_1<...<s_k} dX^{i_1}_{s_1} ⋯ dX^{i_k}_{s_k}

    Интерпретация по уровням:
      Уровень 1 (D компонент):
        S^i = X^i_T - X^i_0 — суммарное изменение канала i (тренд)

      Уровень 2 (D² компонент):
        S^{ij} = ∫∫_{s<t} dX^i_s dX^j_t — захватывает порядок событий
        и ковариацию инкрементов каналов i и j.
        S^{ij} - S^{ji} = "площадь" между путями i и j (signed area),
        мера нелинейной взаимосвязи.

      Уровень 3 (D³ компонент):
        S^{ijk} — трёхточечные корреляции, нелинейные зависимости.

    Связь с генератором
    -------------------
    TransformationModule создаёт нелинейные зависимости между каналами
    через вычислительный граф (sin, mul, exp). Сигнатура уровня 2–3
    непосредственно измеряет эти зависимости.
    S^{ij} ≠ S^{ji} означает что канал i "ведёт" за каналом j —
    это появляется при случайных сдвигах латентных компонент.

    Размерность выхода
    ------------------
    K = siglength(D, depth) = D + D² + ... + D^depth
    Например: D=3, depth=3 → K = 3 + 9 + 27 = 39

    Parameters
    ----------
    x     : np.ndarray, shape (B, T, D)
    depth : int, default 3
        Глубина усечения сигнатуры.
        Рекомендации: D≤5 → depth=3, D≤10 → depth=2, D>10 → depth=2
        с предварительной проекцией (см. projected_signature).

    Returns
    -------
    np.ndarray, shape (B, K)
        Вектор сигнатуры для каждого ряда.
        Ряды с NaN → строка из NaN.

    Raises
    ------
    ImportError : если iisignature не установлен

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((10, 100, 3))  # 10 рядов, T=100, D=3
    >>> sig = signature(x, depth=3)
    >>> sig.shape  # (10, 39)
    (10, 39)
    """
    try:
        import iisignature
    except ImportError:
        raise ImportError(
            "Для вычисления сигнатур необходим пакет iisignature:\n"
            "    pip install iisignature"
        )

    x = _validate_input_3d(x)
    B, T, D = x.shape
    K = _sig_length(D, depth)
    result = np.full((B, K), np.nan, dtype=np.float64)

    for b in range(B):
        path = _prepare_path(x[b])
        if path is None:
            continue
        try:
            result[b] = iisignature.sig(path, depth)
        except Exception:
            pass   # оставляем NaN

    return result


# ---------------------------------------------------------------------------
# 2. log_signature — логарифмическая сигнатура
# ---------------------------------------------------------------------------

def log_signature(
    x: np.ndarray,
    depth: int = 3,
) -> np.ndarray:
    """
    Логарифмическая сигнатура многомерного временного ряда.

    Что измеряет
    ------------
    Лог-сигнатура — компактное представление сигнатуры через проекцию
    на алгебру Ли. Содержит ту же информацию, но в значительно меньшем
    числе компонент за счёт отбрасывания redundant shuffle-произведений.

    Для D=3, depth=3:
      sig:    39 компонент
      logsig: 14 компонент

    Уровень 1 лог-сигнатуры совпадает с уровнем 1 сигнатуры (тренды).
    Уровень 2 содержит коммутаторы [i,j] = S^{ij} - S^{ji} —
    это "площадь Леви" (Lévy area), мера нелинейной взаимосвязи.

    Когда использовать лог-сигнатуру вместо сигнатуры
    ---------------------------------------------------
    - Для калибровки генератора: лог-сигнатура компактнее и не содержит
      избыточной информации — хорошо для гистограмм.
    - Для ML-признаков: лог-сигнатура обычно предпочтительнее.
    - Для полноты: сигнатура содержит чуть больше информации.

    Parameters
    ----------
    x     : np.ndarray, shape (B, T, D)
    depth : int, default 3

    Returns
    -------
    np.ndarray, shape (B, K_log)
        K_log = logsiglength(D, depth) < siglength(D, depth)
        Ряды с NaN → строка из NaN.

    Examples
    --------
    >>> x = np.random.default_rng(0).standard_normal((10, 100, 3))
    >>> ls = log_signature(x, depth=3)
    >>> ls.shape  # (10, 14)  — компактнее чем signature (10, 39)
    (10, 14)
    """
    try:
        import iisignature
    except ImportError:
        raise ImportError(
            "Для вычисления лог-сигнатур необходим пакет iisignature:\n"
            "    pip install iisignature"
        )

    x = _validate_input_3d(x)
    B, T, D = x.shape
    K = iisignature.logsiglength(D, depth)
    result = np.full((B, K), np.nan, dtype=np.float64)

    # prepare кешируется по (D, depth) — создаём один раз
    s = iisignature.prepare(D, depth)

    for b in range(B):
        path = _prepare_path(x[b])
        if path is None:
            continue
        try:
            result[b] = iisignature.logsig(path, s)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# 3. projected_signature — сигнатура после случайной проекции
# ---------------------------------------------------------------------------

def projected_signature(
    x: np.ndarray,
    depth: int = 3,
    proj_dim: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Сигнатура после случайной линейной проекции D → proj_dim.

    Что решает
    ----------
    При большом D (например D=21 для Weather или D=321 для Electricity)
    размерность сигнатуры D^depth экспоненциально растёт и становится
    непрактичной. Случайная проекция Johnson-Lindenstrauss снижает D
    до управляемого размера, сохраняя геометрию путей.

    Теоретическое обоснование: если D достаточно велико для JL-проекции,
    то сигнатура проекции хорошо аппроксимирует сигнатуру исходного пути
    в среднем по случайным проекциям. Это используется в работах по
    обучению на случайных признаках из сигнатур (random signature features).

    Parameters
    ----------
    x        : np.ndarray, shape (B, T, D)
    depth    : int, default 3
    proj_dim : int, default 5
        Размерность после проекции. Рекомендуется 3–8.
    seed     : int, default 42
        Сид для воспроизводимости проекционной матрицы.

    Returns
    -------
    np.ndarray, shape (B, K)
        K = siglength(proj_dim, depth)

    Examples
    --------
    >>> x = np.random.default_rng(0).standard_normal((10, 100, 21))
    >>> ps = projected_signature(x, depth=3, proj_dim=5)
    >>> ps.shape  # (10, 39)  — независимо от исходного D=21
    (10, 39)
    """
    x = _validate_input_3d(x)
    B, T, D = x.shape

    rng = np.random.default_rng(seed)
    # Матрица Гаусса, нормированная по столбцам: (D, proj_dim)
    P = rng.standard_normal((D, proj_dim)) / np.sqrt(proj_dim)

    # Проецируем весь батч: (B, T, D) @ (D, proj_dim) → (B, T, proj_dim)
    x_proj = x @ P   # broadcasting по B, T

    return signature(x_proj, depth=depth)


# ---------------------------------------------------------------------------
# 4. mean_abs_cross_correlation — средняя межканальная корреляция
# ---------------------------------------------------------------------------

def mean_abs_cross_correlation(x: np.ndarray) -> np.ndarray:
    """
    Средняя абсолютная кросс-корреляция между каналами (лаг 0).

    Что измеряет
    ------------
    Агрегированная линейная связанность каналов.
    0 — каналы полностью независимы.
    1 — каналы идеально линейно зависимы.

    Связь с генератором
    -------------------
    Определяется отношением L/D (число латентных факторов / число каналов):
    при L << D все каналы порождены малым числом факторов → высокая корреляция.
    При L ≈ D каналы почти независимы.

    Parameters
    ----------
    x : np.ndarray, shape (B, T, D)

    Returns
    -------
    np.ndarray, shape (B, 1)
        Значения в [0, 1]. Одноканальный ряд (D=1) → NaN.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x_ind = rng.standard_normal((5, 100, 4))   # независимые каналы
    >>> x_dep = np.tile(rng.standard_normal((5, 100, 1)), (1, 1, 4))  # одинаковые
    >>> mean_abs_cross_correlation(x_ind)   # ≈ мало
    >>> mean_abs_cross_correlation(x_dep)   # ≈ 1
    """
    x = _validate_input_3d(x)
    B, T, D = x.shape
    result = np.full((B, 1), np.nan, dtype=np.float64)

    if D < 2:
        return result

    for b in range(B):
        xi = x[b]  # (T, D)
        if np.any(np.isnan(xi)):
            continue
        # Корреляционная матрица (D, D)
        corr = np.corrcoef(xi.T)
        # Берём верхний треугольник без диагонали
        idx = np.triu_indices(D, k=1)
        result[b, 0] = np.abs(corr[idx]).mean()

    return result


# ---------------------------------------------------------------------------
# 5. correlation_rank_ratio — нормированный ранг матрицы корреляций
# ---------------------------------------------------------------------------

def correlation_rank_ratio(
    x: np.ndarray,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Нормированный эффективный ранг матрицы корреляций: rank(Corr) / D.

    Что измеряет
    ------------
    Долю независимых степеней свободы в многомерном ряде.
    1.0 — все каналы независимы (полный ранг).
    ~0  — почти все каналы линейно зависимы (мало общих факторов).

    Является прямым измерением сжимаемости: если rank_ratio = L/D,
    то D каналов порождены примерно L независимыми факторами.

    Parameters
    ----------
    x   : np.ndarray, shape (B, T, D)
    tol : float, default 1e-3
        Порог для определения ненулевых собственных значений.
        Собственное значение считается ненулевым если оно > tol * max(eigenvalue).

    Returns
    -------
    np.ndarray, shape (B, 1)
        Значения в (0, 1]. D=1 → NaN.
    """
    x = _validate_input_3d(x)
    B, T, D = x.shape
    result = np.full((B, 1), np.nan, dtype=np.float64)

    if D < 2:
        return result

    for b in range(B):
        xi = x[b]
        if np.any(np.isnan(xi)):
            continue
        corr = np.corrcoef(xi.T)
        eigvals = np.linalg.eigvalsh(corr)
        threshold = tol * eigvals.max()
        rank = int((eigvals > threshold).sum())
        result[b, 0] = rank / D

    return result


# ---------------------------------------------------------------------------
# 6. levy_area_matrix — матрица площадей Леви (уровень 2 лог-сигнатуры)
# ---------------------------------------------------------------------------

def levy_area_matrix(x: np.ndarray) -> np.ndarray:
    """
    Матрица площадей Леви для всех пар каналов, в виде плоского вектора.

    Что измеряет
    ------------
    Lévy area между каналами i и j:
        A^{ij} = (1/2) * (S^{ij} - S^{ji})
               = (1/2) * ∫∫_{s<t} (dX^i_s dX^j_t - dX^j_s dX^i_t)

    Геометрически — ориентированная площадь проекции пути на плоскость (i,j).
    A^{ij} ≠ 0 означает нелинейную зависимость между каналами.
    A^{ij} = -A^{ji} (антисимметрична).

    Связь с генератором
    -------------------
    Операции mul и sin в TransformationModule создают ненулевые площади Леви.
    Если TransformationModule содержит только линейные операции (add, sub),
    все площади Леви будут ≈ 0.

    Вычисляется аналитически через трапезоидный метод, без iisignature.

    Parameters
    ----------
    x : np.ndarray, shape (B, T, D)

    Returns
    -------
    np.ndarray, shape (B, D*(D-1)//2)
        Верхний треугольник матрицы площадей (i < j).
        D=1 → пустой массив shape (B, 0).
    """
    x = _validate_input_3d(x)
    B, T, D = x.shape
    K = D * (D - 1) // 2
    result = np.full((B, K), np.nan, dtype=np.float64)

    if D < 2:
        return result

    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]

    for b in range(B):
        xi = x[b]   # (T, D)
        if np.any(np.isnan(xi)):
            continue

        # Нормализуем каждый канал
        path = _prepare_path(xi)
        if path is None:
            continue

        dx = np.diff(path, axis=0)   # (T-1, D)
        # S^{ij} ≈ ∑_{t} X^i_t * dX^j_t  (трапезоидная аппроксимация)
        # S^{ij} - S^{ji} = ∑_t (X^i_t * dX^j_t - X^j_t * dX^i_t)
        X_mid = path[:-1]            # (T-1, D)

        for k, (i, j) in enumerate(pairs):
            area = np.sum(X_mid[:, i] * dx[:, j] - X_mid[:, j] * dx[:, i])
            result[b, k] = 0.5 * area

    return result


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def signature_dim(D: int, depth: int) -> int:
    """Размерность вектора сигнатуры для D каналов и глубины depth."""
    return _sig_length(D, depth)


def log_signature_dim(D: int, depth: int) -> int:
    """Размерность вектора лог-сигнатуры для D каналов и глубины depth."""
    return _logsig_length(D, depth)


def print_dimension_table(
    D_values: list[int] = (2, 3, 5, 7, 10),
    depth_values: list[int] = (2, 3, 4),
) -> None:
    """
    Выводит таблицу размерностей сигнатуры и лог-сигнатуры.

    Полезно для выбора depth перед запуском расчётов.

    Examples
    --------
    >>> print_dimension_table()
    D=2  depth=2:  sig=6     logsig=3
    D=2  depth=3:  sig=14    logsig=5
    ...
    """
    try:
        import iisignature
        has_iisig = True
    except ImportError:
        has_iisig = False

    print(f"{'D':>4}  {'depth':>5}  {'sig':>8}  {'logsig':>8}")
    print("-" * 35)
    for D in D_values:
        for m in depth_values:
            sig_k = _sig_length(D, m)
            logsig_k = iisignature.logsiglength(D, m) if has_iisig else "?"
            print(f"{D:>4}  {m:>5}  {sig_k:>8}  {str(logsig_k):>8}")