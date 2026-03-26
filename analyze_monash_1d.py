"""
analyze_monash.py — загрузка датасетов Monash и расчёт статистик.

Что делает
----------
1. Загружает один или несколько датасетов Monash через HuggingFace datasets.
2. Извлекает временные ряды как list[np.ndarray].
3. Вычисляет тот же набор StatSpec что и для генератора.
4. Строит гистограммы и сохраняет CSV.

Это позволяет напрямую сравнить профиль генератора с реальными данными
на одних и тех же статистиках.

Датасеты Monash на HuggingFace
-------------------------------
Все датасеты доступны через:
    datasets.load_dataset("monash_tsf", "<name>")

Популярные имена:
    "m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly",
    "m4_daily",  "m4_hourly",
    "m3_yearly", "m3_quarterly", "m3_monthly", "m3_other",
    "tourism_yearly", "tourism_quarterly", "tourism_monthly",
    "nn5_daily", "nn5_weekly",
    "weather",
    "electricity_hourly", "electricity_weekly",
    "traffic_hourly", "traffic_weekly",
    "hospital", "covid_deaths", "fred_md",
    "car_parts", "pedestrian_counts",
    "australian_electricity_demand",

Быстрый старт
-------------
    from calibration.analyze_monash import load_monash, profile_monash
    from calibration.analyze_latent import StatSpec
    from calibration.statistics import (
        mean, std, acf, adf_statistic, permutation_entropy,
        mann_kendall_z, roughness, forecastability, fft_mean,
        seasonality_strength, trend_strength,
    )

    long_stats = [
        StatSpec("mean",         mean),
        StatSpec("std",          std),
        StatSpec("acf_lag1",     lambda x: acf(x, lags=[1])[:, 0]),
        StatSpec("acf_lag2",     lambda x: acf(x, lags=[2])[:, 0]),
        StatSpec("acf_lag7",     lambda x: acf(x, lags=[7])[:, 0]),
        StatSpec("adf_ct",       lambda x: adf_statistic(x, regression="ct")),
        StatSpec("adf_c",        lambda x: adf_statistic(x, regression="c")),
        StatSpec("perm_ent_m3",  lambda x: permutation_entropy(x, m=3)),
        StatSpec("mk_z",         mann_kendall_z),
        StatSpec("trend_str",    trend_strength),
        StatSpec("seasonality",  seasonality_strength),
        StatSpec("roughness",    roughness),
        StatSpec("forecastability", forecastability),
        StatSpec("fft_mean",     fft_mean),
    ]

    # Один датасет
    df = profile_monash(
        dataset_name = "m4_monthly",
        statistics   = long_stats,
        n_series     = 1000,
        save_dir     = "monash_profile/m4_monthly",
    )

    # Несколько датасетов за один вызов
    dfs = profile_monash_multi(
        dataset_names = ["m4_monthly", "weather", "traffic_hourly"],
        statistics    = long_stats,
        n_series      = 500,
        save_dir      = "monash_profile",
    )
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Импортируем StatSpec и _plot_grid из analyze_latent — не дублируем
from calibration.analyze_latent import StatSpec, _plot_grid


# ---------------------------------------------------------------------------
# Загрузка датасета
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Маппинг: имя датасета → имя zip/tsf файла на HuggingFace
# (имя файла = то что лежит в data/)
# ---------------------------------------------------------------------------

MONASH_FILES = {
    # Строго по именам zip-файлов из репозитория (без .zip)
    "australian_electricity_demand": "australian_electricity_demand_dataset",
    "bitcoin": "bitcoin_dataset_with_missing_values",
    "car_parts": "car_parts_dataset_with_missing_values",
    "cif_2016": "cif_2016_dataset",
    "covid_deaths": "covid_deaths_dataset",
    "fred_md": "fred_md_dataset",
    "hospital": "hospital_dataset",
    "kaggle_web_traffic": "kaggle_web_traffic_dataset_with_missing_values",
    "kaggle_web_traffic_weekly": "kaggle_web_traffic_weekly_dataset",
    "kdd_cup_2018": "kdd_cup_2018_dataset_with_missing_values",
    "london_smart_meters": "london_smart_meters_dataset_with_missing_values",
    "nn5_daily": "nn5_daily_dataset_with_missing_values",
    "nn5_weekly": "nn5_weekly_dataset",
    "oikolab_weather": "oikolab_weather_dataset",
    "pedestrian_counts": "pedestrian_counts_dataset",
    "rideshare": "rideshare_dataset_with_missing_values",
    "saugeenday": "saugeenday_dataset",
    "solar_10_minutes": "solar_10_minutes_dataset",
    "solar_4_seconds": "solar_4_seconds_dataset",
    "solar_weekly": "solar_weekly_dataset",
    "sunspot": "sunspot_dataset_with_missing_values",
    "temperature_rain": "temperature_rain_dataset_with_missing_values",
    "tourism_monthly": "tourism_monthly_dataset",
    "tourism_quarterly": "tourism_quarterly_dataset",
    "tourism_yearly": "tourism_yearly_dataset",
    "traffic_hourly": "traffic_hourly_dataset",
    "traffic_weekly": "traffic_weekly_dataset",
    "us_births": "us_births_dataset",
    "vehicle_trips": "vehicle_trips_dataset_with_missing_values",
    "weather": "weather_dataset",
    "wind_4_seconds": "wind_4_seconds_dataset",
    "wind_farms": "wind_farms_minutely_dataset_with_missing_values",
}

HF_BASE = (
    "https://huggingface.co/datasets/Monash-University/monash_tsf/resolve/main/data"
)


def _parse_tsf(text: str) -> list[np.ndarray]:
    """
    Парсит текстовый формат .tsf (Monash Time Series Forecasting).

    Формат:
        Заголовок с метаданными (@attribute, @frequency и т.д.)
        @data
        series_name:start_timestamp:val1,val2,val3,...
        ...

    Возвращает список одномерных массивов — значения каждого ряда.
    """
    lines = text.splitlines()

    # Ищем начало данных
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_start = i + 1
            break

    series_list = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue

        # Формат: [name:]?[timestamp:]?values
        # Разделитель — двоеточие, значения через запятую
        parts = line.split(":")
        # Последняя часть всегда — значения через запятую
        values_str = parts[-1]

        try:
            values = np.array(
                [
                    float(v) if v.strip() != "?" else np.nan
                    for v in values_str.split(",")
                ],
                dtype=np.float64,
            )
            values = np.where(np.isinf(values), np.nan, values)
            series_list.append(values)
        except ValueError:
            continue  # пропускаем битые строки

    return series_list


def load_monash(
    dataset_name: str,
    n_series: int | None = None,
    min_length: int = 10,
    max_length: int | None = None,
    normalize: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> list[np.ndarray]:
    """
    Загружает датасет Monash и возвращает список рядов.

    Данные скачиваются с HuggingFace как zip-архив, распаковываются
    в памяти, парсится внутренний .tsf файл.

    Parameters
    ----------
    dataset_name : str
        Имя датасета. Допустимые значения: ключи словаря MONASH_FILES.
        Примеры: "m4_monthly", "weather", "traffic_hourly".
    n_series     : int or None, default None
        Максимальное число рядов. None — использовать все.
    min_length   : int, default 10
        Минимальная длина ряда в точках.
    max_length   : int or None, default None
        Если задан — ряды длиннее max_length нарезаются на непересекающиеся
        окна длины max_length. Последний остаток отбрасывается если он
        короче min_length. Полезно для датасетов с малым числом очень
        длинных рядов (electricity, traffic и т.п.).
    normalize    : bool, default True
        Применять z-score нормализацию к каждому ряду: x = (x - mean) / std.
        Ряды с нулевым std (константные) после нормализации заменяются NaN
        и отбрасываются. Рекомендуется True при сравнении с генератором —
        TransformModule применяет z-score нормализацию к наблюдаемым рядам.
        Статистики mean и std становятся тривиальными после нормализации
        (≈0 и ≈1 соответственно), остальные остаются информативными.
    shuffle      : bool, default True
    seed         : int, default 42
    cache_dir    : str or Path or None, default None
        Директория для кеширования zip-файла. Если None — кешируется
        в ~/.cache/monash_tsf/. Повторные вызовы читают из кеша.

    Returns
    -------
    list[np.ndarray]
        Каждый элемент — одномерный ряд shape (T,), dtype float64.

    Raises
    ------
    KeyError  : неизвестное имя датасета
    RuntimeError : ошибка скачивания
    """
    import io
    import zipfile
    import urllib.request
    from pathlib import Path as _Path

    if dataset_name not in MONASH_FILES:
        raise KeyError(
            f"Неизвестный датасет '{dataset_name}'. "
            f"Доступные: {sorted(MONASH_FILES.keys())}"
        )

    file_stem = MONASH_FILES[dataset_name]
    zip_name = f"{file_stem}.zip"
    url = f"{HF_BASE}/{zip_name}"

    # Кеш
    if cache_dir is None:
        cache_dir = _Path.home() / ".cache" / "monash_tsf"
    cache_dir = _Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / zip_name

    if cache_path.exists():
        print(f"[load_monash] Читаю из кеша: {cache_path}")
        zip_bytes = cache_path.read_bytes()
    else:
        print(f"[load_monash] Скачиваю '{dataset_name}' с HuggingFace...")
        try:
            with urllib.request.urlopen(url) as resp:
                zip_bytes = resp.read()
            cache_path.write_bytes(zip_bytes)
            print(f"[load_monash] Сохранено в кеш: {cache_path}")
        except Exception as e:
            raise RuntimeError(
                f"Не удалось скачать '{dataset_name}' по URL: {url}\n" f"Ошибка: {e}"
            )

    # Распаковка и парсинг .tsf
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tsf_files = [n for n in zf.namelist() if n.endswith(".tsf")]
        if not tsf_files:
            raise RuntimeError(f"В архиве {zip_name} не найден .tsf файл.")
        tsf_name = tsf_files[0]
        with zf.open(tsf_name) as f:
            text = f.read().decode("utf-8", errors="replace")

    print(f"[load_monash] Парсинг {tsf_name}...")
    all_series = _parse_tsf(text)
    print(f"[load_monash] Всего рядов: {len(all_series)}")

    # Нарезка длинных рядов на окна
    if max_length is not None:
        split_series = []
        for s in all_series:
            if len(s) <= max_length:
                split_series.append(s)
            else:
                # Нарезаем на непересекающиеся окна [0:W], [W:2W], ...
                n_windows = len(s) // max_length
                for i in range(n_windows):
                    split_series.append(s[i * max_length : (i + 1) * max_length])
                # Остаток отбрасываем — он будет отфильтрован ниже если < min_length
        n_before = len(all_series)
        all_series = split_series
        print(
            f"[load_monash] Нарезка (max_length={max_length}): "
            f"{n_before} → {len(all_series)} рядов"
        )

    # Фильтрация коротких рядов
    all_series = [s for s in all_series if len(s) >= min_length]

    # Z-score нормализация
    if normalize:
        normalized = []
        n_const = 0
        for s in all_series:
            mu = np.nanmean(s)
            sig = np.nanstd(s, ddof=1)
            if sig == 0 or np.isnan(sig):
                n_const += 1
                continue  # константный или вырожденный ряд — отбрасываем
            normalized.append((s - mu) / sig)
        if n_const:
            print(f"[load_monash] Нормализация: отброшено {n_const} константных рядов.")
        all_series = normalized

    # Перемешивание
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_series))
        all_series = [all_series[i] for i in idx]

    # Выборка
    if n_series is not None:
        all_series = all_series[:n_series]

    if not all_series:
        raise RuntimeError(
            f"После фильтрации (min_length={min_length}) не осталось рядов. "
            "Попробуйте уменьшить min_length."
        )
    print(
        f"[load_monash] Итого: {len(all_series)} рядов "
        f"(min_T={min(len(s) for s in all_series)}, "
        f"max_T={max(len(s) for s in all_series)}, "
        f"median_T={int(np.median([len(s) for s in all_series]))})"
    )

    return all_series


def _compute_stats_df(
    series: list[np.ndarray],
    statistics: Sequence[StatSpec],
    extra_cols: dict[str, list] | None = None,
) -> pd.DataFrame:
    """
    Вычисляет все StatSpec поэлементно для рядов разной длины.
    Реплика из analyze_latent, работает с list[np.ndarray].
    """
    records: dict[str, list[float]] = {s.name: [] for s in statistics}

    for xi in series:
        batch = xi[np.newaxis, :]  # (1, T)
        for spec in statistics:
            val = spec.compute(batch)
            records[spec.name].append(float(val[0]))

    df = pd.DataFrame(records)
    if extra_cols:
        for col_name, values in extra_cols.items():
            df[col_name] = values
    return df


# ---------------------------------------------------------------------------
# Публичный API: один датасет
# ---------------------------------------------------------------------------


def profile_monash(
    dataset_name: str,
    statistics: Sequence[StatSpec],
    n_series: int = 1000,
    min_length: int = 10,
    max_length: int | None = None,
    normalize: bool = True,
    save_dir: str | Path = "monash_profile",
    n_bins: int = 40,
    clip_pct: float = 1.0,
    figsize_per_cell: tuple[float, float] = (3.8, 3.0),
    show: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Загружает датасет Monash и строит профиль статистик.

    Parameters
    ----------
    dataset_name     : str
        Имя датасета, например "m4_monthly", "weather", "traffic_hourly".
    statistics       : sequence of StatSpec
        Тот же список что используется для profile_generator —
        результаты напрямую сравнимы.
    n_series         : int, default 1000
        Число рядов для анализа. Если датасет меньше — берётся всё.
    min_length       : int, default 10
        Ряды короче этого значения пропускаются.
    max_length       : int or None, default None
        Нарезать длинные ряды на окна этой длины (передаётся в load_monash).
    normalize        : bool, default True
        Z-score нормализация каждого ряда перед расчётом статистик.
    save_dir         : str or Path
        Директория для сохранения графика и CSV.
    n_bins           : int, default 40
    clip_pct         : float, default 1.0
        Процент обрезки хвостов для гистограмм.
    show             : bool, default False
    seed             : int, default 42

    Returns
    -------
    pd.DataFrame, shape (N, len(statistics))
        Значения статистик для каждого ряда.

    Выходные файлы
    --------------
        {save_dir}/{dataset_name}.png
        {save_dir}/{dataset_name}.csv
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка
    series = load_monash(
        dataset_name,
        n_series=n_series,
        min_length=min_length,
        max_length=max_length,
        normalize=normalize,
        seed=seed,
    )

    # 2. Статистики
    print(f"[profile_monash] Вычисление статистик для {len(series)} рядов...")
    df = _compute_stats_df(series, statistics)

    nan_counts = df.isna().sum()
    if nan_counts.any():
        print(f"[profile_monash] NaN:\n{nan_counts[nan_counts > 0]}")

    # 3. График
    lengths = [len(s) for s in series]
    title = (
        f"Monash: {dataset_name}  |  "
        f"N={len(series)},  "
        f"T∈[{min(lengths)}, {max(lengths)}],  "
        f"median_T={int(np.median(lengths))}"
    )
    save_path = save_dir / f"{dataset_name}.png"

    print(f"[profile_monash] Построение графика → {save_path}")
    fig = _plot_grid(
        df=df,
        statistics=statistics,
        title=title,
        save_path=save_path,
        hue_col=None,
        n_bins=n_bins,
        figsize_per_cell=figsize_per_cell,
        clip_pct=clip_pct,
    )
    if show:
        plt.show()
    plt.close(fig)

    # 4. CSV
    csv_path = save_dir / f"{dataset_name}.csv"
    df.to_csv(csv_path, index=False)

    print(f"[profile_monash] Готово: {save_dir}/")
    _print_summary(df, statistics, dataset_name)

    return df


# ---------------------------------------------------------------------------
# Публичный API: несколько датасетов → один общий график
# ---------------------------------------------------------------------------


def profile_monash_multi(
    dataset_names: list[str] | dict[str, int | None],
    statistics: Sequence[StatSpec],
    n_series: int = 500,
    min_length: int = 10,
    normalize: bool = True,
    save_dir: str | Path = "monash_profile",
    n_bins: int = 40,
    clip_pct: float = 1.0,
    figsize_per_cell: tuple[float, float] = (3.8, 3.0),
    show: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Загружает несколько датасетов Monash, объединяет в общую выборку
    и строит один график с KDE-кривыми по датасетам.

    Отдельные графики на датасет не строятся — только один сводный.
    Это позволяет видеть всё покрытие реальных данных на одном экране.

    Parameters
    ----------
    dataset_names : list[str] или dict[str, int | None]
        Если list  — все датасеты загружаются без нарезки (max_length=None).
        Если dict  — ключ: имя датасета, значение: max_length для нарезки.
        Пример:
            {
                "tourism_monthly":   None,   # не нарезать
                "traffic_hourly":    168,    # нарезать на окна по 168
                "solar_10_minutes":  1008,
            }
    statistics    : sequence of StatSpec
    n_series      : int, default 500
        Максимальное число рядов из одного датасета (после нарезки).
        Датасеты с меньшим числом рядов берутся целиком.
    min_length    : int, default 10
    normalize     : bool, default True
        Z-score нормализация каждого ряда.
    save_dir      : str or Path
    n_bins        : int, default 40
    clip_pct      : float, default 1.0
    show          : bool, default False
    seed          : int, default 42

    Returns
    -------
    pd.DataFrame
        Объединённый DataFrame со всеми рядами всех датасетов.
        Колонки: имена статистик + "dataset".

    Выходные файлы
    --------------
        {save_dir}/monash_all.png   — один график, KDE по датасетам
        {save_dir}/monash_all.csv   — объединённый CSV с колонкой "dataset"
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Нормализуем входной аргумент к dict[name -> max_length]
    if isinstance(dataset_names, list):
        name_maxlen: dict[str, int | None] = {n: None for n in dataset_names}
    else:
        name_maxlen = dict(dataset_names)

    all_dfs: list[pd.DataFrame] = []

    for name, max_length in name_maxlen.items():
        print(f"\n{'='*60}")
        print(
            f"[profile_monash_multi] Датасет: {name}"
            + (f"  max_length={max_length}" if max_length else "")
        )
        try:
            series = load_monash(
                name,
                n_series=n_series,
                min_length=min_length,
                max_length=max_length,
                normalize=normalize,
                seed=seed,
            )
        except Exception as e:
            print(
                f"[profile_monash_multi] ОШИБКА при загрузке '{name}': {e}. Пропускаю."
            )
            continue

        print(f"[profile_monash_multi] Вычисление статистик ({len(series)} рядов)...")
        df = _compute_stats_df(
            series, statistics, extra_cols={"dataset": [name] * len(series)}
        )
        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("Ни один датасет не загружен успешно.")

    # Объединяем все датасеты в одну выборку
    df_all = pd.concat(all_dfs, ignore_index=True)
    total = len(df_all)
    n_datasets = df_all["dataset"].nunique()
    print(f"\n[profile_monash_multi] Всего рядов: {total} из {n_datasets} датасетов")

    nan_counts = df_all[[s.name for s in statistics]].isna().sum()
    if nan_counts.any():
        print(f"[profile_monash_multi] NaN:\n{nan_counts[nan_counts > 0]}")

    # Один общий график — без KDE по датасетам
    title = (
        f"Monash  |  {n_datasets} датасетов,  N_total={total},  "
        f"n_per_dataset≤{n_series}"
    )
    save_path = save_dir / "monash_all.png"
    print(f"[profile_monash_multi] Построение графика → {save_path}")

    fig = _plot_grid(
        df=df_all,
        statistics=statistics,
        title=title,
        save_path=save_path,
        hue_col=None,
        n_bins=n_bins,
        figsize_per_cell=figsize_per_cell,
        clip_pct=clip_pct,
    )
    if show:
        plt.show()
    plt.close(fig)

    df_all.to_csv(save_dir / "monash_all.csv", index=False)
    _print_summary_multi(df_all, statistics)
    print(f"[profile_monash_multi] Готово. Файлы в: {save_dir}/")

    return df_all


# ---------------------------------------------------------------------------
# Утилита: сводная таблица
# ---------------------------------------------------------------------------


def _print_summary(
    df: pd.DataFrame,
    statistics: Sequence[StatSpec],
    dataset_name: str,
) -> None:
    stat_cols = [s.name for s in statistics]
    summary = df[stat_cols].describe().loc[["mean", "50%", "std"]]
    summary.index = ["mean", "median", "std"]
    print(f"\n=== {dataset_name} — сводка ===")
    print(summary.round(3).to_string())
    print()


def _print_summary_multi(
    df_all: pd.DataFrame,
    statistics: Sequence[StatSpec],
) -> None:
    """Печатает медианы по каждому датасету в виде сводной таблицы."""
    stat_cols = [s.name for s in statistics]
    rows = []
    for name, grp in df_all.groupby("dataset"):
        row = {"dataset": name, "N": len(grp)}
        for col in stat_cols:
            v = grp[col].dropna()
            row[col] = round(float(v.median()), 3) if len(v) else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("dataset")
    print("\n=== Медианы по датасетам ===")
    print(summary.to_string())
    print()


# ---------------------------------------------------------------------------
# Утилита: сводная таблица
# ---------------------------------------------------------------------------


def _print_summary(
    df: pd.DataFrame,
    statistics: Sequence[StatSpec],
    dataset_name: str,
) -> None:
    stat_cols = [s.name for s in statistics]
    summary = df[stat_cols].describe().loc[["mean", "50%", "std"]]
    summary.index = ["mean", "median", "std"]
    print(f"\n=== {dataset_name} — сводка ===")
    print(summary.round(3).to_string())
    print()


def _print_summary_multi(
    df_all: pd.DataFrame,
    statistics: Sequence[StatSpec],
) -> None:
    """Печатает медианы по каждому датасету в виде сводной таблицы."""
    stat_cols = [s.name for s in statistics]
    rows = []
    for name, grp in df_all.groupby("dataset"):
        row = {"dataset": name, "N": len(grp)}
        for col in stat_cols:
            v = grp[col].dropna()
            row[col] = round(float(v.median()), 3) if len(v) else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("dataset")
    print("\n=== Медианы по датасетам ===")
    print(summary.to_string())
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Быстрый запуск из командной строки:
        python analyze_monash.py m4_monthly weather traffic_hourly
    """
    import sys

    dataset_config = {
        # стандартные — без нарезки
        "tourism_yearly": 200,
        "tourism_quarterly": 200,
        "tourism_monthly": 200,
        "hospital": 200,
        "fred_md": 200,
        "car_parts": 200,
        "nn5_weekly": 200,
        "traffic_weekly": 200,
        "solar_weekly": 200,
        "nn5_daily": 200,
        "weather": 168,
        "covid_deaths": 200,
        "saugeenday": 200,
        "sunspot": 200,
        "us_births": 200,
        "bitcoin": 200,
        "pedestrian_counts": 200,
        # высокочастотные — с нарезкой
        "traffic_hourly": 200,
        "kdd_cup_2018": 200,
        "oikolab_weather": 200,
        "australian_electricity_demand": 200,
        "london_smart_meters": 200,
        "solar_10_minutes": 200,
        "wind_farms": 200,
    }

    from configs.stat_lists import long_stats

    df = profile_monash_multi(
        dataset_names=dataset_config,
        statistics=long_stats,
        n_series=50,
        normalize=True,
        save_dir="results/monash_profile",
    )
