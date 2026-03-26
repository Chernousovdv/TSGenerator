"""
analyze_latent.py — профилирование TSGenerator через распределения статистик.

Три уровня анализа
------------------
1. Latent per-type  — статистики для каждого типа латентной компоненты
                      отдельно (ARIMA, KernelSynth, TSI, ETS).
                      Позволяет проверить что приор каждой компоненты
                      настроен правильно.

2. Latent mixture   — статистики по всем латентным компонентам вместе,
                      с KDE-кривыми по типам поверх общей гистограммы.
                      Показывает итоговое распределение на латентном уровне
                      и как разные типы перекрываются.

3. Observed         — статистики по итоговому выходу генератора (B, T, D+1)
                      после трансформаций и шума, по каждому каналу D.
                      Это и есть то, на чём будет обучаться модель.

Использование
-------------
    from generator import TSGenerator, GeneratorConfig
    from calibration.analyze_latent import StatSpec, profile_generator
    from calibration.statistics import (
        mean, std, acf, roughness, permutation_entropy,
        mann_kendall_z, forecastability, fft_mean,
    )

    generator = TSGenerator(config=my_config, device="cpu")

    stats = [
        StatSpec("mean",          mean),
        StatSpec("std",           std),
        StatSpec("acf_lag1",      lambda x: acf(x, lags=[1])[:, 0]),
        StatSpec("perm_entropy",  permutation_entropy),
        StatSpec("mk_z",          mann_kendall_z),
        StatSpec("roughness",     roughness),
        StatSpec("forecastability", forecastability),
        StatSpec("fft_mean",      fft_mean),
    ]

    profile_generator(
        generator  = generator,
        statistics = stats,
        n_series   = 1000,
        save_dir   = "latent_profile",
    )

Выходные файлы
--------------
    {save_dir}/1_latent_arima.png
    {save_dir}/1_latent_kernel_synth.png
    ...
    {save_dir}/2_latent_mixture.png
    {save_dir}/3_observed.png
    {save_dir}/profile_data.csv        ← все значения для дальнейшего анализа
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# StatSpec
# ---------------------------------------------------------------------------


@dataclass
class StatSpec:
    """
    Спецификация одной скалярной статистики временного ряда.

    Parameters
    ----------
    name   : str
        Имя — колонка в DataFrame и заголовок на графике.
    fn     : Callable[[np.ndarray], np.ndarray]
        Функция (B, T) → (B,). Один скаляр на ряд.
        Для функций с доп. аргументами используйте lambda:

            StatSpec("acf_lag1",  lambda x: acf(x, lags=[1])[:, 0])
            StatSpec("adf_ct",    lambda x: adf_statistic(x, regression="ct"))

    xlabel : str, optional
        Подпись оси X. Если пустая — используется name.
    """

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    xlabel: str = ""

    def __post_init__(self) -> None:
        if not self.xlabel:
            self.xlabel = self.name

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Вычислить статистику для батча (B, T) → (B,).
        При любом исключении возвращает массив NaN с предупреждением.
        """
        try:
            result = np.asarray(self.fn(x), dtype=np.float64)
            if result.ndim != 1:
                raise ValueError(
                    f"fn должна возвращать (B,), получено shape={result.shape}. "
                    "Для acf: lambda x: acf(x, lags=[k])[:, 0]"
                )
            return result
        except Exception as exc:
            warnings.warn(
                f"StatSpec '{self.name}': ошибка вычисления — {exc}. "
                "Возвращаю NaN для этого батча.",
                stacklevel=2,
            )
            return np.full(x.shape[0], np.nan)


# ---------------------------------------------------------------------------
# Сбор данных
# ---------------------------------------------------------------------------


def _collect_latent(
    generator,
    n_series: int,
    batch_size: int,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Генерирует n_series латентных рядов через generator._sample_latent_plan.

    Длина ряда T сэмплируется из generator.config.seq_len_range для каждого
    батча — так же как это делает generate(). Ряды разной длины хранятся
    как список, статистики вычисляются поэлементно.

    Возвращает
    ----------
    series : list[np.ndarray], длина N, каждый элемент shape (T_i,)
    labels : list[str], длина N — тип компоненты (spec.type)
    """
    all_series: list[np.ndarray] = []
    all_labels: list[str] = []

    while len(all_series) < n_series:
        T = generator._int_range(generator.config.seq_len_range)
        L = generator._int_range(generator.config.latent.l_range)
        plan = generator._sample_latent_plan(batch_size, L)

        with torch.no_grad():
            latent_out = generator.latent_module.execute(batch_size, T, plan)

        latent_np = (
            latent_out.cpu().numpy()
            if hasattr(latent_out, "cpu")
            else np.asarray(latent_out)
        )

        for b, specs in enumerate(plan.items):
            for l, spec in enumerate(specs):
                all_series.append(latent_np[b, :, l])
                all_labels.append(spec.type)

    return all_series[:n_series], all_labels[:n_series]


def _collect_observed(
    generator,
    n_series: int,
    batch_size: int,
) -> list[np.ndarray]:
    """
    Генерирует n_series наблюдаемых рядов через generator.generate().

    Использует полный пайплайн: латентные компоненты → трансформации → шум.
    Из выхода (B, T, D+1) вырезаем каналы [..., 1:] (без временной оси).

    Возвращает
    ----------
    list[np.ndarray], длина N, каждый элемент shape (T_i,)
    """
    all_series: list[np.ndarray] = []

    while len(all_series) < n_series:
        with torch.no_grad():
            output = generator.generate(batch_size=batch_size)

        # output: (B, T, D+1), канал 0 — время, каналы 1: — данные
        output_np = (
            output.cpu().numpy() if hasattr(output, "cpu") else np.asarray(output)
        )
        D = output_np.shape[2] - 1

        for b in range(output_np.shape[0]):
            for d in range(D):
                all_series.append(output_np[b, :, d + 1])  # (T,)

    return all_series[:n_series]


# ---------------------------------------------------------------------------
# Вычисление статистик
# ---------------------------------------------------------------------------


def _compute_stats_df(
    series: list[np.ndarray],
    statistics: Sequence[StatSpec],
    extra_cols: dict[str, list] | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    records: dict[str, list[float]] = {s.name: [] for s in statistics}

    for xi in series:
        # Normalization: Center and scale (Z-score)
        if normalize:
            std = xi.std()
            if std > 1e-9:
                xi = (xi - xi.mean()) / std
            else:
                xi = xi - xi.mean()

        # Wrap in batch (1, T)
        batch = xi[np.newaxis, :]
        for spec in statistics:
            val = spec.compute(batch)
            records[spec.name].append(float(val[0]))

    df = pd.DataFrame(records)
    if extra_cols:
        for col_name, values in extra_cols.items():
            df[col_name] = values
    return df


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

_PALETTE = plt.cm.tab10.colors  # type: ignore[attr-defined]


def _plot_grid(
    df: pd.DataFrame,
    statistics: Sequence[StatSpec],
    title: str,
    save_path: Path,
    hue_col: str | None = None,
    n_bins: int = 40,
    figsize_per_cell: tuple[float, float] = (3.8, 3.0),
    clip_pct: float = 1.0,
) -> plt.Figure:
    """
    Сетка гистограмм: одна ячейка на статистику.

    Parameters
    ----------
    hue_col      : колонка в df для раскраски KDE по группам (напр. "component_type")
    clip_pct     : процент обрезки хвостов с каждой стороны [0, 50)
    """
    n_stats = len(statistics)
    n_cols = min(n_stats, 4)
    n_rows = math.ceil(n_stats / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Группы для KDE
    if hue_col and hue_col in df.columns:
        groups = sorted(df[hue_col].dropna().unique())
        use_hue = len(groups) > 1
    else:
        groups = []
        use_hue = False

    group_color = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(groups)}

    for idx, spec in enumerate(statistics):
        ax = axes_flat[idx]
        col = df[spec.name].values.astype(float)
        finite_mask = np.isfinite(col)
        valid = col[finite_mask]
        n_dropped = col.size - valid.size

        if valid.size == 0:
            ax.text(
                0.5,
                0.5,
                "нет конечных значений",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )
            ax.set_title(spec.name, fontsize=10, fontweight="bold")
            continue

        # Робастный диапазон
        lo_raw, hi_raw = np.percentile(valid, [clip_pct, 100 - clip_pct])
        pad = (hi_raw - lo_raw) * 0.05 if hi_raw > lo_raw else abs(hi_raw) * 0.1 + 0.5
        lo, hi = lo_raw - pad, hi_raw + pad

        clipped = np.clip(valid, lo, hi)
        n_lo = int((valid < lo).sum())
        n_hi = int((valid > hi).sum())

        # Общая гистограмма
        ax.hist(
            clipped,
            bins=n_bins,
            range=(lo, hi),
            color="steelblue",
            alpha=0.28,
            density=True,
            edgecolor="none",
            label="_nolegend_",
        )

        # KDE по группам
        if use_hue:
            xs = np.linspace(lo, hi, 300)
            for grp in groups:
                mask = (df[hue_col] == grp).values & finite_mask
                vals = col[mask]
                if vals.size < 5:
                    continue
                try:
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(vals)
                    ax.plot(
                        xs, kde(xs), color=group_color[grp], linewidth=1.8, label=grp
                    )
                except Exception:
                    pass

        # Медиана
        med = np.median(valid)
        ax.axvline(
            med, color="tomato", linewidth=1.3, linestyle="--", label=f"med={med:.3g}"
        )

        ax.set_xlim(lo, hi)
        ax.set_xlabel(spec.xlabel, fontsize=9)
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(spec.name, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

        handles, leg_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, framealpha=0.6, loc="upper right")

        # Аннотация
        note = f"N={valid.size}/{col.size}"
        if n_dropped:
            note += f" ({n_dropped} non-fin)"
        ax.text(
            0.98,
            0.97,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="gray",
        )

        # Стрелки для обрезанных хвостов
        arrow_kw = dict(
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=7,
            color="dimgray",
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.0),
        )
        if n_lo > 0:
            ax.annotate(f"◀{n_lo}", xy=(0.01, 0.85), xytext=(0.09, 0.85), **arrow_kw)
        if n_hi > 0:
            ax.annotate(f"{n_hi}▶", xy=(0.99, 0.85), xytext=(0.91, 0.85), **arrow_kw)

    # Скрыть лишние ячейки
    for ax in axes_flat[n_stats:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def profile_generator(
    generator,
    statistics: Sequence[StatSpec],
    n_series: int = 1000,
    batch_size: int = 64,
    save_dir: str | Path = "latent_profile",
    n_bins: int = 40,
    clip_pct: float = 1.0,
    figsize_per_cell: tuple[float, float] = (3.8, 3.0),
    show: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Профилирует TSGenerator на трёх уровнях: latent per-type, latent mixture, observed.

    Parameters
    ----------
    generator        : TSGenerator
        Полностью инициализированный генератор. Длина последовательностей
        сэмплируется из generator.config.seq_len_range — как при обычной
        генерации.
    statistics       : sequence of StatSpec
        Список статистик. Применяются одинаково на всех трёх уровнях.
    n_series         : int, default 1000
        Целевое число рядов на каждом уровне.
        - Latent: n_series компонент (b, l).
        - Observed: n_series каналов (b, d).
    batch_size       : int, default 64
        Размер батча при генерации.
    save_dir         : str or Path, default "latent_profile"
        Директория для сохранения графиков и CSV.
        Создаётся автоматически если не существует.
    n_bins           : int, default 40
        Число бинов в гистограммах.
    clip_pct         : float, default 1.0
        Процент обрезки хвостов с каждой стороны для робастного диапазона.
    figsize_per_cell : tuple[float, float], default (3.8, 3.0)
        Размер одной ячейки сетки в дюймах.
    show             : bool, default False
        Вызывать ли plt.show() для каждого графика.

    Returns
    -------
    dict[str, pd.DataFrame] со ключами:
        "latent_per_type" — DataFrame (n_series, stats + "component_type")
        "latent_mixture"  — тот же DataFrame, используется для панели 2
        "observed"        — DataFrame (n_series, stats)

    Выходные файлы
    --------------
        {save_dir}/1_latent_{type}.png   — по одному на каждый тип компоненты
        {save_dir}/2_latent_mixture.png  — все типы вместе с KDE-раскраской
        {save_dir}/3_observed.png        — наблюдаемые каналы
        {save_dir}/profile_data.csv      — все данные для дальнейшего анализа

    Examples
    --------
    >>> generator = TSGenerator(config=my_config, device="cpu")
    >>> stats = [
    ...     StatSpec("mean",            mean),
    ...     StatSpec("std",             std),
    ...     StatSpec("acf_lag1",        lambda x: acf(x, lags=[1])[:, 0]),
    ...     StatSpec("perm_entropy",    permutation_entropy),
    ...     StatSpec("mk_z",            mann_kendall_z),
    ...     StatSpec("roughness",       roughness),
    ...     StatSpec("forecastability", forecastability),
    ...     StatSpec("fft_mean",        fft_mean),
    ... ]
    >>> dfs = profile_generator(generator, stats, n_series=1000, save_dir="profile")
    >>> print(dfs["latent_per_type"].groupby("component_type").median())
    """
    # Create base directory
    base_dir = Path(save_dir)

    # Find the next available run number
    run_number = 1
    while (base_dir / f"run_{run_number}").exists():
        run_number += 1

    # Create the numbered run directory
    save_dir = base_dir / f"run_{run_number}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[profile_generator] Saving results to: {save_dir}")

    prior = generator.config.latent
    type_names = sorted(prior.type_probs.keys())

    # ------------------------------------------------------------------
    # 1. Latent — сбор рядов
    # ------------------------------------------------------------------
    print(f"[profile_generator] Уровень 1+2: латентные ряды (n={n_series})...")
    latent_series, latent_labels = _collect_latent(generator, n_series, batch_size)
    type_counts = {t: latent_labels.count(t) for t in type_names if t in latent_labels}
    print(f"[profile_generator] Латентные типы: {type_counts}")

    # ------------------------------------------------------------------
    # 1. Latent — статистики
    # ------------------------------------------------------------------
    print("[profile_generator] Вычисление статистик (латентный уровень)...")
    df_latent = _compute_stats_df(
        latent_series,
        statistics,
        extra_cols={"component_type": latent_labels},
    )

    # ------------------------------------------------------------------
    # 1. Latent per-type — по одному графику на тип
    # ------------------------------------------------------------------
    for comp_type in type_names:
        mask = df_latent["component_type"] == comp_type
        df_type = df_latent[mask].reset_index(drop=True)

        if df_type.empty:
            print(f"[profile_generator]   {comp_type}: нет рядов, пропускаю.")
            continue

        n_type = mask.sum()
        print(f"[profile_generator]   {comp_type}: N={n_type}")

        save_path = save_dir / f"1_latent_{comp_type}.png"
        fig = _plot_grid(
            df=df_type,
            statistics=statistics,
            title=f"Latent — {comp_type}  |  N={n_type}",
            save_path=save_path,
            hue_col=None,  # один тип → раскраска не нужна
            n_bins=n_bins,
            figsize_per_cell=figsize_per_cell,
            clip_pct=clip_pct,
        )
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Latent mixture — все типы на одном графике с KDE-раскраской
    # ------------------------------------------------------------------
    print("[profile_generator] Уровень 2: смесь латентных компонент...")
    save_path = save_dir / "2_latent_mixture.png"
    fig = _plot_grid(
        df=df_latent,
        statistics=statistics,
        title=f"Latent mixture  |  N={len(df_latent)}  types={type_names}",
        save_path=save_path,
        hue_col="component_type",
        n_bins=n_bins,
        figsize_per_cell=figsize_per_cell,
        clip_pct=clip_pct,
    )
    if show:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Observed — полный пайплайн
    # ------------------------------------------------------------------
    print(f"[profile_generator] Уровень 3: наблюдаемые ряды (n={n_series})...")
    observed_series = _collect_observed(generator, n_series, batch_size)
    print("[profile_generator] Вычисление статистик (наблюдаемый уровень)...")
    df_observed = _compute_stats_df(observed_series, statistics)

    save_path = save_dir / "3_observed.png"
    fig = _plot_grid(
        df=df_observed,
        statistics=statistics,
        title=f"Observed (post-transform + noise)  |  N={len(df_observed)}",
        save_path=save_path,
        hue_col=None,
        n_bins=n_bins,
        figsize_per_cell=figsize_per_cell,
        clip_pct=clip_pct,
    )
    if show:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # CSV — все данные вместе
    # ------------------------------------------------------------------
    df_latent_out = df_latent.copy()
    df_latent_out["level"] = "latent"
    df_observed_out = df_observed.copy()
    df_observed_out["level"] = "observed"
    df_observed_out["component_type"] = "observed"

    df_all = pd.concat([df_latent_out, df_observed_out], ignore_index=True)
    csv_path = save_dir / "profile_data.csv"
    df_all.to_csv(csv_path, index=False)

    print(f"[profile_generator] Готово. Файлы сохранены в: {save_dir}/")
    _print_summary(df_latent, df_observed, statistics)

    return {
        "latent_per_type": df_latent,
        "latent_mixture": df_latent,
        "observed": df_observed,
    }


def _print_summary(
    df_latent: pd.DataFrame,
    df_observed: pd.DataFrame,
    statistics: Sequence[StatSpec],
) -> None:
    """Печатает сводную таблицу медиан по типам компонент + observed."""
    stat_cols = [s.name for s in statistics]

    rows = []
    for comp_type, grp in df_latent.groupby("component_type"):
        row = {"source": f"latent/{comp_type}"}
        for col in stat_cols:
            v = grp[col].dropna()
            row[col] = f"{v.median():.3g}" if len(v) else "—"
        rows.append(row)

    obs_row = {"source": "observed"}
    for col in stat_cols:
        v = df_observed[col].dropna()
        obs_row[col] = f"{v.median():.3g}" if len(v) else "—"
    rows.append(obs_row)

    summary = pd.DataFrame(rows).set_index("source")
    print()
    print("=== Медианы по уровням ===")
    print(summary.to_string())
    print()
