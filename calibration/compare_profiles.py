"""
compare_profiles.py — сравнение распределений статистик синтетических и реальных данных.

Визуализация: зеркальная гистограмма (back-to-back).
  - Синтетика (observed выход генератора) — бары вверх, цвет синий.
  - Реальные данные (Monash) — бары вниз, цвет оранжевый.
  - Общая ось X, единый масштаб.
  - KDE-кривая поверх каждой стороны.
  - Медианы обоих распределений — вертикальные линии.

Читать график: чем больше перекрытие KDE-кривых, тем лучше генератор
покрывает реальные данные по данной статистике.

Использование
-------------
    from calibration.compare_profiles import compare_profiles
    from calibration.analyze_latent import StatSpec, profile_generator
    from calibration.analyze_monash import profile_monash_multi

    # Считаем статистики (или загружаем из CSV)
    dfs_gen   = profile_generator(generator, stats, n_series=1000, ...)
    df_real   = profile_monash_multi(dataset_config, stats, n_series=500, ...)

    # Строим сравнение
    compare_profiles(
        df_synthetic = dfs_gen["observed"],   # наблюдаемый выход генератора
        df_real      = df_real,
        statistics   = stats,
        save_path    = "comparison.png",
    )

    # Или напрямую из CSV без повторного расчёта
    import pandas as pd
    compare_profiles(
        df_synthetic = pd.read_csv("latent_profile/profile_data.csv")
                         .query("level == 'observed'"),
        df_real      = pd.read_csv("monash_profile/monash_all.csv"),
        statistics   = stats,
        save_path    = "comparison.png",
    )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def compare_profiles(
    df_synthetic: pd.DataFrame,
    df_real: pd.DataFrame,
    statistics,                                   # Sequence[StatSpec]
    save_path: str | Path = "comparison.png",
    n_bins: int = 40,
    clip_pct: float = 1.0,
    figsize_per_cell: tuple[float, float] = (3.8, 3.5),
    color_synthetic: str = "#4C72B0",             # синий
    color_real: str = "#DD8452",                  # оранжевый
    label_synthetic: str = "synthetic (observed)",
    label_real: str = "real (Monash)",
    show: bool = False,
) -> plt.Figure:
    """
    Зеркальная гистограмма: синтетика вверх, реальные данные вниз.

    Parameters
    ----------
    df_synthetic     : pd.DataFrame
        DataFrame со статистиками синтетических рядов.
        Обычно dfs["observed"] из profile_generator().
    df_real          : pd.DataFrame
        DataFrame со статистиками реальных рядов.
        Обычно результат profile_monash_multi().
    statistics       : sequence of StatSpec
        Тот же список StatSpec что использовался при расчёте.
        Используется для получения имён колонок и подписей осей.
    save_path        : str or Path, default "comparison.png"
    n_bins           : int, default 40
    clip_pct         : float, default 1.0
        Процент обрезки хвостов с каждой стороны (робастный диапазон).
    figsize_per_cell : tuple[float, float], default (3.8, 3.5)
    color_synthetic  : str  — цвет синтетики
    color_real       : str  — цвет реальных данных
    label_synthetic  : str  — метка в легенде
    label_real       : str  — метка в легенде
    show             : bool, default False

    Returns
    -------
    plt.Figure
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    stat_names  = [s.name for s in statistics]
    stat_xlabel = {s.name: s.xlabel for s in statistics}

    n_stats = len(stat_names)
    n_cols  = min(n_stats, 4)
    n_rows  = math.ceil(n_stats / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, name in enumerate(stat_names):
        ax = axes_flat[idx]

        syn  = _get_finite(df_synthetic, name)
        real = _get_finite(df_real, name)

        if syn.size == 0 and real.size == 0:
            ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(name, fontsize=10, fontweight="bold")
            continue

        # Общий робастный диапазон по объединённым данным
        combined = np.concatenate([syn, real]) if (syn.size and real.size) \
                   else (syn if syn.size else real)
        lo, hi = _robust_range(combined, clip_pct)

        bins = np.linspace(lo, hi, n_bins + 1)
        bin_w = bins[1] - bins[0]

        # Гистограммы
        if syn.size:
            counts_s, _ = np.histogram(np.clip(syn, lo, hi), bins=bins)
            density_s   = counts_s / (syn.size * bin_w)
            ax.bar(bins[:-1], density_s, width=bin_w,
                   color=color_synthetic, alpha=0.55,
                   align="edge", label=label_synthetic)

        if real.size:
            counts_r, _ = np.histogram(np.clip(real, lo, hi), bins=bins)
            density_r   = counts_r / (real.size * bin_w)
            ax.bar(bins[:-1], -density_r, width=bin_w,
                   color=color_real, alpha=0.55,
                   align="edge", label=label_real)

        # KDE поверх
        xs = np.linspace(lo, hi, 300)
        _add_kde(ax,  syn,  xs, color_synthetic, positive=True)
        _add_kde(ax, real,  xs, color_real,      positive=False)

        # Медианы
        if syn.size:
            med_s = np.median(syn)
            ax.axvline(med_s, color=color_synthetic, linewidth=1.4,
                       linestyle="--", alpha=0.9)
        if real.size:
            med_r = np.median(real)
            ax.axvline(med_r, color=color_real, linewidth=1.4,
                       linestyle="--", alpha=0.9)

        # Нулевая линия — граница между верхом и низом
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

        # Подписи и форматирование
        ax.set_xlim(lo, hi)
        ax.set_xlabel(stat_xlabel.get(name, name), fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

        # Ось Y: показываем абсолютные значения (плотность)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{abs(v):.2g}")
        )

        # Аннотация N
        note_parts = []
        if syn.size:  note_parts.append(f"syn N={syn.size}")
        if real.size: note_parts.append(f"real N={real.size}")
        ax.text(0.98, 0.98, "  ".join(note_parts),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="gray")

        # Подписи сторон
        y_lim = ax.get_ylim()
        if syn.size and y_lim[1] > 0:
            ax.text(0.01, 0.97, "↑ synthetic", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7,
                    color=color_synthetic, alpha=0.8)
        if real.size and y_lim[0] < 0:
            ax.text(0.01, 0.03, "↓ real", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=7,
                    color=color_real, alpha=0.8)

    # Убрать лишние оси
    for ax in axes_flat[n_stats:]:
        ax.set_visible(False)

    # Общая легенда
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_synthetic, alpha=0.6),
        plt.Rectangle((0, 0), 1, 1, color=color_real,      alpha=0.6),
    ]
    fig.legend(handles, [label_synthetic, label_real],
               loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.8)

    fig.suptitle("Synthetic vs Real — statistics comparison",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    print(f"[compare_profiles] Сохранено: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_finite(df: pd.DataFrame, col: str) -> np.ndarray:
    """Извлечь конечные значения колонки или вернуть пустой массив."""
    if col not in df.columns:
        return np.array([], dtype=np.float64)
    vals = df[col].values.astype(float)
    return vals[np.isfinite(vals)]


def _robust_range(vals: np.ndarray, clip_pct: float) -> tuple[float, float]:
    """[p%, 100-p%] с небольшим отступом."""
    lo_raw, hi_raw = np.percentile(vals, [clip_pct, 100 - clip_pct])
    spread = hi_raw - lo_raw
    pad = spread * 0.05 if spread > 1e-10 else max(abs(hi_raw) * 0.1, 0.5)
    return lo_raw - pad, hi_raw + pad


def _add_kde(
    ax: plt.Axes,
    vals: np.ndarray,
    xs: np.ndarray,
    color: str,
    positive: bool,
) -> None:
    """Добавить KDE-кривую; positive=True → вверх, False → вниз."""
    if vals.size < 5:
        return
    try:
        from scipy.stats import gaussian_kde
        kde_y = gaussian_kde(vals)(xs)
        if not positive:
            kde_y = -kde_y
        ax.plot(xs, kde_y, color=color, linewidth=1.6, alpha=0.9)
    except Exception:
        pass