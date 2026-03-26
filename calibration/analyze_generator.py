from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Импортируем типы из ваших модулей
from sampler import TSGenerator, GeneratorConfig
from calibration.analyze_latent import StatSpec, _compute_stats_df


def _plot_generator_distributions(
    df: pd.DataFrame,
    statistics: Sequence[StatSpec],
    title: str,
    n_bins: int,
    save_path: Path,
    figsize_per_cell: tuple[float, float],
) -> plt.Figure:
    """Визуализация распределения статистик финальных рядов."""
    n_stats = len(statistics)
    n_cols = min(n_stats, 4)
    n_rows = math.ceil(n_stats / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
    )
    axes_flat = np.array(axes).flatten()

    for idx, spec in enumerate(statistics):
        ax = axes_flat[idx]
        col = df[spec.name].values
        valid = col[np.isfinite(col)]

        if valid.size == 0:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Робастное ограничение по квантилям для визуализации
        p1, p99 = np.percentile(valid, [1, 99])
        pad = (p99 - p1) * 0.05 if p99 > p1 else 0.5
        lo, hi = p1 - pad, p99 + pad

        # Гистограмма
        ax.hist(
            valid,
            bins=n_bins,
            range=(lo, hi),
            color="forestgreen",
            alpha=0.4,
            density=True,
        )

        # Медиана
        med = np.median(valid)
        ax.axvline(
            med, color="darkred", linestyle="--", alpha=0.7, label=f"med={med:.3g}"
        )

        ax.set_title(spec.name, fontweight="bold")
        ax.set_xlabel(spec.xlabel)
        ax.legend(fontsize=8)

        # Инфо-текст
        ax.text(
            0.95,
            0.95,
            f"N={len(valid)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="gray",
        )

    for ax in axes_flat[n_stats:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def analyze_generator(
    config: GeneratorConfig,
    statistics: Sequence[StatSpec],
    n_series: int = 1000,
    batch_size: int = 32,
    device: str = "cpu",
    save_path: str | Path = "generator_profile.png",
    n_bins: int = 40,
    figsize_per_cell: tuple[float, float] = (4.0, 3.5),
    show: bool = False,
) -> pd.DataFrame:
    """
    Профилирует полный пайплайн генератора (Latent -> Transform -> Noise).

    В отличие от analyze_latent:
    1. Использует публичный метод gen.generate().
    2. seq_len не передается, а берется из config.seq_len_range для каждого батча.
    3. Статистики считаются по всем измерениям (D) сгенерированного батча.
    """
    generator = TSGenerator(config, device=device)
    save_path = Path(save_path)

    all_stats_dfs = []
    total_collected = 0

    print(f"[analyze_generator] Старт профилирования. Цель: {n_series} рядов.")

    while total_collected < n_series:
        # Генерируем батч. T и D сэмплируются внутри на основе config
        # data shape: (B, T, D)
        data, plan = generator.generate(batch_size=batch_size, return_metadata=True)

        # Перекладываем в (B * D, T) для вычисления статистик по каждому ряду
        # Сначала (B, D, T), потом reshape
        B, T, D = data.shape
        flat_data = data.transpose(1, 2).reshape(-1, T).cpu().numpy()

        # Вычисляем статистики для этого батча
        # flat_data имеет форму (B*D, T), нужно разбить на списки рядов
        series_list = [flat_data[i] for i in range(flat_data.shape[0])]
        batch_df = _compute_stats_df(
            series_list, statistics, extra_cols={}
        )

        # Добавляем метаданные о батче (опционально)
        batch_df["seq_len"] = T
        batch_df["dim"] = D

        all_stats_dfs.append(batch_df)
        total_collected += flat_data.shape[0]

        print(f"  Сгенерировано: {total_collected}/{n_series} (T={T}, D={D})")

    # Объединяем результаты и обрезаем до n_series
    df = pd.concat(all_stats_dfs, ignore_index=True).iloc[:n_series]

    # Визуализация
    title = (
        f"Generator Profile: T∈{config.seq_len_range}, D∈{config.dim_range}\n"
        f"Components: {list(config.latent.type_probs.keys())}"
    )

    fig = _plot_generator_distributions(
        df=df,
        statistics=statistics,
        title=title,
        n_bins=n_bins,
        save_path=save_path,
        figsize_per_cell=figsize_per_cell,
    )

    if show:
        plt.show()
    plt.close(fig)

    print(f"[analyze_generator] Готово. Отчет сохранен в {save_path}")
    return df
