"""
config_calibration.py — Профилирование генератора синтетических данных.

Вычисляет:
1. Одномерные статистики (mean, std, acf, adf, и др.)
2. Многомерные статистики (сигнатуры, кросс-корреляции, площади Леви, ранг)

Использование:
    cd ts_generator
    python config_calibration.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from calibration.analyze_latent import profile_generator, StatSpec
from calibration.stats import (
    mean, std, acf, adf_statistic, permutation_entropy,
    mann_kendall_z, roughness, forecastability, fft_mean,
    seasonality_strength, trend_strength,
)
from calibration.multivariate_analysis import (
    MultivariateAnalyzer,
    SignatureConfig,
    CrossCorrelationConfig,
)
from configs.config_example import config_example
from configs.stat_lists import short_stats
from sampler import TSGenerator, GeneratorConfig


# ---------------------------------------------------------------------------
# Конфигурация одномерных статистик
# ---------------------------------------------------------------------------

long_stats = [
    StatSpec("mean", mean),
    StatSpec("std", std),
    StatSpec("acf_lag1", lambda x: acf(x, lags=[1])[:, 0]),
    StatSpec("acf_lag2", lambda x: acf(x, lags=[2])[:, 0]),
    StatSpec("adf_ct", lambda x: adf_statistic(x, regression="ct")),
    StatSpec("perm_ent_m3", lambda x: permutation_entropy(x, m=3)),
    StatSpec("mk_z", mann_kendall_z),
    StatSpec("seasonality", seasonality_strength),
    StatSpec("forecastability", forecastability),
    StatSpec("fft_mean", fft_mean),
    StatSpec("roughness", roughness),
    StatSpec("trend_str", trend_strength),
]


# ---------------------------------------------------------------------------
# Генерация батча для многомерного анализа
# ---------------------------------------------------------------------------

def generate_multivariate_batch(
    config: GeneratorConfig,
    n_series: int = 100,
    batch_size: int = 4,
    device: str = "cpu",
    seed: int = 42,
    target_dim: int = 4,
) -> np.ndarray:
    """
    Генерирует батч многомерных рядов для анализа.
    
    Returns
    -------
    np.ndarray, shape (B, T, D)
    """
    generator = TSGenerator(config=config, device=device)
    
    all_series = []
    n_generated = 0
    
    while n_generated < n_series:
        data, plan = generator.generate(batch_size=batch_size, return_metadata=True)
        # data: (B, T, D)
        for b in range(data.shape[0]):
            if n_generated >= n_series:
                break
            s = data[b].numpy()
            # Обрезаем или дополняем до target_dim
            if s.shape[1] >= target_dim:
                s = s[:, :target_dim]
            else:
                # Дополняем нулями
                pad = np.zeros((s.shape[0], target_dim - s.shape[1]))
                s = np.concatenate([s, pad], axis=1)
            all_series.append(s)
            n_generated += 1
    
    # Приводим к одинаковой длине T
    T_min = min(s.shape[0] for s in all_series)
    
    result = np.stack([s[:T_min] for s in all_series], axis=0)  # (B, T, D)
    return result


# ---------------------------------------------------------------------------
# Вычисление многомерных метрик
# ---------------------------------------------------------------------------

def compute_multivariate_metrics(
    data: np.ndarray,
    depth: int = 2,
    max_lag: int = 5,
) -> tuple[dict, dict]:
    """
    Вычисляет многомерные метрики для батча данных.
    
    Parameters
    ----------
    data : np.ndarray, shape (B, T, D)
    depth : int
        Глубина сигнатуры.
    max_lag : int
        Максимальный лаг для кросс-корреляции.
    
    Returns
    -------
    (summary, raw_results) — агрегированные метрики и сырые данные
    """
    analyzer = MultivariateAnalyzer(seed=42)
    
    # Для анализа нужен батч сравнения — используем половину данных
    B = data.shape[0]
    half = B // 2
    if half < 2:
        half = max(2, B - 1)
    
    # Сравниваем первую половину со второй
    results = analyzer.analyze_batch(
        synthetic=data[:half],
        real=data[half:half*2] if half*2 <= B else data[half:],
        signature_config=SignatureConfig(depth=depth, backend="manual"),
        cross_corr_config=CrossCorrelationConfig(max_lag=max_lag),
    )
    
    summary = analyzer.compute_summary(results)
    return summary, results


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ПРОФИЛИРОВАНИЕ ГЕНЕРАТОРА СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("=" * 70)
    
    base_dir = Path(__file__).parent
    save_dir = base_dir / "results" / "generator"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Одномерные статистики
    print("\n[1/2] Вычисление одномерных статистик...")
    generator = TSGenerator(config=config_example, device="cpu")
    
    dfs = profile_generator(
        generator=generator,
        statistics=long_stats,
        n_series=2000,  # Увеличено с 500 до 2000 для репрезентативности
        save_dir=save_dir,
        n_bins=40,
        show=False,
    )
    
    # profile_generator возвращает dict, CSV сохраняется внутри функции
    # dfs["observed"] содержит данные для observed уровня
    df_observed = dfs["observed"]
    print(f"   Сохранено: {save_dir / 'profile_data.csv'}")
    
    # 2. Многомерные метрики
    print("\n[2/2] Вычисление многомерных метрик...")
    
    # Генерируем батч для многомерного анализа
    print("   Генерация батча (B=200, T~100, D~4)...")
    mv_data = generate_multivariate_batch(
        config=config_example,
        n_series=200,  # Увеличено с 50 до 200 для репрезентативности
        batch_size=config_example.batch_size,
        device="cpu",
        seed=42,
    )
    
    print(f"   Форма батча: {mv_data.shape}")
    
    # Вычисляем метрики
    summary, raw_results = compute_multivariate_metrics(
        mv_data,
        depth=2,  # depth=2 для D=3 даёт sig_length=12
        max_lag=5,
    )
    
    # Сохраняем агрегированные метрики
    summary_df = pd.DataFrame([summary])
    summary_csv_path = save_dir / "multivariate_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"   Сохранено (summary): {summary_csv_path}")
    
    # Сохраняем сырые данные для визуализации
    # Формат: каждая колонка - одна компонента метрики, каждая строка - наблюдение
    def extract_arrays(results: dict, prefix: str = "") -> dict:
        """Извлекает массивы из результатов."""
        arrays = {}
        for key, arr in results.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                arrays[f"{prefix}{key}"] = arr
            elif isinstance(arr, np.ndarray) and arr.ndim == 2:
                # Для 2D массивов сохраняем каждую колонку
                for i in range(arr.shape[1]):
                    arrays[f"{prefix}{key}_{i}"] = arr[:, i]
        return arrays
    
    synth_arrays = extract_arrays(raw_results, "synthetic_")
    real_arrays = extract_arrays(raw_results, "real_")
    
    # Находим максимальную длину
    max_len = max((len(v) for v in synth_arrays.values()), default=0)
    max_len = max(max_len, max((len(v) for v in real_arrays.values()), default=0))
    
    # Создаем DataFrame с выравниванием по длине
    all_arrays = {**synth_arrays, **real_arrays}
    df_dict = {}
    for key, arr in all_arrays.items():
        if len(arr) < max_len:
            # Дополняем NaN
            df_dict[key] = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
        else:
            df_dict[key] = arr[:max_len]
    
    raw_df = pd.DataFrame(df_dict)
    raw_csv_path = save_dir / "multivariate_metrics_raw.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"   Сохранено (raw): {raw_csv_path}")
    
    # Печатаем сводку
    print("\n" + "=" * 70)
    print("МНОГОМЕРНЫЕ МЕТРИКИ (синтетические данные)")
    print("=" * 70)
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
    
    print("\n" + "=" * 70)
    print(f"ГОТОВО. Результаты в: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
