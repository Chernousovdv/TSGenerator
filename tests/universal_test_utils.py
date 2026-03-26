#!/usr/bin/env python3
"""
Универсальные утилиты для тестирования и визуализации латентных компонент.

Включает:
- Тестирование компонент с метриками
- Визуализация рядов
- Генерация отчетов
- Профилирование времени выполнения
- Расширенные статистики временных рядов
- Загрузка и анализ реальных датасетов (Monash TSF)
"""

import os
import json
import time
import math
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.latent import (
    LatentDynamics,
    LatentModulePlan,
    LatentComponentSpec,
    ARIMASpec, KernelSynthSpec, TSISpec, ETSSpec,
    ARIMAModule, KernelSynthModule, TSIModule, ETSModule
)


# =============================================================================
# Импорты для статистик
# =============================================================================

try:
    from statsmodels.tsa.stattools import adfuller, acf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    adfuller = None
    acf = None

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None


# =============================================================================
# Профилирование времени выполнения
# =============================================================================

@dataclass
class ProfileResult:
    """Результат профилирования одной компоненты."""
    component_type: str
    T: int
    num_samples: int
    total_time_sec: float
    avg_time_per_sample_sec: float
    avg_time_per_step_sec: float
    samples_per_sec: float


def profile_component(
    component_spec,
    T: int = 100,
    num_samples: int = 10,
    device: str = "cpu",
    warmup: int = 3
) -> ProfileResult:
    """Профилирование времени выполнения компоненты."""
    if component_spec.type == "arima":
        module = ARIMAModule(device)
    elif component_spec.type == "kernel_synth":
        module = KernelSynthModule(device)
    elif component_spec.type == "tsi":
        module = TSIModule(device)
    elif component_spec.type == "ets":
        module = ETSModule(device)
    else:
        raise ValueError(f"Неизвестный тип: {component_spec.type}")
    
    for _ in range(warmup):
        _ = module.execute(T, [component_spec])
    
    start_time = time.perf_counter()
    for _ in range(num_samples):
        _ = module.execute(T, [component_spec])
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    
    return ProfileResult(
        component_type=component_spec.type,
        T=T,
        num_samples=num_samples,
        total_time_sec=total_time,
        avg_time_per_sample_sec=total_time / num_samples,
        avg_time_per_step_sec=total_time / (num_samples * T),
        samples_per_sec=num_samples / total_time
    )


def profile_all_components(
    T: int = 100,
    num_samples: int = 10,
    device: str = "cpu"
) -> Dict[str, ProfileResult]:
    """Профилирование всех компонент с типовыми параметрами."""
    results = {}
    
    results["arima"] = profile_component(
        ARIMASpec(type="arima", ar_params=torch.tensor([0.5, -0.3]),
                  ma_params=torch.tensor([0.4]), d=1, intercept=0.1,
                  sigma=0.3, burn_in=50),
        T, num_samples, device)
    
    results["kernel_synth"] = profile_component(
        KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                        lengthscale=0.3, variance=1.5),
        T, num_samples, device)
    
    results["tsi"] = profile_component(
        TSISpec(type="tsi", frequencies=[2.0, 5.0], amplitudes=[1.0, 0.5],
                phases=[0.0, 1.57], decays=[0.1, 0.2]),
        T, num_samples, device)
    
    results["ets"] = profile_component(
        ETSSpec(type="ets", model_type="AAN", alpha=0.3, beta=0.1,
                initial_level=1.0, initial_trend=0.05),
        T, num_samples, device)
    
    return results


def plot_profiling_results(
    results: Dict[str, ProfileResult],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Визуализация результатов профилирования."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    names = list(results.keys())
    total_times = [r.total_time_sec for r in results.values()]
    samples_per_sec = [r.samples_per_sec for r in results.values()]
    time_per_step = [r.avg_time_per_step_sec * 1000 for r in results.values()]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    ax = axes[0]
    bars = ax.bar(names, total_times, color=colors)
    ax.set_ylabel('Время (сек)', fontsize=11)
    ax.set_title('Общее время (10 сэмплов)', fontsize=12)
    ax.set_ylim(0, max(total_times) * 1.3)
    for bar, val in zip(bars, total_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax = axes[1]
    bars = ax.bar(names, samples_per_sec, color=colors)
    ax.set_ylabel('Сэмплов/сек', fontsize=11)
    ax.set_title('Производительность', fontsize=12)
    ax.set_ylim(0, max(samples_per_sec) * 1.15)
    for bar, val in zip(bars, samples_per_sec):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    ax = axes[2]
    bars = ax.bar(names, time_per_step, color=colors)
    ax.set_ylabel('Время на шаг (мс)', fontsize=11)
    ax.set_title('Время генерации одного шага', fontsize=12)
    ax.set_ylim(0, max(time_per_step) * 1.3)
    for bar, val in zip(bars, time_per_step):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# =============================================================================
# Расширенные статистики временных рядов
# =============================================================================

def compute_extended_statistics(series: np.ndarray) -> Dict[str, Any]:
    """
    Вычисление расширенных статистик для временного ряда.
    
    Args:
        series: Временной ряд (T,)
        
    Returns:
        Словарь со статистиками
    """
    T = len(series)
    
    # Базовые статистики
    mean_val = float(np.mean(series))
    std_val = float(np.std(series))
    min_val = float(np.min(series))
    max_val = float(np.max(series))
    
    if HAS_SCIPY:
        skewness_val = float(stats.skew(series))
        kurtosis_val = float(stats.kurtosis(series))
    else:
        skewness_val, kurtosis_val = 0.0, 0.0
    
    # ADF тест
    adf_stat, adf_pval, is_stationary = 0.0, 0.5, False
    if HAS_STATSMODELS and T > 10:
        try:
            maxlag = min(12, T//4) if T > 20 else 1
            adf_result = adfuller(series, maxlag=maxlag, autolag='AIC')
            adf_stat = float(adf_result[0])
            adf_pval = float(adf_result[1])
            is_stationary = adf_pval < 0.05
        except:
            pass
    
    # Нормальность
    normality_pval = 1.0
    if HAS_SCIPY and T >= 8:
        try:
            _, normality_pval = stats.normaltest(series)
            normality_pval = float(normality_pval)
        except:
            normality_pval = 0.0
    
    # ACF
    acf_lag1, acf_lag5, acf_lag10 = 0.0, 0.0, 0.0
    if HAS_STATSMODELS:
        try:
            acf_vals = acf(series - mean_val, nlags=max(10, T//4), fft=True)
            acf_lag1 = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
            acf_lag5 = float(acf_vals[5]) if len(acf_vals) > 5 else 0.0
            acf_lag10 = float(acf_vals[10]) if len(acf_vals) > 10 else 0.0
        except:
            pass
    
    # Спектральный анализ
    spectral_entropy, dominant_freq = 0.0, 0.0
    try:
        fft_vals = np.fft.fft(series - mean_val)
        freqs = np.fft.fftfreq(T)
        power_spectrum = np.abs(fft_vals[:T//2])**2
        power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1
        dominant_freq = float(abs(freqs[dominant_freq_idx]))
    except:
        pass
    
    # Sample Entropy
    sample_entropy_val = _compute_sample_entropy(series)
    
    # Hurst Exponent
    hurst_val = _compute_hurst_exponent(series)
    
    # Тренд
    trend_slope = 0.0
    if HAS_SCIPY:
        try:
            x = np.arange(T)
            trend_slope, _, _, _, _ = stats.linregress(x, series)
            trend_slope = float(trend_slope)
        except:
            pass
    
    # Mann-Kendall
    mk_tau, mk_pval = _mann_kendall_test(series)
    
    # Сезонность
    detected_period, seasonality_strength = 0, 0.0
    try:
        detrended = series - np.mean(series)
        window = np.hamming(T)
        detrended_windowed = detrended * window
        fft_vals = np.fft.fft(detrended_windowed)
        freqs = np.fft.fftfreq(T)
        power = np.abs(fft_vals[:T//2])**2
        power[0] = 0
        
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        
        if dominant_freq > 0:
            detected_period = int(round(1.0 / abs(dominant_freq)))
        
        total_power = np.sum(power)
        if total_power > 0:
            seasonality_strength = float(power[dominant_idx] / total_power)
            median_power = np.median(power[1:])
            if power[dominant_idx] > 3 * median_power:
                seasonality_strength = min(1.0, seasonality_strength * 2)
            else:
                seasonality_strength *= 0.5
        
        if detected_period < 2 or detected_period > T // 2:
            detected_period = 0
            seasonality_strength = 0.0
    except:
        pass
    
    return {
        "mean": mean_val, "std": std_val, "min": min_val, "max": max_val,
        "skewness": skewness_val, "kurtosis": kurtosis_val,
        "adf_statistic": adf_stat, "adf_pvalue": adf_pval,
        "is_stationary": is_stationary, "normality_pvalue": normality_pval,
        "acf_lag1": acf_lag1, "acf_lag5": acf_lag5, "acf_lag10": acf_lag10,
        "spectral_entropy": float(spectral_entropy),
        "dominant_frequency": dominant_freq,
        "sample_entropy": float(sample_entropy_val),
        "hurst_exponent": float(hurst_val),
        "trend_slope": trend_slope,
        "mann_kendall_tau": float(mk_tau),
        "mann_kendall_pvalue": float(mk_pval),
        "seasonality_strength": seasonality_strength,
        "detected_period": detected_period
    }


def compute_fast_statistics(series: np.ndarray) -> Dict[str, Any]:
    """
    Быстрое вычисление только тех статистик, которые нужны для гистограмм.
    
    Статистики для гистограмм: mean, std, acf_lag1, perm_ent, mk_z, roughness
    
    Args:
        series: Временной ряд (T,)
        
    Returns:
        Словарь с 6 статистиками
    """
    T = len(series)
    
    # Базовые статистики (быстро)
    mean_val = float(np.mean(series))
    std_val = float(np.std(series))
    
    # Roughness (быстро)
    if T > 1:
        diff = np.diff(series)
        roughness_val = float(np.sqrt(np.mean(diff ** 2)))
    else:
        roughness_val = 0.0
    
    # ACF lag-1 (быстро, если есть statsmodels)
    acf_lag1 = 0.0
    if HAS_STATSMODELS and T > 3:
        try:
            acf_vals = acf(series - mean_val, nlags=1, fft=True)
            acf_lag1 = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
        except:
            pass
    
    # Permutation Entropy (средняя скорость)
    perm_ent_val = _compute_permutation_entropy_fast(series)
    
    # Mann-Kendall Z (быстрая версия)
    mk_z_val = _mann_kendall_z_fast(series)
    
    return {
        "mean": mean_val,
        "std": std_val,
        "acf_lag1": acf_lag1,
        "perm_ent": perm_ent_val,
        "mk_z": mk_z_val,
        "roughness": roughness_val
    }


def _compute_permutation_entropy_fast(series: np.ndarray, order: int = 3) -> float:
    """Permutation entropy (быстрая версия)."""
    n = len(series)
    if n < order + 1:
        return 0.0
    
    # Извлекаем паттерны
    patterns = {}
    for i in range(n - order):
        window = series[i:i + order + 1]
        pattern = tuple(np.argsort(window))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    # Вычисляем энтропию
    total = sum(patterns.values())
    entropy = 0.0
    for count in patterns.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)
    
    # Нормализуем
    max_entropy = math.log(math.factorial(order))
    if max_entropy > 0:
        entropy /= max_entropy
    
    return float(entropy)


def _mann_kendall_z_fast(series: np.ndarray, max_length: int = 200) -> float:
    """
    Mann-Kendall Z-value (быстрая версия).
    
    Для длинных рядов используется субсэмплинг.
    """
    T = len(series)
    
    # Для длинных рядов - субсэмплинг
    if T > max_length:
        indices = np.linspace(0, T-1, max_length, dtype=int)
        series = series[indices]
        T = max_length
    
    # Векторизованная версия
    n = T
    S = 0
    for i in range(n-1):
        diff = series[i+1:] - series[i]
        S += np.sum(np.sign(diff))
    
    var_S = n * (n-1) * (2*n + 5) / 18
    z = S / np.sqrt(var_S) if var_S > 0 else 0
    
    return float(z)


def _compute_sample_entropy(series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy."""
    T = len(series)
    std_val = np.std(series)
    r = r * std_val
    
    def count_matches(data, template_len):
        matches, total = 0, 0
        for i in range(T - template_len - 1):
            template = data[i:i+template_len]
            for j in range(i+1, T - template_len):
                if np.max(np.abs(template - data[j:j+template_len])) < r:
                    matches += 1
                total += 1
        return matches, total
    
    matches_m, total_m = count_matches(series, m)
    matches_m1, total_m1 = count_matches(series, m+1)
    
    if matches_m == 0 or matches_m1 == 0:
        return 0.0
    return float(-np.log((matches_m1/total_m1) / (matches_m/total_m)))


def _compute_hurst_exponent(series: np.ndarray) -> float:
    """Hurst exponent via classic R/S analysis."""
    T = len(series)
    if T < 20:
        return 0.5
    
    data = series.astype(float)
    n_windows = 8
    min_n = max(10, T // 20)
    max_n = T // 2
    
    n_values = np.unique(np.logspace(
        np.log10(min_n), np.log10(max_n), 
        num=n_windows, dtype=int
    ))
    
    rs_values = []
    for n in n_values:
        if n < 5 or n > T // 2:
            continue
        n_segments = T // n
        if n_segments < 2:
            continue
        
        rs_sum = 0.0
        count = 0
        for i in range(n_segments):
            segment = data[i * n:(i + 1) * n]
            mean_seg = np.mean(segment)
            cumsum = np.cumsum(segment - mean_seg)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(segment)
            
            if S < 1e-10:
                continue
            rs_sum += R / S
            count += 1
        
        if count > 0:
            rs_values.append(rs_sum / count)
    
    if len(rs_values) < 2:
        return 0.5
    
    try:
        log_n = np.log(n_values[:len(rs_values)])
        log_rs = np.log(rs_values)
        slope = np.polyfit(log_n, log_rs, 1)[0]
        hurst = float(slope)
        return max(0.0, min(1.0, hurst))
    except:
        return 0.5


def _mann_kendall_test(series: np.ndarray, max_length: int = 500) -> tuple:
    """
    Mann-Kendall test (оптимизированная версия).
    
    Для длинных рядов используется субсэмплинг для ускорения.
    """
    T = len(series)
    
    # Для длинных рядов - субсэмплинг
    if T > max_length:
        indices = np.linspace(0, T-1, max_length, dtype=int)
        series = series[indices]
        T = max_length
    
    # Векторизованная версия
    n = T
    S = 0
    for i in range(n-1):
        diff = series[i+1:] - series[i]
        S += np.sum(np.sign(diff))
    
    var_S = n * (n-1) * (2*n + 5) / 18
    z = S / np.sqrt(var_S) if var_S > 0 else 0
    
    if HAS_SCIPY:
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        pval = 1.0
    
    tau = S / (n * (n-1) / 2) if n > 1 else 0
    return tau, pval


def compute_component_statistics(
    component_spec,
    T: int = 100,
    num_samples: int = 10,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Вычисление расширенных статистик для компоненты."""
    if component_spec.type == "arima":
        module = ARIMAModule(device)
    elif component_spec.type == "kernel_synth":
        module = KernelSynthModule(device)
    elif component_spec.type == "tsi":
        module = TSIModule(device)
    elif component_spec.type == "ets":
        module = ETSModule(device)
    else:
        raise ValueError(f"Неизвестный тип: {component_spec.type}")
    
    all_stats = []
    for i in range(num_samples):
        torch.manual_seed(42 + i)
        result = module.execute(T, [component_spec])
        series = result.cpu().numpy()[0]
        stats_dict = compute_extended_statistics(series)
        all_stats.append(stats_dict)
    
    avg_stats = {}
    for key in all_stats[0].keys():
        values = [s[key] for s in all_stats]
        avg_stats[f"{key}_mean"] = float(np.mean(values))
        avg_stats[f"{key}_std"] = float(np.std(values))
    
    return avg_stats


# =============================================================================
# Загрузка данных из TSF (Monash Repository)
# =============================================================================

def parse_tsf_file(filepath: Path) -> List[np.ndarray]:
    """
    Парсинг TSF файла и извлечение всех временных рядов.
    
    Args:
        filepath: Путь к TSF файлу
        
    Returns:
        List[np.ndarray]: Список временных рядов
    """
    print(f"📊 Чтение TSF: {filepath.name}")
    
    series_list = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    data_lines = [line for line in lines 
                  if not line.startswith('@') and line.strip()]
    
    for line in data_lines:
        if ':' not in line:
            continue
        
        parts = line.strip().split(':')
        if len(parts) < 2:
            continue
        
        data_part = ':'.join(parts[1:])
        values_str = data_part.split(',')
        
        values = []
        for val_str in values_str[1:]:
            try:
                val = float(val_str.strip())
                if not np.isnan(val) and not np.isinf(val):
                    values.append(val)
            except ValueError:
                continue
        
        if len(values) > 50:
            series = np.array(values)
            series_list.append(series)
    
    print(f"   ✅ Извлечено рядов: {len(series_list)}")
    return series_list


def compute_real_dataset_statistics_from_series(
    series_list: List[np.ndarray],
    max_series: int = 100
) -> Dict[str, Any]:
    """
    Вычисление статистик для списка временных рядов.
    
    Args:
        series_list: Список временных рядов
        max_series: Максимальное число рядов для обработки
        
    Returns:
        Словарь со статистиками по всем рядам
    """
    all_stats = []
    
    for i, series in enumerate(series_list[:max_series]):
        stats = compute_extended_statistics(series)
        all_stats.append(stats)
        
        if (i + 1) % 50 == 0:
            print(f"   Обработано: {i + 1}/{min(len(series_list), max_series)}")
    
    # Агрегация
    summary = {}
    for key in all_stats[0].keys():
        values = [s.get(key, 0) for s in all_stats if key in s]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
    
    return {
        "num_series": len(all_stats),
        "all_stats": all_stats,
        "summary": summary
    }


def load_and_compute_tsf_stats(
    tsf_path: str,
    max_series: int = 100
) -> Dict[str, Any]:
    """
    Загрузка TSF файла и вычисление статистик.
    
    Args:
        tsf_path: Путь к TSF файлу
        max_series: Максимальное число рядов
        
    Returns:
        Словарь со статистиками
    """
    filepath = Path(tsf_path)
    if not filepath.exists():
        print(f"❌ Файл не найден: {filepath}")
        return {}
    
    series_list = parse_tsf_file(filepath)
    if not series_list:
        return {}
    
    category_name = filepath.stem.replace('_dataset', '')
    results = compute_real_dataset_statistics_from_series(series_list, max_series)
    results["category_name"] = category_name
    
    return results


# =============================================================================
# Визуализация статистик реальных данных (гистограммы)
# =============================================================================

def plot_real_dataset_histograms(
    all_stats: List[Dict[str, float]],
    category_name: str,
    save_path: Optional[str] = None,
    confidence_interval: float = 0.95
):
    """
    Построение гистограмм для статистик реального датасета.
    
    Args:
        all_stats: Список статистик для каждого ряда
        category_name: Имя категории датасета
        save_path: Путь для сохранения графика
        confidence_interval: Доверительный интервал (по умолчанию 98%)
    
    Стиль:
    - 2x3 сетка: mean, std, acf_lag1, mk_z, roughness, perm_ent
    - Данные обрезаны по доверительному интервалу (указано на графике)
    - Медиана показана красной пунктирной линией
    """
    n_total = len(all_stats)
    alpha = (1 - confidence_interval) / 2
    lower_pct = alpha * 100
    upper_pct = (1 - alpha) * 100
    
    # Извлекаем данные с обрезкой по доверительному интервалу
    data = {}
    data_full = {}  # Сохраняем полные данные для подсчета
    for stat_key in ['mean', 'std', 'acf_lag1', 'perm_ent', 'mk_z', 'roughness']:
        values = [s.get(stat_key, 0) for s in all_stats]
        values = np.array(values)
        data_full[stat_key] = values.copy()
        if len(values) > 0:
            q_lower, q_upper = np.percentile(values, [lower_pct, upper_pct])
            mask = (values >= q_lower) & (values <= q_upper)
            data[stat_key] = values[mask]
        else:
            data[stat_key] = values
    
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Заголовок с указанием доверительного интервала
    fig.suptitle(
        f'Real Dataset: {category_name} | N={n_total} series | '
        f'Data trimmed to {confidence_interval*100:.0f}% CI [{lower_pct:.0f}%-{upper_pct:.0f}%]',
        fontsize=14, y=0.995
    )
    
    def plot_histogram(ax, data, data_full, title, xlabel):
        """Построение гистограммы с обрезкой выбросов."""
        n_retained = len(data)
        n_total = len(data_full)
        
        if n_retained == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        
        median_val = np.median(data)
        
        # Гистограмма
        sns.histplot(data, kde=True, color='steelblue', ax=ax, bins=40, 
                     alpha=0.8, line_kws={'linewidth': 2, 'color': 'darkblue'})
        
        # Медиана
        ax.axvline(median_val, color='crimson', linestyle='--', linewidth=2,
                   label=f'Median = {median_val:.3f}')
        
        # Границы доверительного интервала
        if n_retained > 0:
            q_lower, q_upper = np.percentile(data, [lower_pct, upper_pct])
            ax.axvline(q_lower, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axvline(q_upper, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Статистика в углу
        stats_text = f'n = {n_retained}/{n_total}\n'
        if n_retained > 0:
            stats_text += f'mean = {np.mean(data):.3f}\nstd = {np.std(data):.3f}'
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plot_histogram(axes[0, 0], data['mean'], data_full['mean'], 
                   'Mean (Среднее)', 'mean')
    plot_histogram(axes[0, 1], data['std'], data_full['std'], 
                   'Std (Стандартное отклонение)', 'std')
    plot_histogram(axes[0, 2], data['acf_lag1'], data_full['acf_lag1'], 
                   'ACF Lag-1 (Автокорреляция)', 'acf_lag1')
    plot_histogram(axes[1, 0], data['mk_z'], data_full['mk_z'], 
                   'Mann-Kendall Z (Тренд)', 'mk_z')
    plot_histogram(axes[1, 1], data['roughness'], data_full['roughness'], 
                   'Roughness (Шероховатость)', 'roughness')
    plot_histogram(axes[1, 2], data['perm_ent'], data_full['perm_ent'], 
                   'Permutation Entropy', 'perm_ent')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"   ✅ Сохранено: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_and_plot_real_dataset(
    tsf_path: str,
    category_name: Optional[str] = None,
    save_dir: str = "./test_reports/real_dataset_stats",
    max_series: int = 100
) -> Dict[str, Any]:
    """
    Полная обработка реального датасета: загрузка, статистики, гистограммы.
    
    Args:
        tsf_path: Путь к TSF файлу
        category_name: Имя категории
        save_dir: Директория для результатов
        max_series: Максимум рядов для обработки
        
    Returns:
        Словарь с результатами
    """
    filepath = Path(tsf_path)
    if not filepath.exists():
        print(f"❌ Файл не найден: {filepath}")
        return {}
    
    if category_name is None:
        category_name = filepath.stem.replace('_dataset', '')
    
    print("=" * 60)
    print(f"📊 Обработка датасета: {category_name}")
    print("=" * 60)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка и статистики
    series_list = parse_tsf_file(filepath)
    if not series_list:
        return {}
    
    results = compute_real_dataset_statistics_from_series(series_list, max_series)
    results["category_name"] = category_name
    
    # Гистограммы
    plot_path = save_dir / f"{category_name}_histograms.png"
    plot_real_dataset_histograms(
        results["all_stats"], 
        category_name, 
        str(plot_path)
    )
    
    # JSON отчет
    report_path = save_dir / f"{category_name}_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "category": category_name,
            "num_series": results["num_series"],
            "summary": results["summary"]
        }, f, indent=2, ensure_ascii=False, default=str)
    
    # Вывод сводки
    print("\n" + "=" * 60)
    print("📊 Сводка:")
    print("=" * 60)
    print(f"{'Статистика':<15} {'Mean':<12} {'Std':<12} {'Median':<12}")
    print("-" * 55)
    
    summary = results["summary"]
    for key in ['mean', 'std', 'acf_lag1', 'perm_ent', 'mk_z', 'roughness']:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        med_key = f"{key}_median"
        if mean_key in summary:
            print(f"{key:<15} {summary[mean_key]:<12.4f} "
                  f"{summary[std_key]:<12.4f} {summary[med_key]:<12.4f}")
    
    print(f"\n✅ Результаты: {save_dir}")
    print("=" * 60)
    
    return results


# =============================================================================
# Тестирование латентных компонент
# =============================================================================

def test_latent_component(
    component_spec: LatentComponentSpec,
    T: int = 100,
    device: str = "cpu",
    num_samples: int = 10
) -> dict:
    """Тестирование латентной компоненты с метриками."""
    from scipy import stats
    
    if component_spec.type == "arima":
        module = ARIMAModule(device)
    elif component_spec.type == "kernel_synth":
        module = KernelSynthModule(device)
    elif component_spec.type == "tsi":
        module = TSIModule(device)
    elif component_spec.type == "ets":
        module = ETSModule(device)
    else:
        raise ValueError(f"Неизвестный тип: {component_spec.type}")
    
    samples = []
    for i in range(num_samples):
        torch.manual_seed(42 + i)
        result = module.execute(T, [component_spec])
        samples.append(result.cpu().numpy()[0])
    
    samples = np.array(samples)
    mean_series = np.mean(samples, axis=0)
    std_series = np.std(samples, axis=0)
    
    normality_p_values = []
    for t in range(0, T, max(1, T // 10)):
        if len(samples[:, t]) >= 8:
            _, p_value = stats.normaltest(samples[:, t])
            normality_p_values.append(p_value)
    
    avg_normality_p = np.mean(normality_p_values) if normality_p_values else 0.0
    
    segment_size = max(10, T // 5)
    variances = []
    for i in range(0, T, segment_size):
        if i + segment_size <= T:
            variances.append(np.var(mean_series[i:i + segment_size]))
    
    variance_stability = np.std(variances) / (np.mean(variances) + 1e-8) if variances else 0.0
    
    return {
        "component_type": component_spec.type,
        "series_shape": mean_series.shape,
        "mean_value": float(np.mean(mean_series)),
        "std_value": float(np.std(mean_series)),
        "min_value": float(np.min(samples)),
        "max_value": float(np.max(samples)),
        "normality_test_p_value": float(avg_normality_p),
        "variance_stability_index": float(variance_stability),
        "num_samples": num_samples,
        "sample_series": mean_series,
        "std_series": std_series
    }


def visualize_latent_component(
    component_spec: LatentComponentSpec,
    T: int = 100,
    device: str = "cpu",
    num_samples: int = 10,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Визуализация латентной компоненты с 10 сэмплами."""
    test_results = test_latent_component(component_spec, T, device, num_samples)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    time_grid = np.linspace(0, 1, T)
    
    if component_spec.type == "arima":
        module_instance = ARIMAModule(device)
    elif component_spec.type == "kernel_synth":
        module_instance = KernelSynthModule(device)
    elif component_spec.type == "tsi":
        module_instance = TSIModule(device)
    elif component_spec.type == "ets":
        module_instance = ETSModule(device)
    else:
        raise ValueError(f"Неизвестный тип: {component_spec.type}")
    
    for i in range(num_samples):
        torch.manual_seed(42 + i)
        result = module_instance.execute(T, [component_spec])
        sample_series = result.cpu().numpy()[0]
        ax1.plot(time_grid, sample_series, alpha=0.4, linewidth=1)
    
    mean_series = test_results["sample_series"]
    std_series = test_results["std_series"]
    
    ax1.plot(time_grid, mean_series, 'k-', linewidth=2.5, label='Mean')
    ax1.fill_between(time_grid, 
                     mean_series - 2 * std_series,
                     mean_series + 2 * std_series,
                     alpha=0.3, color='gray', label='95% CI')
    
    ax1.set_title(title if title else f'{component_spec.type.upper()}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(mean_series, bins=50, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def create_latent_component_report(
    component_spec: LatentComponentSpec,
    T: int = 100,
    device: str = "cpu",
    num_samples: int = 10,
    save_dir: str = "./test_reports"
) -> dict:
    """Создание отчета о компоненте."""
    os.makedirs(save_dir, exist_ok=True)
    test_results = test_latent_component(component_spec, T, device, num_samples)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    component_name = f"{component_spec.type}_{timestamp}"
    
    viz_path = os.path.join(save_dir, f"{component_name}_visualization.png")
    fig = visualize_latent_component(
        component_spec, T, device, num_samples, 
        save_path=viz_path,
        title=f"{component_spec.type.upper()} Analysis"
    )
    plt.close(fig)
    
    if hasattr(component_spec, "model_dump"):
        params = component_spec.model_dump()
        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                params[key] = value.tolist()
    else:
        params = {}
    
    report = {
        "timestamp": timestamp,
        "component_type": test_results["component_type"],
        "parameters": params,
        "test_results": {
            "series_shape": list(test_results["series_shape"]),
            "mean_value": test_results["mean_value"],
            "std_value": test_results["std_value"],
            "min_value": test_results["min_value"],
            "max_value": test_results["max_value"],
            "normality_test_p_value": test_results["normality_test_p_value"],
            "variance_stability_index": test_results["variance_stability_index"],
            "num_samples": test_results["num_samples"]
        },
        "visualization_path": viz_path,
        "interpretation": _generate_interpretation(test_results)
    }
    
    report_path = os.path.join(save_dir, f"{component_name}_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    return report


def _generate_interpretation(test_results: dict) -> dict:
    """Интерпретация результатов."""
    p_value = test_results["normality_test_p_value"]
    stability_index = test_results["variance_stability_index"]
    std_value = test_results["std_value"]
    
    return {
        "normality": {
            "p_value": p_value,
            "interpretation": "Нормальное" if p_value > 0.05 else "Отличается от нормального"
        },
        "stability": {
            "index": stability_index,
            "interpretation": "Хорошая" if stability_index < 0.1 else 
                             ("Умеренная" if stability_index < 0.3 else "Сильная нестационарность")
        },
        "amplitude": {
            "std": std_value,
            "interpretation": "Низкая" if std_value < 0.5 else 
                             ("Средняя" if std_value < 1.5 else "Высокая")
        }
    }


def visualize_combined_batch(
    T: int = 100,
    device: str = "cpu",
    save_path: Optional[str] = None,
    seed: int = 42
) -> plt.Figure:
    """
    Визуализация примера латентных траекторий всех 4 типов в одном батче.
    """
    torch.manual_seed(seed)
    
    latent_dynamics = LatentDynamics(device)
    
    plan = LatentModulePlan(
        items=[
            [ARIMASpec(type="arima", ar_params=torch.tensor([0.5, -0.3]),
                      ma_params=torch.tensor([0.4]), d=1, intercept=0.1,
                      sigma=0.3, burn_in=50)],
            [KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                            lengthscale=0.3, variance=1.5)],
            [KernelSynthSpec(type="kernel_synth", kernel_type="Periodic",
                            lengthscale=0.5, variance=1.0, period=0.2)],
            [ETSSpec(type="ets", model_type="AAN", alpha=0.3, beta=0.1,
                    initial_level=1.0, initial_trend=0.05)],
            [TSISpec(type="tsi", frequencies=[2.0, 5.0], amplitudes=[1.0, 0.5],
                    phases=[0.0, 1.57], decays=[0.1, 0.2])],
        ]
    )
    
    fig = latent_dynamics.visualize(B=5, T=T, plan=plan, on_the_same_axes=True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# Анализ чувствительности к параметрам
# =============================================================================

def plot_arima_parameter_sensitivity(save_dir: str = "./test_reports"):
    """Анализ чувствительности ARIMA по всем параметрам."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    T, device = 100, "cpu"
    colors = ['blue', 'green', 'orange', 'red']
    
    ax = axes[0, 0]
    for ar_val, color in zip([0.0, 0.3, 0.6, 0.9], colors):
        torch.manual_seed(42)
        spec = ARIMASpec(type="arima", ar_params=torch.tensor([ar_val]),
                        ma_params=torch.tensor([0.2]), d=0, sigma=0.5, burn_in=50)
        result = ARIMAModule(device).execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'AR=[{ar_val}]')
    ax.set_title('AR параметры')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for ma_val, color in zip([0.0, 0.3, 0.6, 0.9], colors):
        torch.manual_seed(42)
        spec = ARIMASpec(type="arima", ar_params=torch.tensor([0.3]),
                        ma_params=torch.tensor([ma_val]), d=0, sigma=0.5, burn_in=50)
        result = ARIMAModule(device).execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'MA=[{ma_val}]')
    ax.set_title('MA параметры')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    for d_val, color in zip([0, 1, 2, 3], colors):
        torch.manual_seed(42)
        spec = ARIMASpec(type="arima", ar_params=torch.tensor([0.5]),
                        ma_params=torch.tensor([0.2]), d=d_val, sigma=0.5, burn_in=50)
        result = ARIMAModule(device).execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'd={d_val}')
    ax.set_title('Параметр d')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for c_val, color in zip([0.0, 0.5, 1.0, 2.0], colors):
        torch.manual_seed(42)
        spec = ARIMASpec(type="arima", ar_params=torch.tensor([0.5]),
                        ma_params=torch.tensor([0.2]), d=0, intercept=c_val, sigma=0.5, burn_in=50)
        result = ARIMAModule(device).execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'intercept={c_val}')
    ax.set_title('Intercept')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for sigma_val, color in zip([0.1, 0.5, 1.0, 2.0], colors):
        torch.manual_seed(42)
        spec = ARIMASpec(type="arima", ar_params=torch.tensor([0.5]),
                        ma_params=torch.tensor([0.2]), d=0, sigma=sigma_val, burn_in=50)
        result = ARIMAModule(device).execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'sigma={sigma_val}')
    ax.set_title('Sigma')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    for burn_val, color in zip([10, 30, 50, 100], colors):
        torch.manual_seed(42)
        spec = ARIMASpec(type="arima", ar_params=torch.tensor([0.8]),
                        ma_params=torch.tensor([0.0]), d=0, sigma=1.0, burn_in=burn_val)
        result = ARIMAModule(device).execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'burn_in={burn_val}')
    ax.set_title('Burn-in')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "arima_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_kernel_parameter_sensitivity(save_dir: str = "./test_reports"):
    """Анализ чувствительности KernelSynth по всем параметрам."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    T, device = 100, "cpu"
    colors = ['blue', 'green', 'orange', 'red']
    module = KernelSynthModule(device)
    
    ax = axes[0, 0]
    for ls_val, color in zip([0.1, 0.3, 0.5, 1.0], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                              lengthscale=ls_val, variance=1.0)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'l={ls_val}')
    ax.set_title('Lengthscale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for var_val, color in zip([0.5, 1.0, 2.0, 5.0], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                              lengthscale=0.5, variance=var_val)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'var={var_val}')
    ax.set_title('Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    for kt, color in zip(['RBF', 'Periodic', 'RQ', 'Linear'], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type=kt,
                              lengthscale=0.5, variance=1.0,
                              period=0.2 if kt == 'Periodic' else 1.0)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=kt)
    ax.set_title('Kernel Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for p_val, color in zip([0.1, 0.2, 0.5, 1.0], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type="Periodic",
                              lengthscale=0.5, variance=1.0, period=p_val)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'p={p_val}')
    ax.set_title('Period')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for alpha_val, color in zip([0.5, 1.0, 2.0, 5.0], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type="RQ",
                              lengthscale=0.5, variance=1.0, alpha=alpha_val)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'alpha={alpha_val}')
    ax.set_title('Alpha (RQ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    for a_val, color in zip([0.0, 1.0, 2.0, 5.0], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                              lengthscale=0.5, variance=0.1,
                              mean_a=a_val, mean_b=0, mean_c=0, mean_d=0)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'mean_a={a_val}')
    ax.set_title('mean_a (linear)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    for b_val, color in zip([0.0, 1.0, 2.0, 5.0], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                              lengthscale=0.5, variance=0.1,
                              mean_a=0, mean_b=b_val, mean_c=0, mean_d=0)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'mean_b={b_val}')
    ax.set_title('mean_b (const)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    for c_val, color in zip([0.0, 0.5, 1.0, 2.0], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                              lengthscale=0.5, variance=0.1,
                              mean_a=0, mean_b=0, mean_c=c_val, mean_d=1.0)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'mean_c={c_val}')
    ax.set_title('mean_c (exp)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 2]
    for d_val, color in zip([-1.0, -0.5, 0.5, 1.0], colors):
        torch.manual_seed(42)
        spec = KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                              lengthscale=0.5, variance=0.1,
                              mean_a=0, mean_b=0, mean_c=1.0, mean_d=d_val)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'mean_d={d_val}')
    ax.set_title('mean_d (exp power)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "kernel_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_tsi_parameter_sensitivity(save_dir: str = "./test_reports"):
    """Анализ чувствительности TSI по всем параметрам."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    T, device = 100, "cpu"
    colors = ['blue', 'green', 'orange', 'red']
    module = TSIModule(device)
    
    ax = axes[0, 0]
    for f_val, color in zip([1.0, 3.0, 5.0, 10.0], colors):
        torch.manual_seed(42)
        spec = TSISpec(type="tsi", frequencies=[f_val], amplitudes=[1.0],
                      phases=[0.0], decays=[0.0])
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'f={f_val}')
    ax.set_title('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for a_val, color in zip([0.5, 1.0, 2.0, 5.0], colors):
        torch.manual_seed(42)
        spec = TSISpec(type="tsi", frequencies=[5.0], amplitudes=[a_val],
                      phases=[0.0], decays=[0.0])
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'A={a_val}')
    ax.set_title('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for p_val, color in zip([0.0, 0.5*np.pi, np.pi, 1.5*np.pi], colors):
        torch.manual_seed(42)
        spec = TSISpec(type="tsi", frequencies=[5.0], amplitudes=[1.0],
                      phases=[p_val], decays=[0.0])
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'phi={p_val:.2f}')
    ax.set_title('Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for d_val, color in zip([0.0, 0.5, 1.0, 2.0], colors):
        torch.manual_seed(42)
        spec = TSISpec(type="tsi", frequencies=[5.0], amplitudes=[1.0],
                      phases=[0.0], decays=[d_val])
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'decay={d_val}')
    ax.set_title('Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "tsi_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_ets_parameter_sensitivity(save_dir: str = "./test_reports"):
    """Анализ чувствительности ETS по всем параметрам."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    T, device = 100, "cpu"
    colors = ['blue', 'green', 'orange', 'red']
    module = ETSModule(device)
    
    ax = axes[0, 0]
    for alpha_val, color in zip([0.1, 0.3, 0.5, 0.9], colors):
        torch.manual_seed(42)
        spec = ETSSpec(type="ets", model_type="AAN", alpha=alpha_val,
                      beta=0.1, initial_level=1.0, initial_trend=0.05)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'alpha={alpha_val}')
    ax.set_title('Alpha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for beta_val, color in zip([0.0, 0.1, 0.3, 0.5], colors):
        torch.manual_seed(42)
        spec = ETSSpec(type="ets", model_type="AAN", alpha=0.3,
                      beta=beta_val, initial_level=1.0, initial_trend=0.05)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'beta={beta_val}')
    ax.set_title('Beta')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    for gamma_val, color in zip([0.0, 0.1, 0.3, 0.5], colors):
        torch.manual_seed(42)
        spec = ETSSpec(type="ets", model_type="AAA", alpha=0.3,
                      beta=0.1, gamma=gamma_val,
                      seasonality_period=12,
                      initial_level=1.0, initial_trend=0.05,
                      initial_seasonal=[0.1]*12)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'gamma={gamma_val}')
    ax.set_title('Gamma')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 3]
    for m_val, color in zip([0, 6, 12, 24], colors):
        torch.manual_seed(42)
        spec = ETSSpec(type="ets", model_type="AAA", alpha=0.3,
                      beta=0.1, gamma=0.1,
                      seasonality_period=m_val if m_val > 0 else 1,
                      initial_level=1.0, initial_trend=0.05,
                      initial_seasonal=[0.1]*max(1, m_val))
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'm={m_val}')
    ax.set_title('Season Period')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for l_val, color in zip([0.0, 1.0, 5.0, 10.0], colors):
        torch.manual_seed(42)
        spec = ETSSpec(type="ets", model_type="AAN", alpha=0.3,
                      beta=0.1, initial_level=l_val, initial_trend=0.05)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'level={l_val}')
    ax.set_title('Initial Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for t_val, color in zip([-0.1, -0.05, 0.05, 0.1], colors):
        torch.manual_seed(42)
        spec = ETSSpec(type="ets", model_type="AAN", alpha=0.3,
                      beta=0.1, initial_level=1.0, initial_trend=t_val)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'trend={t_val}')
    ax.set_title('Initial Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    for mt, color in zip(['ANN', 'AAN', 'ANA', 'AAA'], colors):
        torch.manual_seed(42)
        try:
            spec = ETSSpec(type="ets", model_type=mt, alpha=0.3,
                          beta=0.1 if 'A' in mt[1] else 0.0,
                          gamma=0.1 if 'A' in mt[2] else 0.0,
                          seasonality_period=12 if 'A' in mt[2] else 0,
                          initial_level=1.0, initial_trend=0.05,
                          initial_seasonal=[0.1]*12 if 'A' in mt[2] else [])
            result = module.execute(T, [spec]).cpu().numpy()[0]
            ax.plot(result, color=color, alpha=0.8, label=mt)
        except Exception:
            pass
    ax.set_title('Model Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 3]
    for s_val, color in zip([0.0, 0.5, 1.0, 2.0], colors):
        torch.manual_seed(42)
        spec = ETSSpec(type="ets", model_type="AAA", alpha=0.3,
                      beta=0.1, gamma=0.1,
                      seasonality_period=12,
                      initial_level=1.0, initial_trend=0.05,
                      initial_seasonal=[s_val]*12)
        result = module.execute(T, [spec]).cpu().numpy()[0]
        ax.plot(result, color=color, alpha=0.8, label=f'seasonal={s_val}')
    ax.set_title('Initial Seasonal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "ets_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
