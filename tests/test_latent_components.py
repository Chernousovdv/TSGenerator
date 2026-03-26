#!/usr/bin/env python3
"""
Тестирование латентных компонент генератора синтетических временных рядов.

Включает:
- Базовое тестирование компонент
- Профилирование времени выполнения
- Расширенные статистики
- Визуализация результатов
- Анализ чувствительности к параметрам

Режимы работы (аргумент --mode):
- basic: Базовое тестирование + визуализация + анализ чувствительности
- full: Полный набор тестов + профилирование + статистики
- profile: Только профилирование времени выполнения

Запуск:
    cd ts_generator
    python tests/test_latent_components.py --mode full
    python tests/test_latent_components.py --mode basic
    python tests/test_latent_components.py --mode profile
"""

import torch
import os
import json
import numpy as np
from datetime import datetime
from typing import Optional
from pathlib import Path

from universal_test_utils import (
    test_latent_component,
    create_latent_component_report,
    visualize_combined_batch,
    plot_arima_parameter_sensitivity,
    plot_kernel_parameter_sensitivity,
    plot_tsi_parameter_sensitivity,
    plot_ets_parameter_sensitivity,
    profile_all_components,
    plot_profiling_results,
    compute_component_statistics,
    parse_tsf_file,
    plot_real_dataset_histograms,
    compute_fast_statistics,
    compute_extended_statistics,
)
from modules.latent import ARIMASpec, KernelSynthSpec, TSISpec, ETSSpec


def test_all_components():
    """Базовое тестирование всех компонент с генерацией отчетов."""
    print("🧪 Тестирование латентных компонент")
    print("=" * 60)
    
    results = {}
    
    # ARIMA
    print("\n📊 ARIMA...")
    arima_spec = ARIMASpec(
        type="arima",
        ar_params=torch.tensor([0.5, -0.3]),
        ma_params=torch.tensor([0.4]),
        d=1,
        intercept=0.1,
        sigma=0.2,
        burn_in=50
    )
    results["arima"] = test_latent_component(arima_spec)
    create_latent_component_report(arima_spec, save_dir="./test_reports")
    print(f"   Среднее={results['arima']['mean_value']:.3f}, "
          f"Std={results['arima']['std_value']:.3f}")
    
    # KernelSynth
    print("\n📊 KernelSynth...")
    kernel_spec = KernelSynthSpec(
        type="kernel_synth",
        kernel_type="RBF",
        lengthscale=0.3,
        variance=1.5
    )
    results["kernel"] = test_latent_component(kernel_spec)
    create_latent_component_report(kernel_spec, save_dir="./test_reports")
    print(f"   Среднее={results['kernel']['mean_value']:.3f}, "
          f"Std={results['kernel']['std_value']:.3f}")
    
    # TSI
    print("\n📊 TSI...")
    tsi_spec = TSISpec(
        type="tsi",
        frequencies=[2.0, 5.0],
        amplitudes=[1.0, 0.5],
        phases=[0.0, 1.57],
        decays=[0.1, 0.2]
    )
    results["tsi"] = test_latent_component(tsi_spec)
    create_latent_component_report(tsi_spec, save_dir="./test_reports")
    print(f"   Среднее={results['tsi']['mean_value']:.3f}, "
          f"Std={results['tsi']['std_value']:.3f}")
    
    # ETS
    print("\n📊 ETS...")
    ets_spec = ETSSpec(
        type="ets",
        model_type="AAN",
        alpha=0.3,
        beta=0.1,
        initial_level=1.0,
        initial_trend=0.05
    )
    results["ets"] = test_latent_component(ets_spec)
    create_latent_component_report(ets_spec, save_dir="./test_reports")
    print(f"   Среднее={results['ets']['mean_value']:.3f}, "
          f"Std={results['ets']['std_value']:.3f}")
    
    return results


def run_combined_visualization(save_dir: str = "./test_reports"):
    """
    Визуализация примера латентных траекторий всех 4 типов в одном батче
    через LatentDynamics.visualize().
    """
    print("\n" + "=" * 60)
    print("🎨 Визуализация комбинированного батча (LatentDynamics.visualize)")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "combined_batch.png")
    
    print("\n📈 Генерация комбинированной визуализации...")
    visualize_combined_batch(T=100, save_path=save_path)
    print(f"✅ Сохранено: {save_path}")
    
    return save_path


def run_parameter_sensitivity_analysis(save_dir: str = "./test_reports"):
    """Запуск анализа чувствительности всех компонент по всем параметрам."""
    print("\n" + "=" * 60)
    print("📊 Анализ чувствительности к параметрам")
    print("=" * 60)
    
    print("\n📈 ARIMA (6 параметров)...")
    plot_arima_parameter_sensitivity(save_dir)
    
    print("📈 KernelSynth (9 параметров)...")
    plot_kernel_parameter_sensitivity(save_dir)
    
    print("📈 TSI (4 параметра)...")
    plot_tsi_parameter_sensitivity(save_dir)
    
    print("📈 ETS (8 параметров)...")
    plot_ets_parameter_sensitivity(save_dir)
    
    print(f"\n✅ Графики сохранены в: {save_dir}")


def run_profiling(save_dir: str = "./test_reports", T: int = 100,
                  num_samples: int = 10, device: str = "cpu"):
    """Профилирование времени выполнения компонент."""
    print("\n" + "=" * 60)
    print("⏱️  Профилирование времени выполнения")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📊 Параметры: T={T}, сэмплов={num_samples}, device={device}")
    profile_results = profile_all_components(T, num_samples, device)
    
    print("\n📈 Результаты профилирования:")
    print("-" * 60)
    print(f"{'Компонента':<15} {'Время (с)':<12} {'Сэмплов/сек':<12} {'мс/шаг':<10}")
    print("-" * 60)
    for name, result in profile_results.items():
        print(f"{name:<15} {result.total_time_sec:<12.4f} "
              f"{result.samples_per_sec:<12.1f} {result.avg_time_per_step_sec*1000:<10.4f}")
    
    # Визуализация
    save_path = os.path.join(save_dir, "profiling_results.png")
    plot_profiling_results(profile_results, save_path=save_path)
    print(f"\n✅ График профилирования: {save_path}")
    
    return profile_results


def run_extended_statistics(save_dir: str = "./test_reports", T: int = 100,
                            num_samples: int = 10, device: str = "cpu"):
    """Вычисление расширенных статистик для всех компонент."""
    print("\n" + "=" * 60)
    print("📈 Расширенные статистики компонент")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    stats_results = {}
    
    # ARIMA
    print("\n📊 ARIMA...")
    arima_spec = ARIMASpec(type="arima", ar_params=torch.tensor([0.5, -0.3]),
                           ma_params=torch.tensor([0.4]), d=1, intercept=0.1,
                           sigma=0.3, burn_in=50)
    stats_results["arima"] = compute_component_statistics(arima_spec, T, num_samples, device)
    print(f"   ADF p-value: {stats_results['arima']['adf_pvalue_mean']:.3f}, "
          f"Hurst: {stats_results['arima']['hurst_exponent_mean']:.3f}")
    
    # KernelSynth
    print("📊 KernelSynth...")
    kernel_spec = KernelSynthSpec(type="kernel_synth", kernel_type="RBF",
                                  lengthscale=0.3, variance=1.5)
    stats_results["kernel_synth"] = compute_component_statistics(
        kernel_spec, T, num_samples, device)
    print(f"   ADF p-value: {stats_results['kernel_synth']['adf_pvalue_mean']:.3f}, "
          f"Hurst: {stats_results['kernel_synth']['hurst_exponent_mean']:.3f}")
    
    # TSI
    print("📊 TSI...")
    tsi_spec = TSISpec(type="tsi", frequencies=[2.0, 5.0], amplitudes=[1.0, 0.5],
                       phases=[0.0, 1.57], decays=[0.1, 0.2])
    stats_results["tsi"] = compute_component_statistics(tsi_spec, T, num_samples, device)
    print(f"   ADF p-value: {stats_results['tsi']['adf_pvalue_mean']:.3f}, "
          f"Hurst: {stats_results['tsi']['hurst_exponent_mean']:.3f}")
    
    # ETS
    print("📊 ETS...")
    ets_spec = ETSSpec(type="ets", model_type="AAN", alpha=0.3, beta=0.1,
                       initial_level=1.0, initial_trend=0.05)
    stats_results["ets"] = compute_component_statistics(ets_spec, T, num_samples, device)
    print(f"   ADF p-value: {stats_results['ets']['adf_pvalue_mean']:.3f}, "
          f"Hurst: {stats_results['ets']['hurst_exponent_mean']:.3f}")
    
    # Сохранение отчета
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(save_dir, f"extended_stats_{timestamp}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Отчет сохранен: {report_path}")
    
    return stats_results


def run_full_test_suite(save_dir: str = "./test_reports", T: int = 100,
                        num_samples: int = 10, device: str = "cpu"):
    """Полный набор тестов: базовое + профилирование + статистики."""
    print("🚀 Полный набор тестов генератора")
    print("=" * 60)
    
    # 1. Базовое тестирование
    print("\n[1/4] Базовое тестирование компонент...")
    results = test_all_components()
    
    # 2. Визуализация
    print("\n[2/4] Визуализация...")
    run_combined_visualization(save_dir)
    
    # 3. Профилирование
    print("\n[3/4] Профилирование...")
    profile_results = run_profiling(save_dir, T, num_samples, device)
    
    # 4. Расширенные статистики
    print("\n[4/4] Расширенные статистики...")
    stats_results = run_extended_statistics(save_dir, T, num_samples, device)
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("✅ ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ")
    print("=" * 60)
    
    print("\n📊 Итоговая таблица:")
    print(f"{'Компонента':<15} {'Среднее':<10} {'Std':<10} {'ADF p-val':<10}")
    print("-" * 50)
    for name, res in results.items():
        stats_name = {"arima": "arima", "kernel": "kernel_synth",
                      "tsi": "tsi", "ets": "ets"}[name]
        adf = stats_results.get(stats_name, {}).get('adf_pvalue_mean', 0)
        print(f"{name:<15} {res['mean_value']:<10.3f} "
              f"{res['std_value']:<10.3f} {adf:<10.3f}")
    
    print(f"\n📁 Результаты сохранены в: {save_dir}/")
    
    return {
        "basic_results": results,
        "profile_results": {k: {
            "total_time_sec": v.total_time_sec,
            "samples_per_sec": v.samples_per_sec,
            "avg_time_per_step_sec": v.avg_time_per_step_sec
        } for k, v in profile_results.items()},
        "statistics": stats_results
    }


def run_real_dataset_analysis(
    tsf_path: str,
    category_name: Optional[str] = None,
    save_dir: str = "./test_reports/real_dataset_stats",
    max_series: int = 100,
    fast_mode: bool = False
):
    """
    Анализ реального датасета: загрузка TSF, вычисление статистик, гистограммы.
    
    Args:
        tsf_path: Путь к TSF файлу
        category_name: Имя категории (по умолчанию - имя файла)
        save_dir: Директория для результатов
        max_series: Максимум рядов для обработки
        fast_mode: Если True, вычислять только 6 статистик для гистограмм
    """
    print("\n" + "=" * 60)
    print("📊 Анализ реального датасета" + (" (БЫСТРЫЙ РЕЖИМ)" if fast_mode else ""))
    print("=" * 60)
    
    filepath = Path(tsf_path)
    if not filepath.exists():
        print(f"❌ Файл не найден: {filepath}")
        return
    
    if category_name is None:
        category_name = filepath.stem.replace('_dataset', '')
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка данных
    series_list = parse_tsf_file(filepath)
    if not series_list:
        return
    
    # Вычисление статистик
    print(f"\n📈 Вычисление статистик для {min(len(series_list), max_series)} рядов...")
    all_stats = []
    
    compute_func = compute_fast_statistics if fast_mode else compute_extended_statistics
    
    for i, series in enumerate(series_list[:max_series]):
        stats = compute_func(series)
        all_stats.append(stats)
        
        if (i + 1) % 100 == 0:
            print(f"   Обработано: {i + 1}/{min(len(series_list), max_series)}")
    
    print(f"   ✅ Обработано: {len(all_stats)} рядов")
    
    results = {
        "num_series": len(all_stats),
        "all_stats": all_stats,
        "category_name": category_name
    }
    
    # Гистограммы
    plot_path = save_dir / f"{category_name}_histograms.png"
    plot_real_dataset_histograms(
        all_stats, 
        category_name, 
        str(plot_path)
    )
    
    # JSON отчет с полными статистиками
    report_path = save_dir / f"{category_name}_summary.json"
    
    # Вычисляем сводные статистики для каждой метрики
    summary_stats = {}
    for key in ['mean', 'std', 'acf_lag1', 'perm_ent', 'mk_z', 'roughness']:
        values = [s.get(key, 0) for s in all_stats]
        if values:
            summary_stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75))
            }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "category": category_name,
            "num_series": len(all_stats),
            "fast_mode": fast_mode,
            "summary": summary_stats
        }, f, indent=2, ensure_ascii=False, default=str)
    
    # Вывод сводки
    print("\n" + "=" * 60)
    print("📊 Сводка:")
    print("=" * 60)
    print(f"{'Статистика':<15} {'Mean':<12} {'Std':<12} {'Median':<12}")
    print("-" * 55)
    
    for key in ['mean', 'std', 'acf_lag1', 'perm_ent', 'mk_z', 'roughness']:
        values = [s.get(key, 0) for s in all_stats]
        if values:
            print(f"{key:<15} {np.mean(values):<12.4f} "
                  f"{np.std(values):<12.4f} {np.median(values):<12.4f}")
    
    print(f"\n✅ Результаты: {save_dir}")
    print("=" * 60)
    
    return results


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Тестирование латентных компонент генератора")
    parser.add_argument("--mode", choices=["basic", "full", "profile", "real"], 
                        default="full",
                        help="Режим тестирования")
    parser.add_argument("--T", type=int, default=100,
                        help="Длина временного ряда")
    parser.add_argument("--samples", type=int, default=10,
                        help="Число сэмплов для статистик")
    parser.add_argument("--device", default="cpu",
                        help="Устройство (cpu/cuda)")
    parser.add_argument("--save-dir", default="./test_reports",
                        help="Директория для результатов")
    parser.add_argument("--tsf-path", type=str, default=None,
                        help="Путь к TSF файлу (для режима real)")
    parser.add_argument("--category", type=str, default=None,
                        help="Имя категории (для режима real)")
    parser.add_argument("--max-series", type=int, default=100,
                        help="Максимум рядов для обработки (для режима real)")
    parser.add_argument("--fast", action="store_true",
                        help="Быстрый режим (только 6 статистик для гистограмм)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "basic":
            test_all_components()
            run_combined_visualization(args.save_dir)
            run_parameter_sensitivity_analysis(args.save_dir)
        
        elif args.mode == "full":
            results = run_full_test_suite(args.save_dir, args.T, args.samples, args.device)
        
        elif args.mode == "profile":
            # Только профилирование времени
            run_profiling(args.save_dir, T=args.T, num_samples=args.samples, 
                          device=args.device)
        
        elif args.mode == "real":
            # Анализ реального датасета
            if args.tsf_path is None:
                print("❌ Для режима real необходимо указать --tsf-path")
                return
            
            run_real_dataset_analysis(
                tsf_path=args.tsf_path,
                category_name=args.category,
                save_dir=args.save_dir,
                max_series=args.max_series,
                fast_mode=args.fast
            )
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
