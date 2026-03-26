"""
comp_prof.py — Сравнение профилей синтетических и реальных данных.

Включает:
1. Сравнение одномерных статистик (гистограммы)
2. Сравнение многомерных метрик (зеркальные гистограммы)

Использование:
    cd ts_generator
    python comp_prof.py [--mode diverse|financial|all]

Режимы:
    diverse     — сбалансированная выборка (по умолчанию)
    financial   — финансовые и бизнес-датасеты
    all         — все доступные датасеты
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from calibration.compare_profiles import compare_profiles, _get_finite, _robust_range, _add_kde
from configs.stat_lists import long_stats

base_dir = Path(__file__).parent

# ---------------------------------------------------------------------------
# Определяем режим из аргументов командной строки
# ---------------------------------------------------------------------------

mode = "diverse"  # режим по умолчанию
if len(sys.argv) > 1:
    arg = sys.argv[1]
    # Поддерживаем как --mode diverse, так и просто diverse
    if arg == "--mode" and len(sys.argv) > 2:
        arg = sys.argv[2]
    if arg in ("diverse", "financial", "all"):
        mode = arg
    else:
        print(f"Неизвестный режим: {arg}")
        print("Использование: python comp_prof.py [--mode diverse|financial|all]")
        print("           или: python comp_prof.py diverse|financial|all")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Пути к данным (зависят от режима)
# ---------------------------------------------------------------------------

# Директория для реальных данных зависит от режима
real_profile_dir = base_dir / "results" / "monash_profile" / mode
real_mv_dir = base_dir / "results" / "monash_profile" / mode

# Ищем последний запуск генератора (run_X)
generator_dir = base_dir / "results" / "generator"
if generator_dir.exists():
    run_dirs = [d for d in generator_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if run_dirs:
        run_dirs.sort(key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0)
        generator_dir = run_dirs[-1]
        print(f"Используем последний запуск генератора: {generator_dir}")
    synth_profile_path = generator_dir / "profile_data.csv"
else:
    synth_profile_path = base_dir / "results" / "generator" / "profile_data.csv"

# Пути к файлам реальных данных (зависят от режима)
real_profile_path = real_profile_dir / "monash_all.csv"
synth_mv_path = base_dir / "results" / "generator" / "multivariate_metrics_raw.csv"
real_mv_path = real_mv_dir / "multivariate_metrics_raw.csv"

# ---------------------------------------------------------------------------
# 1. Сравнение одномерных статистик
# ---------------------------------------------------------------------------

print("=" * 70)
print("СРАВНЕНИЕ ПРОФИЛЕЙ: СИНТЕТИЧЕСКИЕ vs РЕАЛЬНЫЕ ДАННЫЕ")
print("=" * 70)

print("\n[1/3] Загрузка данных для сравнения...")

print(f"Загрузка синтетических данных из {synth_profile_path}")
df_synth = pd.read_csv(synth_profile_path).query("level == 'observed'")

print(f"Загрузка реальных данных из {real_profile_path}")
df_real = pd.read_csv(real_profile_path)

# Директория для результатов сравнения (зависит от режима)
compare_dir = base_dir / "results" / f"comparison_{mode}"
compare_dir.mkdir(parents=True, exist_ok=True)

print("\n[2/3] Сравнение одномерных статистик...")

fig = compare_profiles(
    df_synthetic=df_synth,
    df_real=df_real,
    statistics=long_stats,
    save_path=compare_dir / "comparison.png",
    n_bins=40,
    label_synthetic="synthetic",
    label_real=f"real (Monash {mode})",
    show=False,
)

print(f"Сохранено: {compare_dir}/comparison.png")

# ---------------------------------------------------------------------------
# 2. Сравнение многомерных метрик — гистограммы распределений
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("СРАВНЕНИЕ МНОГОМЕРНЫХ МЕТРИК (распределения)")
print("=" * 70)

try:
    print("\nЗагрузка сырых данных для многомерных метрик...")
    
    if synth_mv_path.exists():
        synth_raw = pd.read_csv(synth_mv_path)
        print(f"  Синтетические: {synth_mv_path} ({len(synth_raw)} строк)")
    else:
        print(f"  Файл не найден: {synth_mv_path}")
        synth_raw = None
    
    if real_mv_path.exists():
        real_raw = pd.read_csv(real_mv_path)
        print(f"  Реальные: {real_mv_path} ({len(real_raw)} строк)")
    else:
        print(f"  Файл не найден: {real_mv_path}")
        real_raw = None
    
    if synth_raw is not None and real_raw is not None:
        print("\n[3/3] Построение зеркальных гистограмм многомерных метрик...")
        
        # Группируем колонки по типу метрики
        def group_columns(df):
            """
            Группирует колонки по типу метрики.
            
            Поддерживаемые категории:
            - signature: компоненты сигнатуры (signature_0, signature_1, ...)
            - signature_energy: энергия по уровням
            - signature_entropy: энтропия сигнатуры
            - brownian_distance: расстояние до броуновского движения
            - gram_eigenvalues: собственные значения Грама
            - cross_corr: кросс-корреляции
            - rank_ratio: ранг корреляционной матрицы
            - condition_number: число обусловленности
            - effective_rank: эффективный ранг
            - levy_area: площади Леви
            - rotation_number: число вращения
            - total_levy: общая площадь Леви
            """
            groups = {}
            for col in df.columns:
                # Определяем категорию метрики
                if "signature_energy" in col:
                    key = "signature_energy"
                elif "signature_entropy" in col:
                    key = "signature_entropy"
                elif "brownian_distance" in col:
                    key = "brownian_distance"
                elif "gram_eigenvalues" in col:
                    key = "gram_eigenvalues"
                elif "signature_" in col and "wasserstein" not in col:
                    key = "signature"
                elif "cross_corr" in col and "wasserstein" not in col:
                    key = "cross_corr"
                elif "condition_number" in col:
                    key = "condition_number"
                elif "effective_rank" in col:
                    key = "effective_rank"
                elif "rank_ratio" in col:
                    key = "rank_ratio"
                elif "rotation_number" in col:
                    key = "rotation_number"
                elif "total_levy" in col:
                    key = "total_levy"
                elif "levy_area_" in col:
                    key = "levy_area"
                elif "wasserstein" in col:
                    key = "wasserstein"
                else:
                    continue
                
                if key not in groups:
                    groups[key] = []
                groups[key].append(col)
            return groups
        
        synth_groups = group_columns(synth_raw)
        real_groups = group_columns(real_raw)
        
        # Создаем фигуру с подграфиками
        # Включаем ВСЕ доступные метрики из групп
        all_available_metrics = list(set(synth_groups.keys()) | set(real_groups.keys()))
        
        # Приоритетный порядок метрик
        # Исключаем gram_eigenvalues (не считается для реальных данных) и rank_ratio (не информативен из-за нормализации)
        excluded_metrics = {"gram_eigenvalues", "rank_ratio"}
        priority_order = [
            "signature", "signature_energy", "signature_entropy",
            "brownian_distance",
            "cross_corr", "condition_number", "effective_rank",
            "levy_area", "rotation_number", "total_levy",
            "wasserstein"
        ]
        
        # Сортируем метрики по приоритету, исключая ненужные
        metrics_to_plot = sorted(
            [m for m in all_available_metrics if m not in excluded_metrics],
            key=lambda x: (priority_order.index(x) if x in priority_order else 999)
        )
        
        # Ограничиваем число метрик для читаемости (максимум 10)
        metrics_to_plot = metrics_to_plot[:10]
        
        n_cols = 3
        n_rows = (len(metrics_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        color_synth = "#4C72B0"  # синий
        color_real = "#DD8452"   # оранжевый
        n_bins = 40
        clip_pct = 1.0
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            synth_cols = synth_groups.get(metric, [])
            real_cols = real_groups.get(metric, [])
            
            # Собираем все значения
            synth_vals = synth_raw[synth_cols].values.ravel() if synth_cols else np.array([])
            real_vals = real_raw[real_cols].values.ravel() if real_cols else np.array([])
            
            synth_vals = synth_vals[np.isfinite(synth_vals)]
            real_vals = real_vals[np.isfinite(real_vals)]
            
            if synth_vals.size == 0 and real_vals.size == 0:
                ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_title(metric, fontsize=10, fontweight="bold")
                continue
            
            # Общий робастный диапазон
            combined = np.concatenate([synth_vals, real_vals]) if (synth_vals.size and real_vals.size) \
                       else (synth_vals if synth_vals.size else real_vals)
            lo, hi = _robust_range(combined, clip_pct)
            
            bins = np.linspace(lo, hi, n_bins + 1)
            bin_w = bins[1] - bins[0]
            
            # Гистограммы (зеркальные)
            if synth_vals.size:
                counts_s, _ = np.histogram(np.clip(synth_vals, lo, hi), bins=bins)
                density_s = counts_s / (synth_vals.size * bin_w)
                ax.bar(bins[:-1], density_s, width=bin_w,
                       color=color_synth, alpha=0.55, align="edge",
                       label='synthetic')
            
            if real_vals.size:
                counts_r, _ = np.histogram(np.clip(real_vals, lo, hi), bins=bins)
                density_r = counts_r / (real_vals.size * bin_w)
                ax.bar(bins[:-1], -density_r, width=bin_w,
                       color=color_real, alpha=0.55, align="edge",
                       label='real')
            
            # KDE поверх
            xs = np.linspace(lo, hi, 300)
            _add_kde(ax, synth_vals, xs, color_synth, positive=True)
            _add_kde(ax, real_vals, xs, color_real, positive=False)
            
            # Средние значения (пунктирные линии)
            if synth_vals.size:
                mean_s = np.mean(synth_vals)
                ax.axvline(mean_s, color=color_synth, linewidth=1.4, linestyle="--", alpha=0.9)
            if real_vals.size:
                mean_r = np.mean(real_vals)
                ax.axvline(mean_r, color=color_real, linewidth=1.4, linestyle="--", alpha=0.9)
            
            # Нулевая линия
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
            
            # Подписи и форматирование
            ax.set_xlim(lo, hi)
            ax.set_xlabel(metric, fontsize=9)
            ax.set_title(f'{metric} distribution', fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{abs(v):.2g}")
            )
            
            # Аннотация N
            note_parts = []
            if synth_vals.size: note_parts.append(f"synth N={synth_vals.size}")
            if real_vals.size: note_parts.append(f"real N={real_vals.size}")
            ax.text(0.98, 0.98, "  ".join(note_parts),
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color="gray")
            
            # Подписи сторон
            y_lim = ax.get_ylim()
            if synth_vals.size and y_lim[1] > 0:
                ax.text(0.01, 0.97, "↑ synthetic", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7,
                        color=color_synth, alpha=0.8)
            if real_vals.size and y_lim[0] < 0:
                ax.text(0.01, 0.03, "↓ real", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=7,
                        color=color_real, alpha=0.8)
        
        # Удаляем пустые подграфики
        for idx in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[idx])
        
        # Общая легенда
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_synth, alpha=0.6),
            plt.Rectangle((0, 0), 1, 1, color=color_real, alpha=0.6),
        ]
        fig.legend(handles, ['synthetic', 'real (Monash)'],
                   loc="lower center", ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02), framealpha=0.8)
        
        fig.suptitle(f"Multivariate Metrics Comparison ({mode})", fontsize=12, fontweight="bold", y=1.01)
        fig.tight_layout()
        save_path = compare_dir / "multivariate_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
        plt.close(fig)
        
        # Таблица агрегированных метрик
        print("\n" + "-" * 70)
        print("АГРЕГИРОВАННЫЕ МЕТРИКИ")
        print("-" * 70)
        print(f"{'Метрика':<40} | {'Синт.':<15} | {'Реал.':<15} | {'Разн.':<10}")
        print("-" * 70)
        
        def agg_stats(df, cols):
            """Вычисляет mean и std для набора колонок."""
            if not cols:
                return None, None
            vals = df[cols].values.ravel()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return None, None
            return float(np.mean(vals)), float(np.std(vals))
        
        metrics_summary = {}
        for metric in metrics_to_plot:
            synth_cols = synth_groups.get(metric, [])
            real_cols = real_groups.get(metric, [])
            
            synth_mean, synth_std = agg_stats(synth_raw, synth_cols)
            real_mean, real_std = agg_stats(real_raw, real_cols)
            
            metrics_summary[f"{metric}_mean"] = (synth_mean, real_mean)
            metrics_summary[f"{metric}_std"] = (synth_std, real_std)
        
        for name, (synth_val, real_val) in metrics_summary.items():
            if synth_val is not None and real_val is not None:
                diff = abs(synth_val - real_val)
                print(f"{name:<40} | {synth_val:<15.6f} | {real_val:<15.6f} | {diff:<10.6f}")
            elif synth_val is not None:
                print(f"{name:<40} | {synth_val:<15.6f} | {'N/A':<15} | {'-':<10}")
            elif real_val is not None:
                print(f"{name:<40} | {'N/A':<15} | {real_val:<15.6f} | {'-':<10}")
        
        print("-" * 70)
        
except Exception as e:
    print(f"Ошибка при работе с многомерными метриками: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ГОТОВО!")
print("=" * 70)
print(f"\nРежим: {mode}")
print("\nРезультаты:")
print(f"  - {compare_dir}/comparison.png              — сравнение гистограмм (1D)")
print(f"  - {compare_dir}/multivariate_comparison.png — сравнение гистограмм (многомерные)")
print("=" * 70)
