"""
calibration — Модули для калибровки генератора синтетических данных.

Подмодули:
----------
- stats: Одномерные статистики временных рядов (10 функций)
- multivariate_analysis: Многомерные статистики и анализ
- analyze_latent: Профилирование латентных компонент
- analyze_generator: Профилирование полного пайплайна генератора
- compare_profiles: Сравнение профилей статистик
"""

from calibration.stats import (
    mean, std, acf, adf_statistic, permutation_entropy,
    mann_kendall_z, roughness, forecastability, fft_mean,
    seasonality_strength, trend_strength,
)

from calibration.multivariate_analysis import (
    MultivariateAnalyzer,
    SignatureConfig,
    CrossCorrelationConfig,
    ProjectionConfig,
    analyze_with_projections,
    analyze_all_5d_combinations,
    compute_signature,
    compute_cross_correlations,
    compute_correlation_rank,
    compute_levy_area,
    signature_mmd_loss,
    signature_kernel_mmd_loss,
)

from calibration.analyze_latent import StatSpec, profile_generator
from calibration.analyze_generator import analyze_generator

__all__ = [
    # Одномерные статистики
    'mean', 'std', 'acf', 'adf_statistic', 'permutation_entropy',
    'mann_kendall_z', 'roughness', 'forecastability', 'fft_mean',
    'seasonality_strength', 'trend_strength',
    
    # Многомерные статистики и анализ
    'MultivariateAnalyzer',
    'SignatureConfig',
    'CrossCorrelationConfig',
    'ProjectionConfig',
    'analyze_with_projections',
    'analyze_all_5d_combinations',
    'compute_signature',
    'compute_cross_correlations',
    'compute_correlation_rank',
    'compute_levy_area',
    'signature_mmd_loss',
    'signature_kernel_mmd_loss',
    
    # Профилирование
    'StatSpec',
    'profile_generator',
    'analyze_generator',
]
