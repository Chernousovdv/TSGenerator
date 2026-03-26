from calibration.analyze_latent import StatSpec
from calibration.stats import (
    mean,
    std,
    acf,
    adf_statistic,
    permutation_entropy,
    mann_kendall_z,
    roughness,
    forecastability,
    fft_mean,
    seasonality_strength,
    trend_strength,
)

short_stats = [
    StatSpec("acf_lag1", lambda x: acf(x, lags=[1])[:, 0]),
    StatSpec("adf_ct", lambda x: adf_statistic(x, regression="ct")),
    StatSpec("perm_ent_m3", lambda x: permutation_entropy(x, m=3)),
    StatSpec("mk_z", mann_kendall_z),
    StatSpec("seasonality", seasonality_strength),
    StatSpec("forecastability", forecastability),
    StatSpec("fft_mean", fft_mean),
]

us = [
    StatSpec("fft_mean", fft_mean),
]

long_stats = [
    StatSpec("acf_lag1", lambda x: acf(x, lags=[1])[:, 0]),
    StatSpec("adf_ct", lambda x: adf_statistic(x, regression="ct")),
    StatSpec("perm_ent_m3", lambda x: permutation_entropy(x, m=3)),
    StatSpec("mk_z", mann_kendall_z),
    StatSpec("seasonality", seasonality_strength),
    StatSpec("forecastability", forecastability),
    StatSpec("fft_mean", fft_mean),
]


long_long_stats = [
    # --- Уровень / амплитуда ---
    StatSpec("mean", mean),
    StatSpec("std", std),
    # --- Автокорреляция ---
    StatSpec("acf_lag1", lambda x: acf(x, lags=[1])[:, 0]),
    StatSpec("acf_lag2", lambda x: acf(x, lags=[2])[:, 0]),
    StatSpec("acf_lag7", lambda x: acf(x, lags=[7])[:, 0]),
    # StatSpec("acf_lag365",        lambda x: acf(x, lags=[365])[:, 0]),
    # --- Стационарность ---
    StatSpec("adf_ct", lambda x: adf_statistic(x, regression="ct")),
    StatSpec("adf_c", lambda x: adf_statistic(x, regression="c")),
    # --- Сложность / энтропия ---
    StatSpec("perm_ent_m3", lambda x: permutation_entropy(x, m=3)),
    # StatSpec("perm_ent_m5",       lambda x: permutation_entropy(x, m=5)),
    # --- Тренд ---
    StatSpec("mk_z", mann_kendall_z),
    StatSpec("trend_str", trend_strength),
    # --- Сезонность ---
    StatSpec("seasonality", seasonality_strength),
    # --- Гладкость ---
    StatSpec("roughness", roughness),
    # --- Предсказуемость (спектральная) ---
    StatSpec("forecastability", forecastability),
    # StatSpec("forecastability_raw", lambda x: forecastability(x, detrend=False)),
    # --- Спектральная мощность ---
    StatSpec("fft_mean", fft_mean),
    # StatSpec("fft_mean_raw",      lambda x: fft_mean(x, detrend=False)),
]
