from modules.latent import (
    ARIMAPrior,
    LatentDynamics,
    LatentPrior,
    Range,
    KernelSynthPrior,
    TSIPrior,
    ETSPrior,
)
from modules.noise_module import NoisePrior
from modules.transformations import Range, SmoothingPrior, TransformationPrior
from sampler import GeneratorConfig, TSGenerator


# =============================================================================
# КАЛИБРОВКА v3: Агрессивное увеличение амплитуды и нелинейности
# Цели:
# - signature_energy: 27% → 50-60% (агрессивно увеличить амплитуду)
# - cross_corr: 43% → 60% (больше латентных компонент)
# - rotation_number: 40% → 60% (максимальная периодичность)
# - signature_entropy: 70% → 85% (глубокий граф + нелинейность)
# =============================================================================
transform_prior = TransformationPrior(
    max_depth=10,  # Увеличено с 7 до 10 для максимальной сложности
    node_kind_probs={"op": 0.60, "terminal": 0.25, "constant": 0.15},  # Ещё больше операций
    op_weights={
        "add": 0.20,     # Уменьшено
        "sub": 0.20,     # Уменьшено
        "mul": 0.70,     # Увеличено для нелинейности взаимодействий
        "sin": 0.80,     # Увеличено для периодичности
        "tanh": 0.60,    # Увеличено для нелинейности
        "exp": 0.50,     # Увеличено
        "log": 0.40,     # Увеличено
        "abs": 0.25,     # Увеличено
        "cos": 0.60,     # Увеличено для периодичности
    },
    constant_range=Range(min=-3.0, max=3.0),  # Расширен диапазон констант
    smoothing_prob=0.003,  # Увеличено
    smoothing_prior=SmoothingPrior(
        methods=["moving_average", "gaussian"],
        window_range=(3, 15),
        std_range=Range(min=0.5, max=2.0),
    ),
    normalization_choices=[None],
    output_scale_range=Range(min=-3.0, max=4.0),  # Агрессивно увеличена амплитуда
)


from modules.transformations import Range

# =============================================================================
# КАЛИБРОВКА v3: Агрессивное увеличение амплитуды и периодичности
# =============================================================================
latent_prior = LatentPrior(
    # 1. Распределение вероятностей выбора модуля
    # Максимальная доля Periodic и ETS для финансовых рядов
    type_probs={"arima": 0.08, "kernel_synth": 0.50, "tsi": 0.12, "ets": 0.30},
    l_range=(4, 8),  # Агрессивно увеличено для cross_corr
    # 2. Конфигурация ARIMA
    arima=ARIMAPrior(
        p_range=(0, 3),
        d_choices=[0],
        q_range=(0, 3),
        ar_range=Range(min=-0.8, max=0.8),
        ma_range=Range(min=-0.8, max=0.8),
        intercept_range=Range(min=-0.5, max=0.5),
        sigma_range=Range(min=0.01, max=0.5),
    ),
    # 3. Конфигурация KernelSynth (Гауссовские процессы)
    # Агрессивно увеличена дисперсия для signature_energy
    kernel_synth=KernelSynthPrior(
        kernel_type_probs={"RBF": 0.10, "Periodic": 0.70, "RQ": 0.10, "Linear": 0.10},
        lengthscale_range=Range(min=0.1, max=0.5),
        variance_range=Range(min=4.0, max=15.0),  # Агрессивно увеличена
        period_range=Range(min=2.0, max=60.0),  # Максимально расширен диапазон
        alpha_range=Range(min=0.5, max=5.0),
        # Параметры тренда m(t) = a*t + b + c*exp(d*t)
        mean_a_range=Range(min=-1.0, max=1.0),
        mean_b_range=Range(min=-0.5, max=0.5),
        mean_c_range=Range(min=-0.2, max=0.2),
        mean_d_range=Range(min=-1.0, max=1.0),
    ),
    # 4. Конфигурация TSI (Внутренние моды/Синусоиды)
    tsi=TSIPrior(
        n_modes_range=(4, 8),  # Увеличено число мод
        frequency_range=Range(min=0.5, max=8.0),
        amplitude_range=Range(min=0.5, max=3.0),  # Агрессивно увеличена
        phase_range=Range(min=0.0, max=3.14),
        decay_range=Range(min=0.0, max=1.0),
    ),
    # 5. Конфигурация ETS (Экспоненциальное сглаживание)
    # Максимальная доля сезонности
    ets=ETSPrior(
        model_type_probs={"ANN": 0.10, "AAN": 0.20, "AAA": 0.70},  # Максимум сезонности
        alpha_range=Range(min=0.1, max=0.9),
        beta_range=Range(min=0.0, max=0.5),
        gamma_range=Range(min=0.0, max=0.5),
        seasonality_period_range=(4, 24),
        initial_level_range=Range(min=-3.0, max=3.0),  # Агрессивно увеличена
        initial_trend_range=Range(min=-0.3, max=0.3),
        initial_seasonal_range=Range(min=-1.5, max=1.5),  # Увеличена сезонность
    ),
)


noise_prior = NoisePrior(
    additive_scale_range=Range(min=0.01, max=0.1),  # было 0.001–0.005
    additive_df_range=Range(min=5.0, max=20.0),  # тяжелее хвосты
    multiplicative_scale_range=Range(min=0.02, max=0.1),
    multiplicative_shape_range=Range(min=2.0, max=5.0),
)


config_example = GeneratorConfig(
    batch_size=4,
    seq_len_range=(50, 100),
    dim_range=(1, 3),
    latent=latent_prior,
    transform=transform_prior,
    noise=noise_prior,
)
