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


transform_prior = TransformationPrior(
    max_depth=3,
    node_kind_probs = {"op": 0.8, "terminal": 0.2, "constant": 0.0},
    op_weights={"add": 1.0, "sub": 1.0, "mul": 0.5, "sin": 0.5, "exp": 0.0, "abs": 0.1},
    constant_range=Range(min=0.9, max=4),
    smoothing_prob=0.2,
    smoothing_prior=SmoothingPrior(
        methods=["moving_average", "gaussian"],
        window_range=(3, 15),
        std_range=Range(min=0.01, max=1.0),
    ),
    normalization_choices=["z-score"],
    output_scale_range=Range(min=1.0, max=1.0),
)


from modules.transformations import Range

latent_prior = LatentPrior(
    # 1. Распределение вероятностей выбора модуля
    type_probs={"arima": 0.25, "kernel_synth": 0.25, "tsi": 0.25, "ets": 0.25},
    #type_probs={"arima": 0.0, "kernel_synth": 1.0, "tsi": 0.0, "ets": 0.0},

    l_range=(1, 5),
    # 2. Конфигурация ARIMA
    arima=ARIMAPrior(
        p_range=(0, 3),
        d_choices=[
            0, 1
        ],
        q_range=(0, 3),
        ar_range=Range(min=-0.8, max=0.8),
        ma_range=Range(min=-0.8, max=0.8),
        intercept_range=Range(min=-0.5, max=0.5),
        sigma_range=Range(min=0.01, max=1.5),
    ),
    # 3. Конфигурация KernelSynth (Гауссовские процессы)
    kernel_synth=KernelSynthPrior(
        kernel_type_probs={"RBF": 0.4, "Periodic": 0.3, "RQ": 0.2, "Linear": 0.1},
        lengthscale_range=Range(min=0.01, max=0.1),
        variance_range=Range(min=0.1, max=2.0),
        period_range=Range(min=0.01, max=0.2),
        alpha_range=Range(min=0.1, max=5.0),
        # Параметры тренда m(t) = a*t + b + c*exp(d*t)
        mean_a_range=Range(min=-0.5, max=0.5),
        mean_b_range=Range(min=-0.1, max=0.1),
        mean_c_range=Range(min=-0.1, max=0.1),
        mean_d_range=Range(min=-1.0, max=1.0),
    ),
    # 4. Конфигурация TSI (Внутренние моды/Синусоиды)
    tsi=TSIPrior(
        n_modes_range=(1, 3),
        frequency_range=Range(min=0.5, max=20.0),  # убрать очень высокие
        amplitude_range=Range(min=0.1, max=1.0),
        phase_range=Range(min=0.0, max=3.14),
        decay_range=Range(min=0.0, max=1.0),
    ),
    # 5. Конфигурация ETS (Экспоненциальное сглаживание)
    ets=ETSPrior(
        # Вероятности для типов: Error (A/M), Trend (N/A/M), Seasonality (N/A/M)
        model_type_probs={"ANN": 0.4, "AAN": 0.3, "AAA": 0.3},
        alpha_range=Range(min=0.1, max=0.9),
        beta_range=Range(min=0.0, max=0.5),
        gamma_range=Range(min=0.0, max=0.5),
        seasonality_period_range=(
            4,
            24,
        ),  # Диапазон периодов (например, квартал или сутки)
        initial_level_range=Range(min=-1.0, max=1.0),
        initial_trend_range=Range(min=-0.1, max=0.1),
        initial_seasonal_range=Range(min=-0.5, max=0.5),
    ),
)


# noise_prior = NoisePrior(
#     additive_scale_range       = Range(min=1e-8, max=1e-8),
#     additive_df_range          = Range(min=100.0, max=100.0),  # df→∞ ≈ гауссиан
#     multiplicative_scale_range = Range(min=1e-8, max=1e-8),
#     multiplicative_shape_range = Range(min=100.0, max=100.0),  # shape→∞ ≈ константа
# )

noise_prior = NoisePrior(
    additive_scale_range       = Range(min=0.05, max=1.0),
    additive_df_range          = Range(min=100.0, max=100.0),  # df→∞ ≈ гауссиан
    multiplicative_scale_range = Range(min=0.01, max=0.02),
    multiplicative_shape_range = Range(min=100.0, max=100.0),  # shape→∞ ≈ константа
)

# noise_prior = NoisePrior(
#     additive_scale_range       = Range(min=1e-8, max=1e-8),
#     additive_df_range          = Range(min=100.0, max=100.0),  # df→∞ ≈ гауссиан
#     multiplicative_scale_range = Range(min=1e-8, max=1e-8),
#     multiplicative_shape_range = Range(min=100.0, max=100.0),  # shape→∞ ≈ константа
# )

config_example = GeneratorConfig(
    batch_size=4,
    seq_len_range=(199, 201),
    dim_range=(1, 1),
    latent=latent_prior,
    transform=transform_prior,
    noise=noise_prior,
)
