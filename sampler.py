import random
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from modules.latent import (
    ARIMAPrior,
    ARIMASpec,
    KernelSynthSpec,
    TSISpec,
    ETSSpec,
    LatentComponentSpec,
    LatentDynamics,
    LatentModulePlan,
    LatentPrior,
    KernelSynthPrior,
    TSIPrior,
    ETSPrior,
)
from modules.noise_module import NoiseModule, NoiseModulePlan, NoisePrior
from modules.observation_module import (
    ObservationModule,
)
from modules.transformations import (
    NodeDTO,
    Range,
    TransformationPrior,
    TransformationsModule,
    TransformModulePlan,
    SmoothingParams,
)


class GenerationPlan(BaseModel):
    """Полный пакет инструкций для generate_explicit."""

    batch_size: int
    seq_len: int
    dim: int
    latent: LatentModulePlan
    transform: TransformModulePlan
    noise: NoiseModulePlan

    class Config:
        arbitrary_types_allowed = True


class GeneratorConfig(BaseModel):
    """Верхнеуровневый конфиг"""

    # Параметры батча
    batch_size: int
    seq_len_range: Tuple[int, int]
    dim_range: Tuple[int, int]

    # Флаг вывода метаданных по умолчанию
    return_metadata: bool = False

    # Приоры модулей
    latent: LatentPrior
    transform: TransformationPrior
    noise: NoisePrior


class Sampler:
    def __init__(self, config: LatentPrior, device: str):
        self.config = config
        self.device = device
        self.rng = torch.Generator(device=device)

    # --- Универсальные "сэмплилки" ---

    def _uniform(self, r: Range) -> float:
        """Для непрерывных параметров (коэффициенты, веса)."""
        return random.uniform(r.min, r.max)

    def _choice(self, options: List[Any]) -> Any:
        """Для дискретных параметров (порядки d, типы сглаживания)."""
        return random.choice(options)

    def _int_range(self, r: Tuple[int, int]) -> int:
        """Для целых диапазонов (порядки p, q, количество L)."""
        return random.randint(r[0], r[1])

    def _sample_from_dict(self, prob_dict: Dict[str, float]) -> str:
        keys, probs = list(prob_dict.keys()), list(prob_dict.values())
        total = sum(probs)
        probs = (
            [p / total for p in probs] if total > 0 else [1.0 / len(probs)] * len(probs)
        )
        idx = torch.multinomial(torch.tensor(probs), 1, generator=self.rng).item()
        return keys[idx]


class TSGenerator(Sampler):
    def __init__(self, config: GeneratorConfig, device: str = "cpu"):
        super().__init__(config, device)  # Передаем конфиг и девайс в Sampler

        # Инициализация вычислительных модулей
        self.latent_module = LatentDynamics(device=device)
        self.transform_module = TransformationsModule(device=device)
        self.noise_module = NoiseModule(device=device)
        self.observation_module = ObservationModule(device=device)

    def __call__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, GenerationPlan]]:
        """ """
        return self.generate()

    def generate(
        self,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        dim: Optional[int] = None,
        seed: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, GenerationPlan]]:
        """ """
        # Сэмплирование размеров, если не заданы
        B = batch_size or self.config.batch_size
        T = seq_len or self._int_range(self.config.seq_len_range)
        D = dim or self._int_range(self.config.dim_range)

        # Создание плана
        plan = self._sample_plan(B, T, D, seed)

        return self.generate_explicit(plan, return_metadata)

    def generate_explicit(
        self, plan: GenerationPlan, return_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, GenerationPlan]]:
        """
        Выполняет цепочку модулей среднего уровня строго по плану.
        """
        latent_out = self.latent_module.execute(
            plan.batch_size, plan.seq_len, plan.latent
        )
        observed_clean = self.transform_module.execute(latent_out, plan.transform)
        noisy_data = self.noise_module.execute(observed_clean, plan.noise)
        final_output = self.observation_module.execute(
            noisy_data
        )  # Simple version (no plan for obs module)

        if return_metadata:
            return final_output, plan
        return final_output

    def _sample_component_type(self) -> str:
        types = list(self.config.latent.type_probs.keys())
        weights = list(self.config.latent.type_probs.values())
        return random.choices(types, weights=weights, k=1)[0]

    def _is_stable_ar(self, ar_params: list[float]) -> bool:
        """
        Проверяет стационарность AR(p) через корни характеристического полинома.

        Условие стационарности: все корни полинома
            1 - φ₁z - φ₂z² - ... - φₚzᵖ
        лежат строго вне единичного круга (|root| > 1).

        Для AR(1) эквивалентно |φ₁| < 1.
        Для AR(p≥2) является более строгим условием чем просто |φᵢ| < 1.
        """
        if len(ar_params) == 0:
            return True
        # numpy.roots ожидает коэффициенты от старшей степени к младшей:
        # полином φₚzᵖ + ... + φ₁z - 1 = 0  →  корни те же что и у 1 - φ₁z - ...
        coeffs = np.concatenate(
            [[-ar_params[i] for i in range(len(ar_params) - 1, -1, -1)], [1.0]]
        )
        roots = np.roots(coeffs)
        return bool(np.all(np.abs(roots) > 1.0))

    def _sample_arima_spec(self, prior: ARIMAPrior) -> ARIMASpec:
        p = self._int_range(prior.p_range)
        q = self._int_range(prior.q_range)
        d = self._choice(prior.d_choices)

        # --- Rejection sampling для AR-коэффициентов ---
        # Семплируем до тех пор пока набор коэффициентов не даёт стационарный процесс.
        # Fallback после MAX_TRIES итераций: используем AR(0) чтобы не зависнуть.
        MAX_TRIES = 100
        ar_params = []
        if p > 0:
            for attempt in range(MAX_TRIES):
                candidate = [self._uniform(prior.ar_range) for _ in range(p)]
                if self._is_stable_ar(candidate):
                    ar_params = candidate
                    break
            else:
                # За MAX_TRIES попыток стабильный набор не найден.
                # Это сигнал что ar_range слишком широк для данного p —
                # используем AR(0) и логируем предупреждение.
                import warnings

                warnings.warn(
                    f"ARIMASampler: не удалось найти стабильные AR-параметры "
                    f"за {MAX_TRIES} попыток (p={p}, ar_range={prior.ar_range}). "
                    "Используется AR(0).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                ar_params = []

        ma_params = [self._uniform(prior.ma_range) for _ in range(q)]

        # --- Масштабирование intercept ---
        # E[X] = intercept / (1 - sum(ar_params)).
        # Если знаменатель мал (процесс близок к нестационарному),
        # сырой intercept даст огромное среднее.
        # Масштабируем так чтобы E[X] оставалось в том же диапазоне что и intercept_range.
        raw_intercept = self._uniform(prior.intercept_range)
        ar_sum = sum(ar_params)
        stability_margin = 1.0 - ar_sum  # > 0 гарантировано после rejection sampling
        intercept = raw_intercept * stability_margin

        return ARIMASpec(
            ar_params=torch.tensor(ar_params),
            ma_params=torch.tensor(ma_params),
            d=d,
            intercept=intercept,
            sigma=self._uniform(prior.sigma_range),
        )

    def _sample_kernel_synth_spec(self, prior: KernelSynthPrior) -> KernelSynthSpec:
        # Сэмплируем тип ядра
        kernel_type = self._sample_from_dict(prior.kernel_type_probs)

        # Сэмплируем гиперпараметры ядра
        lengthscale = self._uniform(prior.lengthscale_range)
        variance = self._uniform(prior.variance_range)
        period = self._uniform(prior.period_range)
        alpha = self._uniform(prior.alpha_range)

        # Сэмплируем параметры функции среднего
        mean_a = self._uniform(prior.mean_a_range)
        mean_b = self._uniform(prior.mean_b_range)
        mean_c = self._uniform(prior.mean_c_range)
        mean_d = self._uniform(prior.mean_d_range)

        return KernelSynthSpec(
            kernel_type=kernel_type,
            lengthscale=lengthscale,
            variance=variance,
            period=period,
            alpha=alpha,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_c=mean_c,
            mean_d=mean_d,
        )

    def _sample_tsi_spec(self, prior: TSIPrior) -> TSISpec:
        # Сэмплируем количество мод
        n_modes = self._int_range(prior.n_modes_range)

        # Сэмплируем параметры для каждой моды
        frequencies = [self._uniform(prior.frequency_range) for _ in range(n_modes)]
        amplitudes = [self._uniform(prior.amplitude_range) for _ in range(n_modes)]
        phases = [self._uniform(prior.phase_range) for _ in range(n_modes)]
        decays = [self._uniform(prior.decay_range) for _ in range(n_modes)]

        return TSISpec(
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            decays=decays,
        )

    def _sample_ets_spec(self, prior: ETSPrior) -> ETSSpec:
        # Сэмплируем тип модели
        model_type = self._sample_from_dict(prior.model_type_probs)

        # Сэмплируем параметры сглаживания
        alpha = self._uniform(prior.alpha_range)
        beta = self._uniform(prior.beta_range)
        gamma = self._uniform(prior.gamma_range)

        # Сэмплируем период сезонности
        seasonality_period = self._int_range(prior.seasonality_period_range)

        # Сэмплируем начальные значения
        initial_level = self._uniform(prior.initial_level_range)
        initial_trend = self._uniform(prior.initial_trend_range)

        # Сэмплируем начальные сезонные компоненты
        initial_seasonal = []
        if seasonality_period > 0:
            initial_seasonal = [
                self._uniform(prior.initial_seasonal_range)
                for _ in range(seasonality_period)
            ]

        return ETSSpec(
            model_type=model_type,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            seasonality_period=seasonality_period,
            initial_level=initial_level,
            initial_trend=initial_trend,
            initial_seasonal=initial_seasonal,
        )

    def _grow_tree(self, current_depth: int, L: int) -> NodeDTO:
        prior = self.config.transform

        if current_depth >= prior.max_depth:
            kind = self._sample_from_dict({"terminal": 0.5, "constant": 0.5})
        else:
            kind = self._sample_from_dict(prior.node_kind_probs)

        smoothing = None
        if random.random() < prior.smoothing_prob:
            s_p = prior.smoothing_prior
            smoothing = SmoothingParams(
                method=self._choice(s_p.methods),
                window_size=self._int_range(s_p.window_range),
                std=self._uniform(s_p.std_range),
            )

        if kind == "terminal":
            return NodeDTO(
                kind="terminal",
                value=f"x_{random.randint(0, L-1)}",
                smoothing=smoothing,
            )
        if kind == "constant":
            return NodeDTO(
                kind="constant",
                value=self._uniform(prior.constant_range),
                smoothing=smoothing,
            )

        if kind == "op":
            op_name = self._sample_from_dict(prior.op_weights)
            num_children = 2 if op_name in ["add", "sub", "mul", "div", "pow"] else 1
            return NodeDTO(
                kind="op",
                value=op_name,
                children=[
                    self._grow_tree(current_depth + 1, L) for _ in range(num_children)
                ],
                smoothing=smoothing,
            )

    def _sample_latent_plan(self, B: int, L: int) -> LatentModulePlan:
        """Сэмплирует параметры латентных факторов для всего батча."""
        all_rows = []
        for _ in range(B):
            row_factors = []
            for _ in range(L):
                # 1. Выбираем тип
                component_type = self._sample_from_dict(self.config.latent.type_probs)

                if component_type == "arima":
                    # 2. Генерируем СРАЗУ ARIMASpec (он уже содержит поле type="arima")
                    spec = self._sample_arima_spec(self.config.latent.arima)
                elif component_type == "kernel_synth":
                    # Генерируем KernelSynthSpec
                    spec = self._sample_kernel_synth_spec(
                        self.config.latent.kernel_synth
                    )
                elif component_type == "tsi":
                    # Генерируем TSISpec
                    spec = self._sample_tsi_spec(self.config.latent.tsi)
                elif component_type == "ets":
                    # Генерируем ETSSpec
                    spec = self._sample_ets_spec(self.config.latent.ets)
                else:
                    raise NotImplementedError(f"Тип {component_type} еще не реализован")

                row_factors.append(spec)
            all_rows.append(row_factors)

        # Передаем в items (как требует твой LatentModulePlan)
        return LatentModulePlan(items=all_rows)

    def _sample_transform_plan(self, B: int, D: int, L: int) -> TransformModulePlan:
        """Генерирует символьные деревья и параметры пост-обработки."""
        prior = self.config.transform

        # 1. Генерируем матрицу деревьев [B][D]
        trees = []
        for _ in range(B):
            batch_trees = [self._grow_tree(0, L) for _ in range(D)]
            trees.append(batch_trees)

        # 2. Сэмплируем параметры пост-обработки
        # Нормализация выбирается одна на весь батч для консистентности
        norm = self._choice(prior.normalization_choices)

        # Масштабы выходных каналов (D)
        output_scales = [self._uniform(prior.output_scale_range) for _ in range(D)]

        return TransformModulePlan(
            trees=trees, normalization=norm, output_scales=output_scales
        )

    def _sample_noise_plan(self, B: int, D: int) -> NoiseModulePlan:
        """Векторизованное сэмплирование параметров шума."""
        p = self.config.noise

        # Используем torch для быстрой генерации тензоров параметров (B, D)
        return NoiseModulePlan(
            additive_scale=torch.empty(B, D, device=self.device).uniform_(
                p.additive_scale_range.min, p.additive_scale_range.max
            ),
            additive_df=torch.empty(B, D, device=self.device).uniform_(
                p.additive_df_range.min, p.additive_df_range.max
            ),
            multiplicative_scale=torch.empty(B, D, device=self.device).uniform_(
                p.multiplicative_scale_range.min, p.multiplicative_scale_range.max
            ),
            multiplicative_shape=torch.empty(B, D, device=self.device).uniform_(
                p.multiplicative_shape_range.min, p.multiplicative_shape_range.max
            ),
        )

    def _sample_plan(
        self, B: int, T: int, dim: int, seed: Optional[int]
    ) -> GenerationPlan:
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # 1. Сэмплируем количество латентных факторов L (общее для батча)
        L = self._int_range(self.config.latent.l_range)

        # 2. Формируем планы модулей
        latent_plan = self._sample_latent_plan(B, L)
        transform_plan = self._sample_transform_plan(B, dim, L)
        noise_plan = self._sample_noise_plan(B, dim)

        return GenerationPlan(
            batch_size=B,
            seq_len=T,
            dim=dim,
            latent=latent_plan,
            transform=transform_plan,
            noise=noise_plan,
        )
