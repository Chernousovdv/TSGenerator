from calibration.analyze_latent import profile_generator
from configs.Danila_config import config_example
from configs.stat_lists import short_stats, us
from sampler import TSGenerator

generator = TSGenerator(config=config_example, device="cpu")


df = profile_generator(
    generator=generator,  # один объект вместо prior + latent_dynamics
    statistics=us,
    n_series=50,
    save_dir="results/generator",
)

