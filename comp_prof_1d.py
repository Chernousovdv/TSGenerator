import pandas as pd
from calibration.compare_profiles import compare_profiles
from configs.stat_lists import long_stats

compare_profiles(
    df_synthetic=pd.read_csv(
        "/home/danila/TokDiT/results/generator/run_78/profile_data.csv"
    ).query("level == 'observed'"),
    df_real=pd.read_csv(
        "/home/danila/TokDiT/results/monash_profile/monash_all.csv"
    ),
    statistics=long_stats,
    save_path="results/comparison.png",
)
