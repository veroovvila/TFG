RANDOM_STATE = 42
RUN_MODE = 'sweep'  # 'single' o 'sweep'

# Parámetros modo single
ALPHA_TRUE = 0.2

# Parámetros modo sweep
SWEEP_ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05]
SWEEP_SEEDS = [0, 1, 2, 3, 4]

# Ruido gaussiano en features (0 = sin ruido; p.ej. 0.3 = 30% de la std por feature)
NOISE_LEVEL = 0.5

# Parámetros generales
TOP_K = 10
EXPERIMENT_NAME = "breast_cancer"
RUN_NAME = "v2.1.2"