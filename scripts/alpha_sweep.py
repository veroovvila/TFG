import pandas as pd
import numpy as np
import mlflow
from sklearn.datasets import load_breast_cancer

from src.data_utiles import generar_etiquetas_pu
from src.pu_model import entrenar_clasificador_pu, obtener_scores, estimar_alpha, estimar_probabilidad_real
from src.evaluacion import calcular_mi_naive
from src.mi_utiles import calcular_mi_ranking
from src.config import TOP_K, SWEEP_ALPHAS, SWEEP_SEEDS


def topk_overlap(ranking_a, ranking_b, k):
    set_a = set(ranking_a[:k])
    set_b = set(ranking_b[:k])
    return len(set_a & set_b)


def run_sweep(alphas=None, seeds=None):
    """Ejecuta sweep de alphas variando el alpha verdadero."""
    if alphas is None:
        alphas = SWEEP_ALPHAS
    if seeds is None:
        seeds = SWEEP_SEEDS
        
    X, y = load_breast_cancer(return_X_y=True)
    rows = []

    for alpha in alphas:
        for seed in seeds:
            S = generar_etiquetas_pu(y, alpha, random_state=seed)
            modelo = entrenar_clasificador_pu(X, S, random_state=seed)
            scores = obtener_scores(modelo, X)
            
            alpha_hat = estimar_alpha(scores, S)
            mi_naive_scores, ranking_naive = calcular_mi_naive(X, S)
            
            try:
                p_train = estimar_probabilidad_real(scores, alpha_hat)
                mi_pu_scores, ranking_pu = calcular_mi_ranking(X, p_train, metodo="regresion", random_state=seed)
            except:
                ranking_pu = np.array([])
                
            mi_real_scores, ranking_real = calcular_mi_ranking(X, y, metodo="regresion", random_state=seed)
            
            overlap_naive = topk_overlap(ranking_naive, ranking_real, TOP_K)
            overlap_pu = topk_overlap(ranking_pu, ranking_real, TOP_K) if ranking_pu.size else np.nan

            rows.append({
                "alpha_true": alpha,
                "seed": seed,
                "alpha_hat": float(alpha_hat),
                "overlap_naive": int(overlap_naive),
                "overlap_pu": float(overlap_pu),
            })

    df = pd.DataFrame(rows)
    summary = df.groupby('alpha_true').agg(
        alpha_hat_mean=('alpha_hat', 'mean'),
        alpha_hat_std=('alpha_hat', 'std'),
        overlap_naive_mean=('overlap_naive', 'mean'),
        overlap_naive_std=('overlap_naive', 'std'),
        overlap_pu_mean=('overlap_pu', 'mean'),
        overlap_pu_std=('overlap_pu', 'std'),
    ).reset_index()

    df.to_csv('alpha_sweep_runs.csv', index=False)
    summary.to_csv('alpha_sweep_summary.csv', index=False)

    # Log en MLflow
    mlflow.log_param('sweep_alphas', str(alphas))
    mlflow.log_param('sweep_seeds', str(seeds))
    mlflow.log_param('top_k', int(TOP_K))
    mlflow.log_param('n_runs', int(len(rows)))

    mlflow.log_artifact('alpha_sweep_runs.csv')
    mlflow.log_artifact('alpha_sweep_summary.csv')

    for _, row in summary.iterrows():
        a = row['alpha_true']
        mlflow.log_metric(f'alpha_{a}_hat_mean', float(row['alpha_hat_mean']))
        mlflow.log_metric(f'alpha_{a}_hat_std', float(row['alpha_hat_std']))
        mlflow.log_metric(f'alpha_{a}_overlap_naive_mean', float(row['overlap_naive_mean']))
        mlflow.log_metric(f'alpha_{a}_overlap_pu_mean', float(row['overlap_pu_mean']))

    print('\nAlpha sweep - runs saved to alpha_sweep_runs.csv')
    print('Summary saved to alpha_sweep_summary.csv\n')
    print(summary.to_string(index=False))

    return df, summary
