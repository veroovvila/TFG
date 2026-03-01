import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend sin ventana
import matplotlib.pyplot as plt
import mlflow
from sklearn.datasets import load_breast_cancer

from src.data_utiles import generar_etiquetas_pu
from src.config import *
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, obtener_scores, estimar_probabilidad_real
from src.mi_utiles import calcular_mi_ranking, guardar_ranking
from src.evaluacion import comparar_metodos, calcular_mi_naive, calcular_mi_real, calcular_varianza


def _topk_overlap(ranking_a, ranking_b, k):
    return len(set(ranking_a[:k]) & set(ranking_b[:k]))


def main():
    """Ejecuta experimento según el modo configurado (single o sweep)."""
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer().feature_names

    # En modo single se itera una sola vez; en sweep se recorre la rejilla completa
    alphas = SWEEP_ALPHAS if RUN_MODE == 'sweep' else [ALPHA_TRUE]
    seeds  = SWEEP_SEEDS  if RUN_MODE == 'sweep' else [RANDOM_STATE]

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("dataset", "breast_cancer")
        mlflow.log_param("run_mode", RUN_MODE)
        mlflow.log_param("alpha_true", str(alphas) if RUN_MODE == 'sweep' else ALPHA_TRUE)
        mlflow.log_param("random_state", str(seeds)  if RUN_MODE == 'sweep' else RANDOM_STATE)

        rows = []  # acumula resultados por iteración (sweep)

        for alpha in alphas:
            for seed in seeds:
                # Generar escenario PU
                S = generar_etiquetas_pu(y, alpha, random_state=seed)
                if RUN_MODE == 'single':
                    mlflow.log_metric("n_labeled_positive", int(sum(S)))
                    mlflow.log_metric("n_unlabeled", int(len(S) - sum(S)))

                # Entrenar modelo PU
                modelo = entrenar_clasificador_pu(X, S, random_state=seed)
                if RUN_MODE == 'single':
                    mlflow.sklearn.log_model(modelo, "pu_model")

                # Estimar alpha
                scores = obtener_scores(modelo, X)
                alpha_hat = estimar_alpha(scores, S)
                if RUN_MODE == 'single':
                    mlflow.log_metric("alpha_estimated", float(alpha_hat))

                # Rankings
                try:
                    p_train = estimar_probabilidad_real(scores, alpha_hat)
                    mi_scores, ranking = calcular_mi_ranking(X, p_train, metodo="regresion", random_state=seed)
                except Exception:
                    ranking = np.array([])

                mi_naive_scores, ranking_naive = calcular_mi_naive(X, S)
                mi_real_scores,  ranking_real  = calcular_mi_real(X, y)
                var_scores,      ranking_var   = calcular_varianza(X)

                if RUN_MODE == 'single':
                    guardar_ranking("PU_corregido", ranking,       feature_names, TOP_K, mi_scores)
                    guardar_ranking("MI_naive",     ranking_naive, feature_names, TOP_K, mi_naive_scores)
                    guardar_ranking("MI_real",      ranking_real,  feature_names, TOP_K, mi_real_scores)
                    guardar_ranking("Varianza",     ranking_var,   feature_names, TOP_K, var_scores)

                    # Comparar métodos
                    resultados = comparar_metodos(
                        X, y, S, ranking, ranking_naive, ranking_real, ranking_var, k=TOP_K
                    )
                    for metodo, auc in resultados.items():
                        mlflow.log_metric(f"AUC_{metodo}", float(auc))

                    print("\nResultados AUC (Top", TOP_K, "features):")
                    for metodo, auc in resultados.items():
                        print(f"  {metodo}: {auc:.4f}")

                else:  # sweep: acumular métricas de overlap y AUC
                    overlap_naive = _topk_overlap(ranking_naive, ranking_real, TOP_K)
                    overlap_pu    = _topk_overlap(ranking, ranking_real, TOP_K) if ranking.size else np.nan

                    # Comparar métodos para obtener AUC
                    aucs = comparar_metodos(
                        X, y, S, ranking if ranking.size else ranking_real,
                        ranking_naive, ranking_real, ranking_var, k=TOP_K
                    )

                    rows.append({
                        "alpha_true":         alpha,
                        "seed":               seed,
                        "alpha_hat":          float(alpha_hat),
                        "overlap_naive":      int(overlap_naive),
                        "overlap_pu":         float(overlap_pu),
                        "auc_PU_corregido":   float(aucs["PU_corregido"]),
                        "auc_MI_naive":       float(aucs["MI_naive"]),
                        "auc_MI_real":        float(aucs["MI_real"]),
                        "auc_Varianza":       float(aucs["Varianza"]),
                    })

        # Logging específico del modo sweep 
        if RUN_MODE == 'sweep':
            df = pd.DataFrame(rows)
            summary = df.groupby('alpha_true').agg(
                alpha_hat_mean=    ('alpha_hat',     'mean'),
                alpha_hat_std=     ('alpha_hat',     'std'),
                overlap_naive_mean=('overlap_naive', 'mean'),
                overlap_naive_std= ('overlap_naive', 'std'),
                overlap_pu_mean=   ('overlap_pu',    'mean'),
                overlap_pu_std=    ('overlap_pu',    'std'),
            ).reset_index()

            df.to_csv('alpha_sweep_runs.csv',    index=False)
            summary.to_csv('alpha_sweep_summary.csv', index=False)

            mlflow.log_param('sweep_alphas', str(alphas))
            mlflow.log_param('sweep_seeds',  str(seeds))
            mlflow.log_param('top_k',        int(TOP_K))
            mlflow.log_param('n_runs',       int(len(rows)))

            mlflow.log_artifact('alpha_sweep_runs.csv')
            mlflow.log_artifact('alpha_sweep_summary.csv')

            for _, row in summary.iterrows():
                a = row['alpha_true']
                mlflow.log_metric(f'alpha_{a}_hat_mean',           float(row['alpha_hat_mean']))
                mlflow.log_metric(f'alpha_{a}_hat_std',            float(row['alpha_hat_std']))
                mlflow.log_metric(f'alpha_{a}_overlap_naive_mean', float(row['overlap_naive_mean']))
                mlflow.log_metric(f'alpha_{a}_overlap_pu_mean',    float(row['overlap_pu_mean']))

            # Gráfico AUC vs alpha por método
            metodos = ['PU_corregido', 'MI_naive', 'MI_real', 'Varianza']
            auc_summary = df.groupby('alpha_true')[
                [f'auc_{m}' for m in metodos]
            ].agg(['mean', 'std'])

            fig, ax = plt.subplots(figsize=(8, 5))
            for metodo in metodos:
                means = auc_summary[(f'auc_{metodo}', 'mean')].values
                stds  = auc_summary[(f'auc_{metodo}', 'std')].values
                x     = auc_summary.index.values
                ax.plot(x, means, marker='o', label=metodo)
                ax.fill_between(x, means - stds, means + stds, alpha=0.15)

            ax.set_xlabel('Alpha verdadero')
            ax.set_ylabel(f'AUC (top-{TOP_K} features)')
            ax.set_title('AUC vs Alpha por método de selección')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()

            plot_path = 'auc_vs_alpha.png'
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(plot_path)

            print('\nAlpha sweep - runs saved to alpha_sweep_runs.csv')
            print('Summary saved to alpha_sweep_summary.csv\n')
            print(summary.to_string(index=False))

        print("\nRun guardado en MLflow")


if __name__ == "__main__":
    main()