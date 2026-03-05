import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend sin ventana
import matplotlib.pyplot as plt
import mlflow
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from src.data_utiles import generar_etiquetas_pu, añadir_ruido_gaussiano
from src.config import *
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, obtener_scores, estimar_probabilidad_real
from src.mi_utiles import calcular_mi_ranking, guardar_ranking
from src.evaluacion import comparar_metodos, calcular_mi_naive, calcular_mi_real, calcular_varianza, spearman_rankings


def _topk_overlap(ranking_a, ranking_b, k):
    return len(set(ranking_a[:k]) & set(ranking_b[:k]))


def _ranking_instability(rankings_list):
    """Devuelve la inestabilidad media del ranking entre semillas:
    std promedio de la posición de cada feature a lo largo de las semillas."""
    n = len(rankings_list[0])
    pos_matrix = np.zeros((len(rankings_list), n))
    for i, r in enumerate(rankings_list):
        pos_matrix[i, r] = np.arange(n)
    return float(np.mean(np.std(pos_matrix, axis=0)))


def main():
    """Ejecuta experimento según el modo configurado (single o sweep)."""
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer().feature_names

    # En modo single se itera una sola vez; en sweep se recorre la rejilla completa
    alphas = SWEEP_ALPHAS if RUN_MODE == 'sweep' else [ALPHA_TRUE]
    seeds  = SWEEP_SEEDS  if RUN_MODE == 'sweep' else [RANDOM_STATE]

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        # Estructura del dataset
        n_samples, n_features = X.shape
        n_positives = int(y.sum())
        n_negatives = int((y == 0).sum())
        class_balance = round(n_positives / n_samples, 4) # proporción de positivos (que tan minoritaria es la clase positiva)

        mlflow.log_param("dataset",            "breast_cancer")
        mlflow.log_param("n_samples",          n_samples)
        mlflow.log_param("n_features",         n_features)
        mlflow.log_param("n_positives",        n_positives)
        mlflow.log_param("n_negatives",        n_negatives)
        mlflow.log_param("class_balance",      class_balance)

        # Estadísticas descriptivas de features (antes de ruido) 
        stats = pd.DataFrame({
            "feature":  feature_names,
            "mean":     np.mean(X, axis=0),
            "std":      np.std(X, axis=0),
            "min":      np.min(X, axis=0),
            "max":      np.max(X, axis=0),
            "median":   np.median(X, axis=0),
        })
        stats_path = "dataset_feature_stats.csv"
        stats.to_csv(stats_path, index=False)
        mlflow.log_artifact(stats_path)

        # Configuración del experimento 
        mlflow.log_param("run_mode",    RUN_MODE)
        mlflow.log_param("noise_level", NOISE_LEVEL)
        mlflow.log_param("top_k",       TOP_K)
        if RUN_MODE != 'sweep':
            mlflow.log_param("alpha_true", ALPHA_TRUE)
            mlflow.log_param("random_state", RANDOM_STATE)

        rows = []  # acumula resultados por iteración (sweep)
        rankings_by_alpha = {}  # {alpha: {'pu': [], 'naive': [], 'real': [], 'var': []}}

        for alpha in alphas:
            for seed in seeds:
                # Añadir ruido gaussiano a X para este seed (noise_level=0 → sin cambios)
                X_noisy = añadir_ruido_gaussiano(X, NOISE_LEVEL, random_state=seed)

                # Generar escenario PU
                S = generar_etiquetas_pu(y, alpha, random_state=seed)
                if RUN_MODE == 'single':
                    mlflow.log_metric("n_labeled_positive", int(sum(S)))
                    mlflow.log_metric("n_unlabeled", int(len(S) - sum(S)))

                # Entrenar modelo PU
                modelo = entrenar_clasificador_pu(X_noisy, S, random_state=seed)
                if RUN_MODE == 'single':
                    mlflow.sklearn.log_model(modelo, "pu_model")

                # Estimar alpha (sobre todo X_noisy; el modelo PU usa S, no y)
                scores = obtener_scores(modelo, X_noisy)
                alpha_hat = estimar_alpha(scores, S)
                if RUN_MODE == 'single':
                    mlflow.log_metric("alpha_estimated", float(alpha_hat))

                # Probabilidad real estimada sobre todo X_noisy; split train/test en un solo paso
                # para garantizar alineación. Rankings se calcularán SOLO sobre train.
                p_y = estimar_probabilidad_real(scores, alpha_hat)
                X_train, X_test, y_train, y_test, S_train, S_test, p_y_train, _ = \
                    train_test_split(X_noisy, y, S, p_y, test_size=0.3, random_state=seed, stratify=y)
                mi_scores, ranking = calcular_mi_ranking(
                    X_train, p_y_train, metodo="regresion", random_state=seed
                )

                mi_naive_scores, ranking_naive = calcular_mi_naive(X_train, S_train)
                mi_real_scores,  ranking_real  = calcular_mi_real(X_train, y_train)
                var_scores,      ranking_var   = calcular_varianza(X_train)

                if RUN_MODE == 'single':
                    guardar_ranking("PU_corregido", ranking,       feature_names, TOP_K, mi_scores)
                    guardar_ranking("MI_naive",     ranking_naive, feature_names, TOP_K, mi_naive_scores)
                    guardar_ranking("MI_real",      ranking_real,  feature_names, TOP_K, mi_real_scores)
                    guardar_ranking("Varianza",     ranking_var,   feature_names, TOP_K, var_scores)

                    # Comparar métodos: train para selección y entrenamiento, test para AUC
                    resultados = comparar_metodos(
                        X_train, X_test, y_train, y_test,
                        ranking,
                        ranking_naive, ranking_real, ranking_var, k=TOP_K
                    )
                    for metodo, auc in resultados.items():
                        mlflow.log_metric(f"AUC_{metodo}", float(auc))

                    print("\nResultados AUC (Top", TOP_K, "features):")
                    for metodo, auc in resultados.items():
                        print(f"  {metodo}: {auc:.4f}")

                else:  # sweep: acumular métricas de overlap y AUC
                    overlap_naive = _topk_overlap(ranking_naive, ranking_real, TOP_K)
                    overlap_pu    = _topk_overlap(ranking, ranking_real, TOP_K)

                    spearman_naive = spearman_rankings(ranking_naive, ranking_real)
                    spearman_pu    = spearman_rankings(ranking,       ranking_real)
                    spearman_var   = spearman_rankings(ranking_var,   ranking_real)
                    spearman_real  = 1.0  # MI_real comparado consigo mismo, siempre 1

                    # Acumular rankings completos para medir estabilidad entre semillas
                    if alpha not in rankings_by_alpha:
                        rankings_by_alpha[alpha] = {'pu': [], 'naive': [], 'real': [], 'var': []}
                    rankings_by_alpha[alpha]['pu'].append(ranking)
                    rankings_by_alpha[alpha]['naive'].append(ranking_naive)
                    rankings_by_alpha[alpha]['real'].append(ranking_real)
                    rankings_by_alpha[alpha]['var'].append(ranking_var)

                    # Comparar métodos: train para selección y entrenamiento, test para AUC
                    aucs = comparar_metodos(
                        X_train, X_test, y_train, y_test,
                        ranking,
                        ranking_naive, ranking_real, ranking_var, k=TOP_K
                    )

                    rows.append({
                        "alpha_true":         alpha,
                        "seed":               seed,
                        "alpha_hat":          float(alpha_hat),
                        "overlap_naive":      int(overlap_naive),
                        "overlap_pu":         float(overlap_pu),
                        "spearman_naive":     float(spearman_naive),
                        "spearman_pu":        float(spearman_pu),
                        "spearman_var":       float(spearman_var),
                        "spearman_real":      float(spearman_real),
                        "auc_PU_corregido":   float(aucs["PU_corregido"]),
                        "auc_MI_naive":       float(aucs["MI_naive"]),
                        "auc_MI_real":        float(aucs["MI_real"]),
                        "auc_Varianza":       float(aucs["Varianza"]),
                    })

        # Logging específico del modo sweep 
        if RUN_MODE == 'sweep':
            df = pd.DataFrame(rows)
            summary = df.groupby('alpha_true').agg(
                alpha_hat_mean=      ('alpha_hat',      'mean'),
                alpha_hat_std=       ('alpha_hat',      'std'),
                overlap_naive_mean=  ('overlap_naive',  'mean'),
                overlap_naive_std=   ('overlap_naive',  'std'),
                overlap_pu_mean=     ('overlap_pu',     'mean'),
                overlap_pu_std=      ('overlap_pu',     'std'),
                spearman_naive_mean= ('spearman_naive', 'mean'),
                spearman_naive_std=  ('spearman_naive', 'std'),
                spearman_pu_mean=    ('spearman_pu',    'mean'),
                spearman_pu_std=     ('spearman_pu',    'std'),
                spearman_var_mean=   ('spearman_var',   'mean'),
                spearman_var_std=    ('spearman_var',   'std'),
                spearman_real_mean=  ('spearman_real',  'mean'),
                spearman_real_std=   ('spearman_real',  'std'),
            ).reset_index()

            # Estabilidad del ranking por alpha (std promedio de posición entre semillas)
            stability_rows = []
            for alpha_val, mrankings in sorted(rankings_by_alpha.items()):
                stability_rows.append({
                    'alpha_true':      alpha_val,
                    'stability_pu':    _ranking_instability(mrankings['pu']),
                    'stability_naive': _ranking_instability(mrankings['naive']),
                    'stability_real':  _ranking_instability(mrankings['real']),
                    'stability_var':   _ranking_instability(mrankings['var']),
                })
            stability_df = pd.DataFrame(stability_rows)
            summary = summary.merge(stability_df, on='alpha_true')

            df.to_csv('alpha_sweep_runs.csv',    index=False)
            summary.to_csv('alpha_sweep_summary.csv', index=False)

            mlflow.log_param('sweep_alphas', str(alphas))
            mlflow.log_param('sweep_seeds',  str(seeds))
            mlflow.log_param('n_runs',       int(len(rows)))

            mlflow.log_artifact('alpha_sweep_runs.csv')
            mlflow.log_artifact('alpha_sweep_summary.csv')

            for _, row in summary.iterrows():
                a = row['alpha_true']
                mlflow.log_metric(f'alpha_{a}_hat_mean',             float(row['alpha_hat_mean']))
                mlflow.log_metric(f'alpha_{a}_hat_std',              float(row['alpha_hat_std']))
                mlflow.log_metric(f'alpha_{a}_overlap_naive_mean',   float(row['overlap_naive_mean']))
                mlflow.log_metric(f'alpha_{a}_overlap_pu_mean',      float(row['overlap_pu_mean']))
                mlflow.log_metric(f'alpha_{a}_spearman_naive_mean',  float(row['spearman_naive_mean']))
                mlflow.log_metric(f'alpha_{a}_spearman_pu_mean',     float(row['spearman_pu_mean']))
                mlflow.log_metric(f'alpha_{a}_stability_pu',         float(row['stability_pu']))
                mlflow.log_metric(f'alpha_{a}_stability_naive',      float(row['stability_naive']))

            # --- Gráfico 1: AUC vs alpha por método ---
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
            fig.savefig('auc_vs_alpha.png', dpi=150)
            plt.close(fig)
            mlflow.log_artifact('auc_vs_alpha.png')

            # --- Gráfico 2: Correlación de Spearman vs alpha ---
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            for label, col_mean, col_std in [
                ('MI_real (ref.)', 'spearman_real_mean',  'spearman_real_std'),
                ('PU_corregido',   'spearman_pu_mean',    'spearman_pu_std'),
                ('MI_naive',       'spearman_naive_mean', 'spearman_naive_std'),
                ('Varianza',       'spearman_var_mean',   'spearman_var_std'),
            ]:
                means = summary[col_mean].values
                stds  = summary[col_std].values
                x     = summary['alpha_true'].values
                ax2.plot(x, means, marker='o', label=label)
                ax2.fill_between(x, means - stds, means + stds, alpha=0.15)

            ax2.set_xlabel('Alpha verdadero')
            ax2.set_ylabel('Correlación de Spearman con MI_real')
            ax2.set_title('Correlación de Spearman vs Alpha')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            fig2.tight_layout()
            fig2.savefig('spearman_vs_alpha.png', dpi=150)
            plt.close(fig2)
            mlflow.log_artifact('spearman_vs_alpha.png')

            # --- Gráfico 3: Estabilidad del ranking vs alpha ---
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            for label, col in [
                ('PU_corregido', 'stability_pu'),
                ('MI_naive',     'stability_naive'),
                ('MI_real',      'stability_real'),
                ('Varianza',     'stability_var'),
            ]:
                ax3.plot(stability_df['alpha_true'].values,
                         stability_df[col].values,
                         marker='o', label=label)

            ax3.set_xlabel('Alpha verdadero')
            ax3.set_ylabel('Inestabilidad (std promedio de posición entre semillas)')
            ax3.set_title('Estabilidad del ranking vs Alpha')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.5)
            fig3.tight_layout()
            fig3.savefig('stability_vs_alpha.png', dpi=150)
            plt.close(fig3)
            mlflow.log_artifact('stability_vs_alpha.png')

            print('\nAlpha sweep - runs saved to alpha_sweep_runs.csv')
            print('Summary saved to alpha_sweep_summary.csv\n')
            print(summary.to_string(index=False))

        print("\nRun guardado en MLflow")


if __name__ == "__main__":
    main()