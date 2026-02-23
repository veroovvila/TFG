import os
import pandas as pd
import mlflow
import mlflow.sklearn

from src.data_utiles import generar_etiquetas_pu
from src.config import *
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, obtener_scores, estimar_probabilidad_real
from src.mi_utiles import calcular_mi_ranking, guardar_ranking
from src.evaluacion import comparar_metodos, calcular_mi_naive, calcular_mi_real, calcular_varianza
from sklearn.datasets import load_breast_cancer

def main():
    # Configuración de MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME):

        # Cargar dataset
        X, y = load_breast_cancer(return_X_y=True)
        print("DATASET CARGADO.\nNúmero de muestras totales:", X.shape[0], "\nNúmero de características:", X.shape[1], "\nNúmero de positivos:", sum(y), "\nNúmero de negativos:", len(y) - sum(y))
        feature_names = load_breast_cancer().feature_names
        df = pd.DataFrame(X, columns=feature_names)
        print("Primeras 5 filas del dataset:\n")
        print(df.head())
        print("-" * 150)

        mlflow.log_param("dataset", "breast_cancer")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("alpha_true", ALPHA_TRUE)
        mlflow.log_param("top_k", TOP_K)

        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_positive_real", int(sum(y)))
        mlflow.log_param("n_negative_real", int(len(y) - sum(y)))

        # Generar escenario PU
        S = generar_etiquetas_pu(y, ALPHA_TRUE, random_state=RANDOM_STATE)
        print("Etiquetas PU GENERADAS. \nNúmero de positivos etiquetados (S=1):", sum(S), "\nNúmero de unlabeled (S=0):", len(S) - sum(S))

        mlflow.log_metric("n_labeled_positive", int(sum(S)))
        mlflow.log_metric("n_unlabeled", int(len(S) - sum(S)))

        # Entrenar modelo PU
        modelo_pu = entrenar_clasificador_pu(X, S, random_state=42)

        mlflow.sklearn.log_model(modelo_pu, "pu_model")

        # Obtener scores P(S=1 | x)
        scores = obtener_scores(modelo_pu, X)

        # Estimar alpha
        alpha_hat = estimar_alpha(scores, S)
        print(f"Alpha estimado: {alpha_hat:.4f}")

        mlflow.log_metric("alpha_estimated", float(alpha_hat))

        # Estimar P(Y=1 | x) para el conjunto de entrenamiento
        p_train = estimar_probabilidad_real(scores, alpha_hat)

        # Calcular MI y ranking de variables
        mi_scores, ranking = calcular_mi_ranking(
            X, p_train,
            metodo="regresion",   
            random_state=RANDOM_STATE
        )

        mlflow.log_param("mi_method", "regresion")

        # Guardar top features como artefacto
        # Ranking PU corregido
        guardar_ranking("PU_corregido", ranking, feature_names, TOP_K, mi_scores)

        # Ranking MI naive
        mi_naive_scores, ranking_naive = calcular_mi_naive(X, S)
        guardar_ranking("MI_naive", ranking_naive, feature_names, TOP_K, mi_naive_scores)

        # Ranking MI real
        mi_real_scores, ranking_real = calcular_mi_real(X, y)
        guardar_ranking("MI_real", ranking_real, feature_names, TOP_K, mi_real_scores)

        # Ranking Varianza
        var_scores, ranking_varianza = calcular_varianza(X)
        guardar_ranking("Varianza", ranking_varianza, feature_names, TOP_K, var_scores)


        resultados = comparar_metodos(
            X,
            y,
            S,
            ranking,
            ranking_naive,
            ranking_real,
            ranking_varianza,
            k=TOP_K
        )

        # Loggear AUCs
        for metodo, auc in resultados.items():
            mlflow.log_metric(f"AUC_{metodo}", float(auc))

        print("\nResultados AUC (Top", TOP_K, "features):\n")
        for metodo, auc in resultados.items():
            print(f"{metodo}: {auc:.4f}")

        print("\nRun guardado en MLflow")


if __name__ == "__main__":
    main()