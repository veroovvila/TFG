import os
import pandas as pd
import mlflow
import mlflow.sklearn

from src.data_utiles import generar_etiquetas_pu
from src.config import *
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, obtener_scores, estimar_probabilidad_real
from src.mi_utiles import calcular_mi_ranking
from src.evaluacion import comparar_metodos, calcular_mi_naive, calcular_mi_real, calcular_varianza
from sklearn.datasets import load_breast_cancer

def main():
    # Configuración de MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME):

        # Cargar dataset
        X, y = load_breast_cancer(return_X_y=True)
        print("DATASET CARGADO.\nNúmero de muestras totales:", X.shape[0], "\nNúmero de características:", X.shape[1], "\nNúmero de positivos:", sum(y), "\nNúmero de negativos:", len(y) - sum(y))

        df = pd.DataFrame(X, columns=load_breast_cancer().feature_names)
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
            random_state=42
        )

        mlflow.log_param("mi_method", "regresion")

        # Guardar top features como artefacto
        top_features = ranking[:TOP_K]
        top_df = pd.DataFrame({
            "feature_index": top_features,
            "mi_score": [mi_scores[idx] for idx in top_features]
        })

        top_df.to_csv("top_features.csv", index=False)
        mlflow.log_artifact("top_features.csv")


        # Comparar con otros métodos
        _, ranking_pu = calcular_mi_ranking(X, p_train, metodo="regresion")
        _, ranking_naive = calcular_mi_naive(X, S)
        _, ranking_real = calcular_mi_real(X, y)
        _, ranking_varianza = calcular_varianza(X)


        resultados = comparar_metodos(
            X,
            y,
            S,
            ranking_pu,
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