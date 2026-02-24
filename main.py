import pandas as pd
import mlflow
from sklearn.datasets import load_breast_cancer

from src.data_utiles import generar_etiquetas_pu
from src.config import *
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, obtener_scores, estimar_probabilidad_real
from src.mi_utiles import calcular_mi_ranking, guardar_ranking
from src.evaluacion import comparar_metodos, calcular_mi_naive, calcular_mi_real, calcular_varianza


def run_single():
    """Ejecuta experimento simple con alpha fijo."""
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer().feature_names
    
    mlflow.log_param("dataset", "breast_cancer")
    mlflow.log_param("alpha_true", ALPHA_TRUE)
    mlflow.log_param("random_state", RANDOM_STATE)
    
    # Generar escenario PU
    S = generar_etiquetas_pu(y, ALPHA_TRUE, random_state=RANDOM_STATE)
    mlflow.log_metric("n_labeled_positive", int(sum(S)))
    mlflow.log_metric("n_unlabeled", int(len(S) - sum(S)))
    
    # Entrenar modelo PU
    modelo = entrenar_clasificador_pu(X, S, random_state=RANDOM_STATE)
    mlflow.sklearn.log_model(modelo, "pu_model")
    
    # Estimar alpha
    scores = obtener_scores(modelo, X)
    alpha_hat = estimar_alpha(scores, S)
    mlflow.log_metric("alpha_estimated", float(alpha_hat))
    
    # Rankings
    p_train = estimar_probabilidad_real(scores, alpha_hat)
    mi_scores, ranking = calcular_mi_ranking(X, p_train, metodo="regresion", random_state=RANDOM_STATE)
    
    guardar_ranking("PU_corregido", ranking, feature_names, TOP_K, mi_scores)
    
    mi_naive_scores, ranking_naive = calcular_mi_naive(X, S)
    guardar_ranking("MI_naive", ranking_naive, feature_names, TOP_K, mi_naive_scores)
    
    mi_real_scores, ranking_real = calcular_mi_real(X, y)
    guardar_ranking("MI_real", ranking_real, feature_names, TOP_K, mi_real_scores)
    
    var_scores, ranking_varianza = calcular_varianza(X)
    guardar_ranking("Varianza", ranking_varianza, feature_names, TOP_K, var_scores)
    
    # Comparar métodos
    resultados = comparar_metodos(X, y, S, ranking, ranking_naive, ranking_real, ranking_varianza, k=TOP_K)
    
    for metodo, auc in resultados.items():
        mlflow.log_metric(f"AUC_{metodo}", float(auc))
    
    print("\nResultados AUC (Top", TOP_K, "features):")
    for metodo, auc in resultados.items():
        print(f"  {metodo}: {auc:.4f}")


def main():
    """Ejecuta experimento según el modo configurado."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param('run_mode', RUN_MODE)
        
        if RUN_MODE == 'sweep':
            from scripts.alpha_sweep import run_sweep
            run_sweep()
        else:
            run_single()
        
        print("\nRun guardado en MLflow")


if __name__ == "__main__":
    main()