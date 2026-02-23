import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
import mlflow


def calcular_mi_ranking(X, p_y, metodo="regresion", n_bins=10, random_state=42):
    """
    Calcula la Información Mutua (MI) entre cada característica de X y la
    probabilidad corregida p_y = P(Y=1|x), y devuelve un ranking de variables.

    Parámetros
    ----------
    X : array-like, shape (n_muestras, n_features)
        Matriz de características.
    p_y : array-like, shape (n_muestras,)
        Probabilidad corregida de ser positivo, en [0,1].
    metodo : str
        - "regresion": usa mutual_info_regression (recomendado, target continuo)
        - "discretizado": discretiza X y p_y y usa mutual_info_classif
    n_bins : int
        Número de bins si metodo="discretizado".
    random_state : int
        Semilla para reproducibilidad (afecta al estimador kNN de MI).

    Devuelve
    -------
    mi_scores : np.ndarray, shape (n_features,)
        Valor de MI para cada característica.
    ranking : np.ndarray
        Índices de características ordenados de mayor a menor MI.
    """

    X = np.asarray(X)
    p_y = np.asarray(p_y)

    if metodo == "regresion":
        # MI con objetivo continuo
        mi_scores = mutual_info_regression(
            X, p_y,
            random_state=random_state
        )

    elif metodo == "discretizado":
        # Discretizar X
        discretizador_X = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile"
        )
        X_disc = discretizador_X.fit_transform(X)

        # Discretizar p_y (target) en bins y convertir a entero
        discretizador_p = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile"
        )
        p_disc = discretizador_p.fit_transform(p_y.reshape(-1, 1)).astype(int).ravel()

        mi_scores = mutual_info_classif(
            X_disc, p_disc,
            discrete_features=True,
            random_state=random_state
        )

    else:
        raise ValueError('metodo debe ser "regresion" o "discretizado".')

    # Ranking: índices ordenados por MI descendente
    ranking = np.argsort(mi_scores)[::-1]

    return mi_scores, ranking


def guardar_ranking(nombre, ranking, feature_names, top_k, scores=None):
    """
    Guarda el top-k ranking como CSV y lo loggea en MLflow.
    """
    #TODO: compretar descripción
    
    top_features = ranking[:top_k]

    if scores is not None:
        valores = [scores[idx] for idx in top_features]
    else:
        valores = [None] * len(top_features)

    df_rank = pd.DataFrame({
        "feature_index": top_features,
        "feature_name": [feature_names[idx] for idx in top_features],
        "score": valores
    })

    filename = f"top_features_{nombre}.csv"
    df_rank.to_csv(filename, index=False)
    mlflow.log_artifact(filename)

    print(f"\nTop {top_k} features - {nombre}:")
    print(df_rank)