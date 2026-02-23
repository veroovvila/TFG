import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

def evaluar_clasificador_final(X, y_real, ranking, k, random_state=42):
    """
    Entrena un clasificador final usando las top-k características
    y devuelve el AUC en test.
    """

    # Seleccionar top-k features
    features_seleccionadas = ranking[:k]

    X_train, X_test, y_train, y_test = train_test_split(
        X[:, features_seleccionadas],
        y_real,
        test_size=0.3,
        random_state=random_state,
        stratify=y_real
    )

    modelo = Pipeline([
        ("escalado", StandardScaler()),
        ("clasificador", LogisticRegression(max_iter=1000))
    ])

    modelo.fit(X_train, y_train)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilidades)

    return auc


def calcular_mi_naive(X, S, random_state=42):
    """
    Calcula MI usando S como si fuera la etiqueta real.
    """
    mi_scores = mutual_info_classif(
        X, S,
        random_state=random_state
    )

    ranking = np.argsort(mi_scores)[::-1]

    return mi_scores, ranking


def calcular_mi_real(X, y_real, random_state=42):
    """
    Calcula MI usando las etiquetas reales.
    """
    mi_scores = mutual_info_classif(
        X, y_real,
        random_state=random_state
    )

    ranking = np.argsort(mi_scores)[::-1]

    return mi_scores, ranking


def calcular_varianza(X):
    """
    Ranking basado en varianza (método no supervisado).
    """
    varianzas = np.var(X, axis=0)
    ranking = np.argsort(varianzas)[::-1]

    return varianzas, ranking


def comparar_metodos(
    X,
    y_real,
    S,
    ranking_pu,
    ranking_naive,
    ranking_real,
    ranking_varianza,
    k=10
):
    """
    Compara distintos métodos de selección usando AUC.
    """

    resultados = {}

    resultados["PU_corregido"] = evaluar_clasificador_final(
        X, y_real, ranking_pu, k
    )

    resultados["MI_naive"] = evaluar_clasificador_final(
        X, y_real, ranking_naive, k
    )

    resultados["MI_real"] = evaluar_clasificador_final(
        X, y_real, ranking_real, k
    )

    resultados["Varianza"] = evaluar_clasificador_final(
        X, y_real, ranking_varianza, k
    )

    return resultados