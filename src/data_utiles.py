import numpy as np

def añadir_ruido_gaussiano(X, noise_level, random_state=None):
    """
    Añade ruido gaussiano a las features, proporcional a la std de cada columna.
    Esto simula un escenario más exigente manteniendo la escala relativa.

    Parámetros
    ----------
    X : np.ndarray, shape (n_muestras, n_features)
        Matriz de features original.
    noise_level : float
        Fracción de la std de cada feature que se usa como desviación del ruido.
        0 → sin ruido; 0.5 → 50 % de la std; 1.0 → 100 % de la std.
    random_state : int o None
        Semilla para reproducibilidad.

    Devuelve
    -------
    X_noisy : np.ndarray
        Matriz con ruido añadido (no modifica X original).
    """
    if noise_level == 0:
        return X.copy()
    rng = np.random.default_rng(random_state)
    std_por_feature = np.std(X, axis=0)
    ruido = rng.normal(0, noise_level * std_por_feature, size=X.shape)
    return X + ruido


def generar_etiquetas_pu(y, alpha, random_state=None):
    """
    Genera etiquetas Positive–Unlabeled (PU) a partir de etiquetas reales
    bajo el supuesto SCAR (Selected Completely At Random).

    Parámetros
    ----------
    y : array-like, shape (n_muestras,)
        Etiquetas reales del dataset (0 = negativo, 1 = positivo).
    alpha : float
        Probabilidad de que un positivo real esté etiquetado:
        alpha = P(S=1 | Y=1).
    random_state : int o None
        Semilla aleatoria para reproducibilidad.

    Devuelve
    -------
    S : np.ndarray, shape (n_muestras,)
        Etiquetas PU:
        - S = 1 → positivo etiquetado (labeled positive)
        - S = 0 → unlabeled
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    # Inicializar todas las muestras como unlabeled
    S = np.zeros_like(y, dtype=int)

    # Índices de ejemplos positivos reales
    indices_positivos = np.where(y == 1)[0]
    n_positivos = len(indices_positivos)

    # Número de positivos que se mantendrán etiquetados
    n_etiquetados = int(np.round(alpha * n_positivos))

    # Selección aleatoria de positivos etiquetados
    indices_etiquetados = rng.choice(
        indices_positivos,
        size=n_etiquetados,
        replace=False
    )

    # Asignar S=1 a los positivos etiquetados
    S[indices_etiquetados] = 1

    return S
