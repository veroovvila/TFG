import pandas as pd
from src.data_utiles import generar_etiquetas_pu
from src.config import *
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, obtener_scores, estimar_probabilidad_real
from src.mi_utils import calcular_mi_ranking
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer

# Cargar dataset
X, y = load_breast_cancer(return_X_y=True)
print("DATASET CARGADO.\nNúmero de muestras totales:", X.shape[0], "\nNúmero de características:", X.shape[1], "\nNúmero de positivos:", sum(y), "\nNúmero de negativos:", len(y) - sum(y))

df = pd.DataFrame(X, columns=load_breast_cancer().feature_names)
print("Primeras 5 filas del dataset:\n")
print(df.head())
print("-" * 150)

# Generar escenario PU
S = generar_etiquetas_pu(y, ALPHA_TRUE, random_state=RANDOM_STATE)
print("Etiquetas PU GENERADAS. \nNúmero de positivos etiquetados (S=1):", sum(S), "\nNúmero de unlabeled (S=0):", len(S) - sum(S))

# Entrenar modelo PU
modelo_pu = entrenar_clasificador_pu(X, S, random_state=42)

# Obtener scores P(S=1 | x)
scores = obtener_scores(modelo_pu, X)

# Estimar alpha
alpha_hat = estimar_alpha(scores, S)
print(f"Alpha estimado: {alpha_hat:.4f}")

# Estimar P(Y=1 | x) para el conjunto de entrenamiento
p_train = estimar_probabilidad_real(scores, alpha_hat)

# Calcular MI y ranking de variables
mi_scores, ranking = calcular_mi_ranking(
    X, p_train,
    metodo="regresion",   
    random_state=42
)

# Listado de features ordenados por MI
print("\nRanking de características por MI (de mayor a menor):")
for idx in ranking[:10]:
    print(f"Feature {idx} (MI={mi_scores[idx]:.4f})")
