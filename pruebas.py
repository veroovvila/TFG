from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()

# Cargar dataset
X, y = load_breast_cancer(return_X_y=True)
print("DATASET CARGADO.\nNúmero de muestras:", X.shape[0], "\nNúmero de características:", X.shape[1])

df = pd.DataFrame(X, columns=load_breast_cancer().feature_names)
print("Primeras 5 filas del dataset:\n")
print(df.head())