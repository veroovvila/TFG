from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Cargar dataset
sonar = fetch_openml(name='sonar', version=1, as_frame=False, parser='auto')
X = sonar.data.astype(float)
le = LabelEncoder()
y = le.fit_transform(sonar.target)  # Mine=0, Rock=1 (alfabético)
# Reordenar para que Mine (objeto metálico, "positivo") sea y=1
if le.classes_[0] == 'Mine':
    y = 1 - y
# Nombres de características
feature_names = sonar.feature_names
print("DATASET CARGADO.\nNúmero de muestras:", X.shape[0], "\nNúmero de características:", X.shape[1])

df = pd.DataFrame(X, columns=feature_names)
print("Primeras 5 filas del dataset:\n")
print(df.head())