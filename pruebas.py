import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_files

GASES = {1: 'Etanol', 2: 'Etileno', 3: 'Amoniaco', 4: 'Acetaldehído', 5: 'Acetona', 6: 'Tolueno'}
_cache = os.path.join('data', 'gas_sensor_drift')
feature_names = [f'sensor{(i // 8) + 1}_feat{(i % 8) + 1}' for i in range(128)]

# ── Cargar cada batch por separado para poder inspeccionarlos ────────────────
batches = []
for i in range(1, 11):
    parts = load_svmlight_files([os.path.join(_cache, f'batch{i}.dat')], n_features=128)
    X_b = parts[0].toarray()
    y_b = parts[1].astype(int)
    df_b = pd.DataFrame(X_b, columns=feature_names)
    df_b.insert(0, 'batch', i)
    df_b.insert(1, 'clase', y_b)
    df_b.insert(2, 'gas', pd.Series(y_b).map(GASES).values)
    batches.append(df_b)

df_full = pd.concat(batches, ignore_index=True)

# ── Resumen global ───────────────────────────────────────────────────────────
print("=" * 60)
print("GAS SENSOR ARRAY DRIFT DATASET")
print("=" * 60)
print(f"Total muestras : {len(df_full)}")
print(f"Features        : {len(feature_names)}")
print(f"Batches         : 10  (aprox. 36 meses de medición)")
print()

# ── Muestras por batch y clase ───────────────────────────────────────────────
print("── Muestras por batch ──────────────────────────────────")
resumen = (df_full.groupby(['batch', 'gas'])
           .size()
           .unstack(fill_value=0)
           .rename_axis(None, axis=1))
resumen['TOTAL'] = resumen.sum(axis=1)
print(resumen.to_string())
print()

# ── Primeras 5 filas (solo primeras 8 features para legibilidad) ─────────────
print("── Primeras 5 filas del dataset (8 primeras features) ──")
print(df_full[['batch', 'gas'] + feature_names[:8]].head().to_string(index=False))
print()

# ── Estadísticas descriptivas por batch (media global de features) ───────────
print("── Media global de features por batch (muestra el drift) ─")
drift = df_full.groupby('batch')[feature_names].mean().mean(axis=1).rename('media_features')
print(drift.to_string())
print("  → Si los valores cambian entre batch 1 y 10, hay drift de sensores.")