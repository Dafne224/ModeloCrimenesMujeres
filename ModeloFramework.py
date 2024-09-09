import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import IsolationForest
from scipy.stats import mstats
import xgboost as xgb
from xgboost import XGBRegressor

# 1. Cargar el conjunto de datos
data = pd.read_csv('CrimesOnWomenData.csv')

# 2. Refinamiento de outliers utilizando winsorización
def winsorize_series(series, limits):
    return mstats.winsorize(series, limits=limits)

# Aplicamos winsorización a todas las características y el objetivo
data['Rape'] = winsorize_series(data['Rape'], limits=[0.05, 0.05])
data['K&A'] = winsorize_series(data['K&A'], limits=[0.05, 0.05])
data['DD'] = winsorize_series(data['DD'], limits=[0.05, 0.05])
data['AoW'] = winsorize_series(data['AoW'], limits=[0.05, 0.05])
data['AoM'] = winsorize_series(data['AoM'], limits=[0.05, 0.05])
data['WT'] = winsorize_series(data['WT'], limits=[0.05, 0.05])
data['DV'] = winsorize_series(data['DV'], limits=[0.05, 0.05])

# 3. Preparar las características y el objetivo
X = data[['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'WT']]
y = data['DV']

# 4. Aplicar transformación logarítmica a la variable objetivo
y_log = np.log1p(y)  # np.log1p es equivalente a log(1 + y) para evitar logaritmo de 0

# 5. Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 6. Normalizar las características utilizando StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Redefinir el modelo con regularización L2 y dropout
model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),  # Aplicar dropout para evitar sobreajuste
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='linear')  # Salida para regresión
])

# 8. Compilar el modelo usando el optimizador RMSprop
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Implementar Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 9. Entrenar el modelo con Early Stopping
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# 10. Evaluar el modelo en el conjunto de prueba
y_pred_log = model.predict(X_test_scaled)

# 11. Invertir la transformación logarítmica en las predicciones y los valores reales
y_pred = np.expm1(y_pred_log).flatten()  # Aseguramos que sea un array 1D
y_test_orig = np.expm1(y_test.to_numpy()).flatten()  # Convertimos a NumPy antes de aplicar flatten()

# ** Gráfica 1: Pérdida de entrenamiento y validación **
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación con regularización L2 y dropout')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# 12. Comparación entre valores reales y predicciones con XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# 13. Evaluar el modelo XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_xgb = np.expm1(y_pred_xgb)  # Invertir la transformación logarítmica en las predicciones

# 14. Cálculo de MSE y R² finales para XGBoost
final_mse_xgb = mean_squared_error(y_test_orig, y_pred_xgb)
final_r2_xgb = r2_score(y_test_orig, y_pred_xgb)
print(f"Final Mean Squared Error (MSE) con XGBoost: {final_mse_xgb}")
print(f"Final R-squared (R²) con XGBoost: {final_r2_xgb}")

# ** Gráfica 2: Comparación entre valores reales y predicciones con XGBoost **
plt.scatter(y_test_orig, y_pred_xgb)
plt.xlim(0, 25000)  # Limitar el eje X para ver mejor los valores más bajos
plt.ylim(0, 25000)  # Limitar el eje Y para ver mejor las predicciones más bajas
plt.xlabel('Valores reales (Violencia Doméstica)')
plt.ylabel('Predicciones (Violencia Doméstica)')
plt.title('Comparación entre valores reales y predicciones con XGBoost')
plt.show()

# 15. Análisis de residuos (errores)
# Gráfico de residuos
residuos = y_test_orig - y_pred  # Ahora ambas variables son arrays 1D
plt.scatter(y_pred, residuos)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.show()

# 16. Detección de Outliers (valores extremos)
# Aplicar Isolation Forest para detectar outliers
iso_forest = IsolationForest(contamination=0.05)  # Ajustar el nivel de contaminación/outliers
outliers = iso_forest.fit_predict(X)
# Visualizar los outliers detectados
plt.scatter(X['Rape'], y, c=outliers, cmap='coolwarm')
plt.xlabel('Rape')
plt.ylabel('Violencia Doméstica (DV)')
plt.title('Outliers detectados en los datos')
plt.show()

y_pred = model.predict(X_test_scaled)
plt.plot(y_test.values, label='Valores reales')
plt.plot(y_pred, label='Predicciones')
plt.title('Comparación entre valores reales y predicciones')
plt.legend()
plt.show()

# Calcular MSE final
final_mse = mean_squared_error(y_test, y_pred)
# Calcular R² final
final_r2 = r2_score(y_test, y_pred)

# Visualizar MSE y R² finales
print(f"Final Mean Squared Error (MSE): {final_mse}")
print(f"Final R-squared (R²): {final_r2}")
