import csv
import numpy as np
from linear_regression import GradientDescentLinearRegression

def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                AoW = float(row[3])
                DV = float(row[6])
                X.append([1, AoW])
                y.append(DV)
            except ValueError:
                print(f"Warning: Could not convert row {row} to floats. Skipping this row.")
            except IndexError:
                print(f"Warning: Row {row} does not have enough columns. Skipping this row.")
    return X, y

def remove_outliers(X, y, threshold=1.5):
    X, y = np.array(X), np.array(y)
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * threshold)
    upper_bound = q3 + (iqr * threshold)
    mask = (y >= lower_bound) & (y <= upper_bound)
    return X[mask], y[mask]

if __name__ == "__main__":
    print("Inicio del script")

    filename = "CrimesOnWomenData.csv"
    X, y = load_data(filename)

    print(f"Cargados {len(X)} filas de X y {len(y)} valores de y")

    if len(X) == 0 or len(y) == 0:
        print("Error: No se pudieron cargar datos válidos. Revisa el archivo CSV y los índices de columna.")
    else:
        X, y = remove_outliers(X, y)
        print(f"Datos después de remover outliers: {len(X)} filas de X y {len(y)} valores de y")

        model = GradientDescentLinearRegression(learning_rate=0.001)  # Ajusta la tasa de aprendizaje
        X_scaled = model.scale_features(X)
        model.fit(X_scaled, y, epochs=1000)
        
        m, b = model.params[1], model.params[0]
        print(f"Pendiente (m): {m}")
        print(f"Intersección (b): {b}")

        r2 = model.r_squared(X_scaled, y)
        print(f"Coeficiente de Determinación (R²): {r2}")

        model.plot_errors()

        y_pred = model.predict(X_scaled)
        model.plot_regression_line(X_scaled, y, y_pred)

        nuevas_predicciones = [[1, 0.5], [1, 0.8]]
        nuevas_predicciones_escaladas = model.scale_features_new(nuevas_predicciones)
        predicciones = model.predict(nuevas_predicciones_escaladas)
        print(f"Predicciones para nuevos datos: {predicciones}")