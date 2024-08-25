import numpy as np
import matplotlib.pyplot as plt

class GradientDescentLinearRegression:
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.params = None
        self.errors = []

    def h(self, X):
        return np.dot(X, self.params)

    def fit(self, X, y, epochs=1000):
        X = np.asarray(X)
        y = np.asarray(y)
        self.params = np.zeros(X.shape[1])
        self.errors = []

        for epoch in range(epochs):
            self.params = self.gradient_descent(X, y)
            y_pred = self.h(X)
            error = np.mean((y - y_pred) ** 2)
            self.errors.append(error)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Error = {error}")

    def gradient_descent(self, X, y):
        m = len(y)
        y_pred = self.h(X)
        gradients = np.dot(X.T, (y_pred - y)) / m
        params_updated = self.params - (self.learning_rate * gradients)
        return params_updated

    def scale_features(self, X):
        X = np.asarray(X)
        self.mean = np.mean(X[:, 1], axis=0)
        self.std = np.std(X[:, 1], axis=0)
        X[:, 1] = (X[:, 1] - self.mean) / self.std
        return X

    def scale_features_new(self, X):
        X = np.asarray(X)
        X[:, 1] = (X[:, 1] - self.mean) / self.std
        return X

    def predict(self, X):
        X = np.asarray(X)
        return self.h(X)

    def plot_errors(self):
        plt.plot(self.errors)
        plt.xlabel('Épocas')
        plt.ylabel('Error')
        plt.title('Error durante el entrenamiento')
        plt.show()

    def plot_regression_line(self, X, y, y_pred):
        X = np.array(X)
        plt.scatter(X[:, 1], y, color='blue', label='Datos reales')
        plt.plot(X[:, 1], y_pred, color='red', label='Línea de Regresión')
        plt.xlabel('AoW (Delitos contra mujeres)')
        plt.ylabel('DV (Violencia Doméstica)')
        plt.title('Regresión Lineal')
        plt.legend()
        plt.text(0.05, 0.9, f"Pendiente (m): {self.params[1]:.2f}\nIntersección (b): {self.params[0]:.2f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        plt.show()

    def r_squared(self, X, y):
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_total)
        return r2