import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0  # Slope
        self.b = 0  # Intercept

    def predict(self, X):
        return self.m * X + self.b

    def fit(self, X, y):
        n = len(X)

        for _ in range(self.iterations):
            y_pred = self.predict(X)

            # Compute Gradients
            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            # Update Parameters
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db
