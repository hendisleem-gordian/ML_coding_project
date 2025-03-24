import numpy as np
import os

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.training_loss = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        os.makedirs("history", exist_ok=True)
        with open("history/linear_regression_training_log.txt", "a") as log_file:
            log_file.write("\n--- New Training Session ---\n")

            for epoch in range(self.epochs):
                '''y = weights · X + bias'''
                y_pred = np.dot(X, self.weights) + self.bias

                '''Calculating the MSE loss'''
                loss = np.mean((y - y_pred) ** 2)
                self.training_loss.append(loss)

                dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
                db = (-2 / n_samples) * np.sum(y - y_pred)

                '''Update the things'''
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                if (epoch + 1) % 100 == 0 or epoch == self.epochs - 1:
                    log_msg = f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss:.4f}"
                    print(log_msg)
                    log_file.write(log_msg + "\n")

    def predict(self, X):
        X = np.array(X)
        return (np.dot(X, self.weights) + self.bias).flatten().tolist()
